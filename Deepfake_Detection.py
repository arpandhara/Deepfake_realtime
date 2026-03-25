#!/usr/bin/env python
# coding: utf-8

# # 🔍 Deepfake Image Detection
# **Binary Classifier — Real vs Fake**  
# Model: `InceptionResnetV1` (VGGFace2 pretrained backbone, fine-tuned classifier head)  
# Supports: GPU training, pause & resume via checkpoints, probabilistic inference
# 
# ---
# ### Notebook Structure
# | Cell | Job |
# |------|-----|
# | 1 | Imports & GPU Setup |
# | 2 | Configuration (all hyperparams in one place) |
# | 3 | Data Loaders |
# | 4 | Model Definition |
# | 5 | Checkpoint Utilities (Pause & Resume) |
# | 6 | Training Loop |
# | 7 | Validation Loop |
# | 8 | ▶ Run Training |
# | 9 | Plot Training Curves |
# | 10 | 🔎 Single Image Inference |
# | 11 | 💾 Save Final Model |

# In[2]:


# ─────────────────────────────────────────────────────────
# CELL 1 — Imports & GPU Setup
# ─────────────────────────────────────────────────────────
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True   # Reproducible GPU ops
torch.backends.cudnn.benchmark    = False   # Set True for speed if input size is fixed

# ── Device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[OK] PyTorch  : {torch.__version__}')
print(f'[OK] Device   : {device}')
if device.type == 'cuda':
    print(f'   GPU Name : {torch.cuda.get_device_name(0)}')
    print(f'   VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('[WARN]  No GPU detected — training will run on CPU (slower)')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 2 — Configuration
# All hyperparameters and paths live here. Adjust as needed.
# ─────────────────────────────────────────────────────────
CONFIG = {
    # ── Dataset paths ──
    'train_dir' : r'C:\Users\SHINJAN\Downloads\deepfake_dataset\real_vs_fake\real-vs-fake\train',
    'valid_dir' : r'C:\Users\SHINJAN\Downloads\deepfake_dataset\real_vs_fake\real-vs-fake\valid',
    'test_dir'  : r'C:\Users\SHINJAN\Downloads\deepfake_dataset\real_vs_fake\real-vs-fake\test',

    # ── Extra fake sources (modern AI images: Midjourney, SD, DALL-E, Flux etc.) ──
    # Drop any folder of AI-generated face images here.
    # Each folder just needs .jpg/.png files (no subfolder structure required).
    # All images are labelled fake (class 0) automatically.
    # Leave as [] if you have no extra fakes yet.
    'extra_fake_dirs'  : [],   # e.g. [r'C:\path\to\midjourney_faces', r'C:\path\to\sd_faces']

    # ── Training ──
    'epochs'           : 15,
    'batch_size'       : 32,
    'learning_rate'    : 1e-4,
    'freeze_backbone'  : True,
    'num_workers'      : 0,           # 0 on Windows; increase on Linux
    'use_class_weights': True,        # auto-balance loss when extra fakes skew the ratio

    # ── Checkpoint ──
    'checkpoint_path'     : 'checkpoint.pt',
    'checkpoint_interval' : 50,

    # ── Output ──
    'model_save_path' : 'deepfake_model_final.pt',
    'image_size'      : 299,
    'num_classes'     : 2,
}

print('[CONFIG] CONFIG loaded:')
for k, v in CONFIG.items():
    print(f'   {k:<25} = {v}')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 2b — HuggingFace Dataset Loader
#
# Loads: riandika/AI-vs-Deepfake-vs-Real-Resized-Aug (~17k images)
# 3 source classes  -->  mapped to our binary labels:
#   AI-generated  -->  fake (0)
#   Deepfake      -->  fake (0)
#   Real          -->  real (1)
#
# Only the 'train' split is merged into training.
# Set USE_HF_DATASET = False to skip this cell entirely.
# ─────────────────────────────────────────────────────────

USE_HF_DATASET = True   # set False to skip
HF_DATASET_ID  = 'riandika/AI-vs-Deepfake-vs-Real-Resized-Aug'

# Populated here; consumed by Cell 3
hf_train_data = None
HF_LABEL_MAP  = {}

if USE_HF_DATASET:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        import subprocess
        print('Installing huggingface datasets...')
        subprocess.check_call([__import__('sys').executable, '-m', 'pip', 'install', 'datasets', '-q'])
        from datasets import load_dataset as hf_load_dataset

    print(f'Downloading {HF_DATASET_ID} ...')
    print('(This may take a few minutes on first run; cached locally afterwards)')
    hf_ds         = hf_load_dataset(HF_DATASET_ID)
    hf_train_data = hf_ds['train']

    # Inspect label names
    feature = hf_train_data.features['label']
    print(f'Label names in HF dataset: {feature.names}')
    print(f'Total HF train samples: {len(hf_train_data):,}')

    # Anything not 'real' (case-insensitive) -> 0 (fake)
    HF_LABEL_MAP = {
        idx: (1 if name.lower() == 'real' else 0)
        for idx, name in enumerate(feature.names)
    }
    print('Binary label mapping:')
    for src_idx, name in enumerate(feature.names):
        mapped = 'real (1)' if HF_LABEL_MAP[src_idx] == 1 else 'fake (0)'
        print(f'   {name:<20} --> {mapped}')
else:
    print('HF dataset skipped (USE_HF_DATASET = False)')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 3 — Data Loaders
#
# Supports two modes:
#   1. Standard: single ImageFolder train/valid/test split
#   2. Multi-source: merges extra_fake_dirs (modern AI images)
#      into the training set, labelled as fake (class 0)
#
# Augmentations tuned to expose diffusion-model artifacts:
#   • RandomJPEGCompression — real photos have JPEG noise;
#     diffusion images are suspiciously clean
#   • GaussianBlur          — catches over-smoothed diffusion outputs
#   • RandomGrayscale       — prevents colour-distribution overfit
# ─────────────────────────────────────────────────────────

import glob
import io
from torch.utils.data import Dataset, ConcatDataset


# ─── Custom transform: simulate JPEG compression ───────────────────────────────────
class RandomJPEGCompression:
    """
    Randomly re-encodes a PIL image as JPEG at a low quality.
    Real photos naturally have JPEG noise; diffusion/AI images often do not.
    This teaches the model to notice that difference.
    """
    def __init__(self, quality_range=(40, 90), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


# ─── Custom Dataset: flat folder of images → all labelled fake (0) ─────────────────
class FlatFakeDataset(Dataset):
    """
    Loads all .jpg/.jpeg/.png/.webp images from a folder and assigns
    them label 0 (fake). Use for Midjourney, Stable Diffusion, DALL-E,
    Flux, or any modern AI-generated image source.
    """
    EXTS = ('*.jpg', '*.jpeg', '*.png', '*.webp')

    def __init__(self, folder: str, transform=None):
        self.transform = transform
        self.paths = []
        for ext in self.EXTS:
            self.paths.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
        if not self.paths:
            print(f'  \u26a0\ufe0f  No images found in: {folder}')
        else:
            print(f'  \U0001f4c2 Extra fakes  {folder}  \u2192  {len(self.paths):,} images')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0   # always fake



# --- HuggingFace Dataset wrapper -------------------------------------------
class HuggingFaceWrapperDataset(Dataset):
    """
    Wraps a HuggingFace dataset split as a PyTorch Dataset.
    label_map: dict mapping HF integer label -> binary label (0=fake, 1=real)
    """
    def __init__(self, hf_dataset, label_map: dict, transform=None):
        self.data      = hf_dataset
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data[idx]
        img   = row['image'].convert('RGB')
        label = self.label_map[row['label']]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Augmentation pipelines ─────────────────────────────────────────────────────

# Train: diffusion-aware augmentation
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    RandomJPEGCompression(quality_range=(40, 90), p=0.4),   # diffusion-aware
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # diffusion-aware
    transforms.RandomGrayscale(p=0.05),                         # prevents colour overfit
    transforms.ToTensor(),
    fixed_image_standardization,
])

# Val / Test: no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    fixed_image_standardization,
])


# ─── Base datasets (ImageFolder) ────────────────────────────────────────────────────
train_dataset_base = datasets.ImageFolder(CONFIG['train_dir'], transform=train_transform)
valid_dataset      = datasets.ImageFolder(CONFIG['valid_dir'], transform=eval_transform)
test_dataset       = datasets.ImageFolder(CONFIG['test_dir'],  transform=eval_transform)

CLASS_NAMES = {v: k for k, v in train_dataset_base.class_to_idx.items()}  # {0:'fake', 1:'real'}


# ─── Merge extra fake sources ────────────────────────────────────────────────────────
extra_fake_count = 0
extra_datasets   = []

if CONFIG['extra_fake_dirs']:
    print('\n[BOX] Loading extra fake image sources...')
    for folder in CONFIG['extra_fake_dirs']:
        ds = FlatFakeDataset(folder, transform=train_transform)
        if len(ds) > 0:
            extra_datasets.append(ds)
            extra_fake_count += len(ds)

train_dataset = ConcatDataset([train_dataset_base] + extra_datasets) if extra_datasets else train_dataset_base

# --- Merge HuggingFace dataset (if loaded in Cell 2b) ---
hf_sample_count = 0
hf_fake_count   = 0
hf_real_count   = 0
if USE_HF_DATASET and hf_train_data is not None and HF_LABEL_MAP:
    hf_torch_ds = HuggingFaceWrapperDataset(
        hf_train_data, HF_LABEL_MAP, transform=train_transform
    )
    hf_sample_count = len(hf_torch_ds)
    # [OK] FIX: Fetch 'label' column entirely. This avoids memory/speed issues associated with HuggingFace dataset dictionary fetching
    hf_labels = hf_train_data['label']
    hf_fake_count = sum(1 for lbl in hf_labels if HF_LABEL_MAP[lbl] == 0)
    hf_real_count = hf_sample_count - hf_fake_count
    if isinstance(train_dataset, ConcatDataset):
        train_dataset = ConcatDataset(list(train_dataset.datasets) + [hf_torch_ds])
    else:
        train_dataset = ConcatDataset([train_dataset, hf_torch_ds])
    print(f'HF dataset merged: {hf_sample_count:,} samples  (fake={hf_fake_count:,}  real={hf_real_count:,})')
else:
    hf_torch_ds = None


# ─── Class weights (inverse-frequency) ────────────────────────────────────────────────
base_fake  = sum(1 for _, lbl in train_dataset_base.samples if lbl == 0)
base_real  = sum(1 for _, lbl in train_dataset_base.samples if lbl == 1)
# Include HF dataset counts in class weight calculation
total_fake = base_fake + extra_fake_count + hf_fake_count
total_real = base_real + hf_real_count
total_all  = total_fake + total_real

class_weights = torch.tensor([
    total_all / (2 * total_fake) if total_fake > 0 else 1.0,   # weight for fake (0)
    total_all / (2 * total_real) if total_real > 0 else 1.0,   # weight for real (1)
], dtype=torch.float).to(device)


# ─── Data loaders ─────────────────────────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=True,  num_workers=CONFIG['num_workers'], pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)


# ─── Summary ──────────────────────────────────────────────────────────────────────────
print(f'\n[DIR] Classes        : {train_dataset_base.class_to_idx}')
print(f'[STATS] Base train     : {len(train_dataset_base):,}  (fake={base_fake:,}  real={base_real:,})')
if extra_fake_count:
    print(f'[+] Extra fakes    : {extra_fake_count:,}')
if hf_sample_count:
    print(f'HF dataset     : {hf_sample_count:,}  (fake={hf_fake_count:,}  real={hf_real_count:,})')
print(f'[STATS] Total train    : {len(train_dataset):,}')
print(f'[STATS] Valid images   : {len(valid_dataset):,}')
print(f'[STATS] Test  images   : {len(test_dataset):,}')
print(f'[WEIGHT]  Class weights  : fake={class_weights[0]:.3f}  real={class_weights[1]:.3f}')
print(f'[LOOP] Batches/epoch  : {len(train_loader)}')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 4 — Model Definition
# InceptionResnetV1 pretrained on VGGFace2
# We replace the final logits layer to output 2 classes.
# Optionally freeze the visual backbone and only train
# the new classifier head (faster, less overfitting).
# ─────────────────────────────────────────────────────────

def build_model(num_classes: int, freeze_backbone: bool, device: torch.device) -> nn.Module:
    """
    Build InceptionResnetV1 with a fresh classification head.
    
    Args:
        num_classes    : Number of output classes (2 for Real/Fake)
        freeze_backbone: If True, backbone weights are frozen — only head trains
        device         : Target compute device
    Returns:
        model on the specified device
    """
    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes
    ).to(device)

    if freeze_backbone:
        # Freeze all layers …
        for param in model.parameters():
            param.requires_grad = False
        # … then unfreeze only the final logits layer
        for param in model.logits.parameters():
            param.requires_grad = True

    return model


model = build_model(
    num_classes=CONFIG['num_classes'],
    freeze_backbone=CONFIG['freeze_backbone'],
    device=device
)

# ── Summary ──
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_    = sum(p.numel() for p in model.parameters())
frozen    = total_ - trainable

print(f'[MODEL] Model      : InceptionResnetV1 (VGGFace2)')
print(f'   Total params    : {total_:,}')
print(f'   Trainable params: {trainable:,}  ← only these update')
print(f'   Frozen params   : {frozen:,}')

# ── Note on unfreezing backbone layers ──
# If you added many extra fake images (>5000), consider training more of
# the backbone so it can learn newer artifact patterns.
# Change freeze_backbone=False in CONFIG to unfreeze everything, or
# selectively unfreeze the last few blocks:
#
#   for name, param in model.named_parameters():
#       if any(b in name for b in ('block8', 'block7', 'logits')):
#           param.requires_grad = True
#
# More trainable params → more capacity to learn modern artifacts,
# but needs more data to avoid overfitting.


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 5 — Checkpoint Utilities  (Pause & Resume)
# 
# To PAUSE: interrupt the kernel. The last auto-save
#           (every CHECKPOINT_INTERVAL batches) is kept.
# To RESUME: just re-run Cell 8. It detects the checkpoint
#            and picks up from the right epoch + batch.
# ─────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler,
                    epoch, batch_idx, history):
    """Persist training state so we can resume later."""
    torch.save({
        'epoch'               : epoch,
        'batch_idx'           : batch_idx,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history'             : history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    """
    Load training state from disk.
    Returns (start_epoch, start_batch, history).
    If no checkpoint exists, returns (0, 0, default_history).
    """
    default_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    if not os.path.exists(path):
        print('[INFO]  No checkpoint found — starting fresh')
        return 0, 0, default_history

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start_epoch = ckpt['epoch']
    start_batch = ckpt['batch_idx']
    history     = ckpt.get('history', default_history)

    print(f'[OK] Checkpoint loaded — resuming at Epoch {start_epoch + 1}, Batch {start_batch}')
    return start_epoch, start_batch, history


print('[OK] Checkpoint utilities ready  (save_checkpoint / load_checkpoint)')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 6 — Training Loop
# 
# For each batch:
#   1. Forward pass  → model predicts labels
#   2. Loss          → CrossEntropyLoss measures how wrong
#   3. Backward pass → gradients tell model which way to adjust
#   4. Optimizer step→ applies the adjustments (Adam)
#   5. Checkpoint    → saved every N batches automatically
# ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler,
                    epoch, start_batch, checkpoint_path, history,
                    checkpoint_interval, device):
    """
    Train for one epoch.
    Returns: average training loss for this epoch.
    """
    model.train()
    running_loss  = 0.0
    batches_done  = 0

    pbar = tqdm(enumerate(loader, start=1), total=len(loader),
                desc=f'Epoch {epoch+1} [train]', leave=True, ascii=True)

    for batch_idx, (images, labels) in pbar:

        # ── Skip already-trained batches when resuming ──
        if batch_idx <= start_batch:
            continue

        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        # ── Forward ──
        optimizer.zero_grad()
        outputs = model(images)          # raw logits [batch, num_classes]

        # ── Loss (CrossEntropy = softmax + negative log likelihood) ──
        loss = loss_fn(outputs, labels)

        # ── Backward (compute gradients) ──
        loss.backward()

        # ── Optimizer step (update weights) ──
        optimizer.step()

        running_loss += loss.item()
        batches_done += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'lr'  : f'{scheduler.get_last_lr()[0]:.2e}'})

        # ── Auto-save checkpoint every N batches ──
        if batch_idx % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                            epoch, batch_idx, history)

    avg_loss = running_loss / max(batches_done, 1)
    return avg_loss


print('[OK] Training loop function defined')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 7 — Validation Loop
# 
# No gradient computation here — pure inference.
# Returns: average val loss, accuracy %
# ─────────────────────────────────────────────────────────

def validate(model, loader, loss_fn, device, split_name='val'):
    """
    Evaluate model on a data loader.
    Returns: (avg_loss, accuracy_percent)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():   # No gradients → saves GPU memory + faster
        pbar = tqdm(loader, desc=f'  [{split_name}]', leave=False, ascii=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs   = model(images)
            loss      = loss_fn(outputs, labels)
            running_loss += loss.item()

            # ── Prediction = class with highest logit ──
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


print('[OK] Validation loop function defined')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 8 — ▶ Run Training
# 
# This cell wires everything together:
#   - Builds optimizer + scheduler + loss function
#   - Loads checkpoint if one exists (RESUME)
#   - Runs the train → validate loop for each epoch
#   - Saves checkpoint at end of each epoch too
#
# To PAUSE: Kernel → Interrupt (Esc + I + I)
# To RESUME: re-run this cell — it picks up automatically
# ─────────────────────────────────────────────────────────

# ── Optimizer: Adam (adaptive learning rate per parameter) ──
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['learning_rate'],
    weight_decay=1e-4
)

# ── Scheduler: Cosine decay — smoothly lowers LR as training progresses ──
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

# ── Loss: CrossEntropyLoss (softmax + NLL) ──
# Use class weights if configured (handles imbalance when extra fakes are added)
if CONFIG['use_class_weights']:
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f'[WEIGHT]  Weighted loss: fake={class_weights[0]:.3f}  real={class_weights[1]:.3f}')
else:
    loss_fn = nn.CrossEntropyLoss()
    print('[WEIGHT]  Unweighted loss')

# ── Load checkpoint (resume if exists, else start fresh) ──
start_epoch, start_batch, history = load_checkpoint(
    CONFIG['checkpoint_path'], model, optimizer, scheduler
)

print(f'\n[START] Starting training for {CONFIG["epochs"]} epochs\n')

# ── Main training loop ──
for epoch in range(start_epoch, CONFIG['epochs']):

    print(f'\n══════════ Epoch {epoch + 1}/{CONFIG["epochs"]} ══════════')

    # Train
    train_loss = train_one_epoch(
        model, train_loader, loss_fn, optimizer, scheduler,
        epoch, start_batch, CONFIG['checkpoint_path'], history,
        CONFIG['checkpoint_interval'], device
    )

    # After the first resumed epoch, start_batch resets for subsequent epochs
    start_batch = 0

    # Validate
    val_loss, val_acc = validate(model, valid_loader, loss_fn, device)

    # Step the LR scheduler once per epoch
    scheduler.step()

    # Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'\n📉 Train Loss : {train_loss:.4f}')
    print(f'📉 Val   Loss : {val_loss:.4f}')
    print(f'[TARGET] Val   Acc  : {val_acc:.2f}%')
    print(f'[BOOKS] LR         : {scheduler.get_last_lr()[0]:.2e}')

    # Save checkpoint at end of every epoch
    save_checkpoint(CONFIG['checkpoint_path'], model, optimizer, scheduler,
                    epoch + 1, 0, history)

print('\n[OK] Training complete!')


# In[ ]:


sns.set_theme(style='darkgrid')
epochs_ran = list(range(1, len(history['train_loss']) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Deepfake Detection — Training Progress', fontsize=15, fontweight='bold')

# ── Loss plot ──
ax1.plot(epochs_ran, history['train_loss'], marker='o', label='Train Loss',  color='#E74C3C')
ax1.plot(epochs_ran, history['val_loss'],   marker='s', label='Val Loss',    color='#3498DB')
ax1.set_title('Loss over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.legend()

# ── Accuracy plot ──
ax2.plot(epochs_ran, history['val_acc'], marker='^', color='#2ECC71', label='Val Accuracy')
ax2.set_title('Validation Accuracy over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 100)
ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('[STATS] Plot saved as training_curves.png')


# In[ ]:


import os
import torchvision.transforms as transforms
from facenet_pytorch import fixed_image_standardization

# ─── Self-Contained Inference Dependencies ───
eval_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    fixed_image_standardization,
])

# Dynamically fetch class names if possible, else default
try:
    CLASS_NAMES = {v: k for k, v in train_dataset_base.class_to_idx.items()}
except NameError:
    CLASS_NAMES = {0: 'fake', 1: 'real'}

def predict_image(image_path: str, model: nn.Module, device: torch.device) -> dict:
    img = Image.open(image_path).convert('RGB')
    tensor = eval_transform(img).unsqueeze(0).to(device)  # [1, 3, 299, 299]
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    class_probs = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))}
    predicted_idx = probs.argmax().item()
    return {
        'label'        : CLASS_NAMES[predicted_idx].upper(),
        'confidence'   : round(probs[predicted_idx].item(), 4),
        'probabilities': class_probs,
    }

TEST_IMAGE_PATH = r'C:\Users\SHINJAN\Downloads\deepfake_dataset\real_vs_fake\real-vs-fake\test\fake\00000.jpg'
if os.path.exists(TEST_IMAGE_PATH):
    result = predict_image(TEST_IMAGE_PATH, model, device)
    print(f'\n[SEARCH] Image        : {os.path.basename(TEST_IMAGE_PATH)}')
    print(f'   Verdict      : {result["label"]}')
    print(f'   Confidence   : {result["confidence"]*100:.1f}%')
    print(f'   All probs    : {result["probabilities"]}')
    img_display = Image.open(TEST_IMAGE_PATH)
    color = '#2ECC71' if result['label'] == 'REAL' else '#E74C3C'
    plt.figure(figsize=(5, 5))
    plt.imshow(img_display)
    plt.title(f'{result["label"]}  ({result["confidence"]*100:.1f}% confident)', fontsize=14, fontweight='bold', color=color)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f'\n[WARN] Test image {TEST_IMAGE_PATH} not found. Change TEST_IMAGE_PATH to test.')


# In[ ]:


# ─────────────────────────────────────────────────────────
# CELL 11 — [SAVE] Save Final Model
# 
# Saves the trained model weights to disk.
# Use this after training is fully done.
# ─────────────────────────────────────────────────────────

os.makedirs('models', exist_ok=True)

final_path = os.path.join('models', CONFIG['model_save_path'])
torch.save(model.state_dict(), final_path)

print(f'[SAVE] Final model saved → {final_path}')
print(f'   To reload later: model.load_state_dict(torch.load("{final_path}", map_location=device))')

