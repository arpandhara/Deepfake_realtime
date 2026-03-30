import os
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
CUSTOM_DIR    = 'extracted_faces/custom/train'  # Our newly extracted faces
ORIGINAL_DIR  = 'extracted_faces/train'         # Original kaggle 67k dataset
MODEL_PATH    = 'models/best_model.pt'          # Starting point
OUTPUT_PATH   = 'models/best_model_tuned.pt'    # Where to save the finetuned model
EPOCHS        = 5                               # Short finetuning
BATCH_SIZE    = 32
LR            = 3e-5                            # Very low learning rate so we don't break the original weights
IMAGE_SIZE    = 299
NUM_FAKE_SAMPLES = 800  # Number of original fake images to mix in
NUM_REAL_SAMPLES = 200  # Number of original real images to mix in

class SubsetDataset(Dataset):
    """Loads a random subset of images from a specific folder."""
    def __init__(self, folder, label, num_samples, transform=None):
        self.folder = folder
        self.label = label
        self.transform = transform
        self.paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.paths.extend(glob.glob(os.path.join(folder, ext)))
        
        # Randomly sample if we have more than needed
        if len(self.paths) > num_samples:
            self.paths = random.sample(self.paths, num_samples)
            
        print(f"Loaded {len(self.paths)} images as label {label} from {folder}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.label

class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(classify=False, pretrained='vggface2')
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.head(self.backbone(x))

def finetune_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Base model not found at {MODEL_PATH}.")
        print("Please train your model in the Jupyter Notebook first before finetuning!")
        return

    # 1. Transform matches the Notebook exactly
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    # 2. Build our Mixed Dataset
    datasets_to_mix = []
    
    # Custom Data (All of it)
    if os.path.exists(os.path.join(CUSTOM_DIR, 'fake')):
        datasets_to_mix.append(SubsetDataset(os.path.join(CUSTOM_DIR, 'fake'), 0, 99999, train_transform))
    if os.path.exists(os.path.join(CUSTOM_DIR, 'real')):
        datasets_to_mix.append(SubsetDataset(os.path.join(CUSTOM_DIR, 'real'), 1, 99999, train_transform))
        
    # Original Data (Subset to keep memory low and prevent catastrophic forgetting)
    if os.path.exists(os.path.join(ORIGINAL_DIR, 'fake')):
        datasets_to_mix.append(SubsetDataset(os.path.join(ORIGINAL_DIR, 'fake'), 0, NUM_FAKE_SAMPLES, train_transform))
    if os.path.exists(os.path.join(ORIGINAL_DIR, 'real')):
        datasets_to_mix.append(SubsetDataset(os.path.join(ORIGINAL_DIR, 'real'), 1, NUM_REAL_SAMPLES, train_transform))

    train_dataset = ConcatDataset(datasets_to_mix)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Total finetuning dataset size: {len(train_dataset)}")

    # 3. Load Model
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded base model: {MODEL_PATH}")

    # Freeze backbone mostly, only train head and last block
    for name, param in model.backbone.named_parameters():
        if 'block8' not in name:  # Keep most of inception frozen
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    use_amp = device.type == 'cuda'

    # 4. Training Loop
    print("\n--- Starting Targeted Finetuning ---")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        correct = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_acc = correct / len(train_dataset)
        print(f"Epoch {epoch} Complete | Acc: {epoch_acc*100:.2f}%")

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"\n✅ Custom Finetuning Complete! Saved tightly-tuned model to: {OUTPUT_PATH}")

if __name__ == '__main__':
    finetune_model()
