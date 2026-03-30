import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import shutil

# --- CONFIGURATION ---
DATA_ROOT     = 'my_data'
OUTPUT_ROOT   = 'extracted_faces/custom'
NUM_FRAMES    = 150  # Max frames to extract from each video
IMAGE_SIZE    = 299

def process_custom_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, margin=20, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7], device=device, post_process=False
    )

    # Clean existing custom extraction to avoid mixing
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    for label in ['real', 'fake']:
        src_dir = os.path.join(DATA_ROOT, label)
        out_dir = os.path.join(OUTPUT_ROOT, 'train', label)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            continue

        all_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        img_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        video_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in video_exts]
        img_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in img_exts]

        # 1. Process Videos
        if video_paths:
            print(f"\n[{label.upper()}] Processing {len(video_paths)} videos...")
            for vpath in video_paths:
                cap = cv2.VideoCapture(vpath)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    cap.release(); continue

                step = max(1, total_frames // NUM_FRAMES)
                vid_name = os.path.basename(vpath).split('.')[0]
                
                count = 0
                pbar = tqdm(total=min(NUM_FRAMES, total_frames), desc=f"  Video: {vid_name}")
                frame_idx = 0
                while cap.isOpened() and count < NUM_FRAMES:
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx % step == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        try:
                            face_t = mtcnn(pil_img)
                            if face_t is not None:
                                face_pil = Image.fromarray(face_t.permute(1,2,0).byte().cpu().numpy())
                                face_pil.save(os.path.join(out_dir, f"{vid_name}_v{frame_idx:05d}.jpg"), quality=95)
                                count += 1
                                pbar.update(1)
                        except: pass
                    frame_idx += 1
                cap.release(); pbar.close()

        # 2. Process Images
        if img_paths:
            print(f"[{label.upper()}] Processing {len(img_paths)} images...")
            for ipath in tqdm(img_paths, desc="  Images"):
                img_name = os.path.basename(ipath).split('.')[0]
                try:
                    img = Image.open(ipath).convert('RGB')
                    face_t = mtcnn(img)
                    if face_t is not None:
                        face_pil = Image.fromarray(face_t.permute(1,2,0).byte().cpu().numpy())
                        face_pil.save(os.path.join(out_dir, f"{img_name}_img.jpg"), quality=95)
                except: pass

    print(f"\nSuccess! Custom data processed in {OUTPUT_ROOT}")
    print("Ready for finetuning.")

if __name__ == '__main__':
    process_custom_data()
