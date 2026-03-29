# 🔍 Deepfake Image Detection
**Binary Classifier — Real vs Fake**

This project implements a high-performance deepfake detection system using the `InceptionResnetV1` architecture (pretrained on VGGFace2). It is designed to distinguish between real human faces and AI-generated/deepfake images, supporting modern AI sources like Midjourney, Stable Diffusion, and DALL-E.

---

## 🚀 Key Features
- **Modern AI Support**: Custom data loaders for Midjourney, SD, DALL-E, and Flux images.
- **HuggingFace Integration**: Seamlessly merges the `riandika/AI-vs-Deepfake-vs-Real-Resized-Aug` dataset.
- **GPU Acceleration**: Fully optimized for NVIDIA GPUs (RTX 40 series) with CUDA 12.1.
- **Pause & Resume**: Training state is automatically saved to `checkpoint.pt`, allowing you to interrupt and resume anytime.
- **Diffusion-Aware Augmentation**: Specialized transforms (JPEG compression, Gaussian Blur) to catch subtle AI artifacts.
- **Rich Visualization**: Automated plotting of training loss and validation accuracy.

---

## 📂 Project Structure
| File/Folder | Description |
|:---|:---|
| `Deepfake_Detection.ipynb` | Main Jupyter Notebook for training and inference. |
| `Deepfake_Detection.py` | Python script version of the detection pipeline. |
| `requirements.txt` | List of Python dependencies. |
| `checkpoint.pt` | Saved training state (epoch, optimizer, model weights). |
| `models/` | Directory where final trained models are saved. |
| `training_curves.png` | Visualization of the latest training run. |

---

## 🛠️ Prerequisites
- **Python**: 3.12 (Recommended via Conda)
- **Conda**: For environment management (Miniconda or Anaconda)
- **GPU**: NVIDIA GPU with CUDA 12.1 support (Standard training runs on CPU but is much slower)

---

## ⚙️ Installation & Setup

### 1. Clone the Project
```bash
# Navigate to your project folder
cd C:\Users\Downloads\deepfake_main
```

### 2. Create a Virtual Environment (Conda)
It is highly recommended to use a dedicated environment to avoid package conflicts.
```bash
conda create -n deepfake_env python=3.12 -y
conda activate deepfake_env
```

### 3. Install Standard Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Dedicated GPU Installation (NVIDIA)
To leverage your **RTX 40-series GPU (e.g., RTX 4050)**, follow these specific steps to install PyTorch with CUDA 12.1 support.

### 1. Install PyTorch with CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Verify GPU Recognition
Run the following Python snippet in your terminal:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```
*If it returns `True`, your GPU is ready.*

### 3. Troubleshooting
- Ensure you have the latest **NVIDIA Drivers** installed from [nvidia.com](https://www.nvidia.com/Download/index.aspx).
- If you encounter a `DLL load failed` error, try installing the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

## 🎯 Usage Guide

### 1. Configure Paths
Open `Deepfake_Detection.py` or the Notebook and adjust the `CONFIG` dictionary:
```python
CONFIG = {
    'train_dir' : r'C:\path\to\your\train_data',
    'valid_dir' : r'C:\path\to\your\valid_data',
    'test_dir'  : r'C:\path\to\your\test_data',
    # ... other settings
}
```

### 2. Run Training
**Using Jupyter Notebook:**
Simply open `Deepfake_Detection.ipynb` in VS Code or Jupyter Lab and run all cells.

**Using Terminal:**
```bash
python Deepfake_Detection.py
```

### 3. Pause & Resume
- **To Pause**: Simply interrupt the execution (Ctrl+C in terminal or Stop in Notebook).
- **To Resume**: Re-run the script or training cell. It will detect `checkpoint.pt` and pick up where it left off.

---

## 🔎 Single Image Inference
To test an image, use the provided `predict_image` function in the notebook:
```python
result = predict_image("path/to/test_image.jpg", model, device)
print(f"Verdict: {result['label']} ({result['confidence']*100:.2f}%)")
```

---

## 📊 Results
After training, check `training_curves.png` to see how the model's loss and accuracy improved over time. 

Final models are stored in the `/models` directory with a `.pt` extension.
