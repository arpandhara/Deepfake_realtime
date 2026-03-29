# 🎯 Using the Trained Model — Real-Time Deepfake Interview Detection

This guide explains how to export your trained model, run it as a backend API, and connect it to a website or browser extension for **real-time interview video analysis**.

---

## 📐 Architecture Overview

```
Browser / Extension
      │
      │  JPEG frame (base64) via WebSocket / HTTP POST
      ▼
┌─────────────────────────┐
│   FastAPI Backend        │  ← Loads your .pt model once at startup
│   (Python, localhost or  │  ← MTCNN detects face in each frame
│    hosted server)        │  ← InceptionResnetV1 classifies: REAL / FAKE
└─────────────────────────┘
      │
      │  JSON: { label, confidence, latency_ms }
      ▼
Website UI / Extension Overlay
```

---

## STEP 1 — Export the Trained Model After Training

Run this in a new notebook cell **after Cell 8 finishes training**:

```python
# ── CELL: Export model for deployment ──────────────────────────────
import torch

# Load best weights into the model (already built in Cell 4)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
model.eval()

# Save as TorchScript (faster inference, no class definition needed at load time)
scripted = torch.jit.script(model)
scripted.save('models/deepfake_model_scripted.pt')

# Also save a plain state dict copy (easier to load in server)
torch.save(model.state_dict(), 'models/deepfake_model_weights.pt')

print("✅ Exported: models/deepfake_model_scripted.pt")
print("✅ Exported: models/deepfake_model_weights.pt")
```

> **Note:** If TorchScript fails (InceptionResnetV1 has some dynamic ops), use the **plain weights** approach in the server below — it works just as well.

---

## STEP 2 — Backend: FastAPI Inference Server

### 2a. Project structure

```
deepfake_main/
├── models/
│   ├── best_model.pt          ← your trained weights
│   └── deepfake_model_weights.pt
├── server/
│   ├── main.py                ← FastAPI app (create below)
│   └── requirements_server.txt
└── Deepfake_Detection.ipynb
```

### 2b. Install server dependencies

```bash
pip install fastapi uvicorn[standard] python-multipart Pillow facenet-pytorch torch torchvision
```

Or create `server/requirements_server.txt`:
```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
Pillow>=10.2.0
torch>=2.2.0
torchvision>=0.17.0
facenet-pytorch>=2.6.0
numpy>=1.26.0
```

### 2c. Create `server/main.py`

```python
"""
Deepfake Detection API — real-time interview frame analysis
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io, base64, time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, MTCNN

# ── Config ────────────────────────────────────────────────────────
WEIGHTS_PATH = "../models/best_model.pt"   # adjust path if needed
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 299
CONFIDENCE_THRESHOLD = 0.60   # below this → "uncertain"

print(f"[SERVER] Device: {DEVICE}")

# ── Model definition (must match Cell 4 exactly) ──────────────────
class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(
            classify=False, pretrained='vggface2'
        ).to(DEVICE)
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        ).to(DEVICE)

    def forward(self, x):
        return self.head(self.backbone(x))

# ── Load model ────────────────────────────────────────────────────
model = DeepfakeClassifier()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f"[SERVER] Model loaded from {WEIGHTS_PATH}")

# ── MTCNN face detector ───────────────────────────────────────────
mtcnn = MTCNN(
    image_size=IMG_SIZE, keep_all=False, min_face_size=40,
    device=DEVICE, post_process=False, margin=20
)

# ── Transform (same as eval_transform in Cell 3) ─────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    fixed_image_standardization,
])

CLASS_NAMES = {0: "FAKE", 1: "REAL"}

# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(title="Deepfake Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameRequest(BaseModel):
    image_b64: str   # base64-encoded JPEG/PNG frame from browser

class PredictionResponse(BaseModel):
    label: str            # "REAL" or "FAKE"
    confidence: float     # 0.0 – 1.0
    uncertain: bool       # True if below threshold
    face_detected: bool
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: FrameRequest):
    t0 = time.perf_counter()

    # 1. Decode base64 → PIL image
    try:
        img_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    # 2. Detect face with MTCNN
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return PredictionResponse(
            label="UNKNOWN", confidence=0.0,
            uncertain=True, face_detected=False,
            latency_ms=round((time.perf_counter()-t0)*1000, 1)
        )

    # 3. Prepare tensor
    face_pil = Image.fromarray(face_tensor.permute(1,2,0).byte().cpu().numpy())
    inp = transform(face_pil).unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = int(torch.argmax(probs).item())
        conf   = float(probs[pred].item())

    label = CLASS_NAMES[pred]
    latency = round((time.perf_counter() - t0) * 1000, 1)

    return PredictionResponse(
        label=label,
        confidence=round(conf, 3),
        uncertain=(conf < CONFIDENCE_THRESHOLD),
        face_detected=True,
        latency_ms=latency
    )
```

### 2d. Run the server

```bash
# From the server/ directory:
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Test it's alive:
```
http://localhost:8000/health   →  {"status":"ok","device":"cuda"}
http://localhost:8000/docs     →  Swagger UI for manual testing
```

---

## STEP 3 — Frontend: Capture Webcam Frames and Send to API

Add this JavaScript to your website (works with any HTML page or React/Next.js app):

```html
<!-- index.html — Minimal real-time deepfake detector UI -->
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Deepfake Interview Detector</title>
</head>
<body>
  <video id="video" autoplay playsinline width="640" height="480"></video>
  <canvas id="canvas" width="640" height="480" style="display:none"></canvas>
  <div id="result">Waiting...</div>

  <script>
    const video  = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const result = document.getElementById('result');
    const ctx    = canvas.getContext('2d');
    const API    = 'http://localhost:8000/predict';

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { video.srcObject = stream; });

    // Send a frame every 1 second (adjust as needed)
    setInterval(async () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL('image/jpeg', 0.8)
                         .replace('data:image/jpeg;base64,', '');
      try {
        const res  = await fetch(API, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_b64: b64 })
        });
        const data = await res.json();
        const color = data.label === 'REAL' ? 'green' : 'red';
        result.innerHTML = `
          <span style="color:${color}; font-size:2em; font-weight:bold">
            ${data.label} (${(data.confidence * 100).toFixed(1)}%)
          </span>
          <br>Face: ${data.face_detected} | Latency: ${data.latency_ms}ms
          ${data.uncertain ? '<br>⚠️ Low confidence' : ''}
        `;
      } catch(e) {
        result.textContent = 'API error: ' + e.message;
      }
    }, 1000);
  </script>
</body>
</html>
```

---

## STEP 4 — Browser Extension (for Google Meet / Zoom etc.)

A Chrome extension can **overlay the prediction** directly on top of a Google Meet or Zoom tab.

### File structure
```
deepfake_extension/
├── manifest.json
├── content.js       ← injects into Google Meet page
└── popup.html       ← extension popup (optional)
```

### `manifest.json`
```json
{
  "manifest_version": 3,
  "name": "Deepfake Interview Detector",
  "version": "1.0",
  "permissions": ["activeTab", "scripting"],
  "host_permissions": ["https://meet.google.com/*"],
  "content_scripts": [
    {
      "matches": ["https://meet.google.com/*"],
      "js": ["content.js"]
    }
  ]
}
```

### `content.js`
```javascript
// Runs on Google Meet pages — captures the remote video stream
const API = 'http://localhost:8000/predict';

function findRemoteVideo() {
  // Google Meet uses <video> elements; pick the largest (main speaker)
  const videos = Array.from(document.querySelectorAll('video'));
  return videos.sort((a,b) => (b.videoWidth*b.videoHeight) - (a.videoWidth*a.videoHeight))[0];
}

function createOverlay() {
  const el = document.createElement('div');
  el.id = 'deepfake-overlay';
  el.style.cssText = `
    position:fixed; top:12px; right:12px; z-index:99999;
    background:rgba(0,0,0,0.75); color:white;
    padding:8px 16px; border-radius:8px; font-size:16px;
    font-family:monospace; pointer-events:none;
  `;
  el.textContent = '🔍 Deepfake Detector Loading...';
  document.body.appendChild(el);
  return el;
}

async function analyzeFrame(video, overlay) {
  if (!video || video.readyState < 2) return;
  const canvas = document.createElement('canvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];

  try {
    const res  = await fetch(API, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image_b64: b64})
    });
    const d = await res.json();
    const emoji = d.label === 'REAL' ? '✅' : '🚨';
    const color = d.label === 'REAL' ? '#00ff88' : '#ff4444';
    overlay.style.borderLeft = `4px solid ${color}`;
    overlay.innerHTML = `${emoji} <b style="color:${color}">${d.label}</b> ${(d.confidence*100).toFixed(0)}%`
      + (d.uncertain ? ' ⚠️' : '')
      + `<br><small>${d.latency_ms}ms</small>`;
  } catch (e) {
    overlay.textContent = '❌ API offline';
  }
}

// Wait for Meet to load, then start
setTimeout(() => {
  const overlay = createOverlay();
  setInterval(() => {
    const video = findRemoteVideo();
    analyzeFrame(video, overlay);
  }, 1500);   // analyze every 1.5 seconds
}, 4000);
```

### Load the extension in Chrome
1. Go to `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **"Load unpacked"** → select the `deepfake_extension/` folder
4. Open Google Meet — the overlay will appear in the top-right corner

---

## STEP 5 — Deployment Options

| Mode | How | When to Use |
|---|---|---|
| **Local** | `uvicorn main:app --port 8000` | Development & demos |
| **Docker** | See Dockerfile below | Portable, shareable |
| **Cloud (Render/Railway)** | Push to GitHub → connect to Render | Hosting for others |
| **ngrok tunnel** | `ngrok http 8000` → share public URL | Quick demo without hosting |

### Dockerfile (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY server/requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt
COPY server/main.py .
COPY models/best_model.pt ./models/best_model.pt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t deepfake-api .
docker run -p 8000:8000 --gpus all deepfake-api
```

---

## STEP 6 — Performance Tuning Tips

| Tip | Impact |
|---|---|
| Analyze every **1–2 seconds** (not every frame) | Reduces API load, GPU stays cool |
| Send frames at **480p max** (not 1080p) | ~4× smaller payload, faster inference |
| Use `torch.compile(model)` (PyTorch 2.x) | 10-30% faster inference |
| Run MTCNN on CPU, model on GPU | Frees GPU VRAM for model |
| Add a **moving average** over last 5 predictions | Smoother, less flickery output |

### Moving average example (add to `content.js` or `index.html`)
```javascript
const history = [];
function smoothPrediction(label, conf) {
  history.push({ label, conf });
  if (history.length > 5) history.shift();
  const fakeCount = history.filter(h => h.label === 'FAKE').length;
  return fakeCount >= 3 ? 'FAKE' : 'REAL';   // majority vote over last 5
}
```

---

## Quick Start Checklist

```
[ ] Training complete → best_model.pt exists in models/
[ ] pip install fastapi uvicorn[standard] facenet-pytorch ...
[ ] Create server/main.py (copy code from Step 2c)
[ ] Run: uvicorn main:app --port 8000
[ ] Check: http://localhost:8000/health
[ ] Open index.html in browser — grant webcam permission
[ ] See REAL / FAKE label update every second
[ ] (Optional) Load Chrome extension for Google Meet detection
```

---

*Model trained on DFD video dataset + InceptionResnetV1 (VGGFace2) backbone.*  
*RTX 4050 GPU inference: ~15–30ms per frame including face detection.*
