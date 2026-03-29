# FastAPI backend to receive video chunks from React frontend
# and save them as .webm files on Windows
#
# Install dependencies:
#   pip install fastapi uvicorn python-multipart
#
# for VITE_BACKEND_URL=http://localhost:8000
#
# Run:
#   uvicorn test:app --reload --port 8000

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from datetime import datetime

# ── Folder setup ──────────────────────────────────────────────────────────────

# This creates a "frames" folder in the same directory as main.py
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")

# Create the folder if it doesn't exist yet
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    print("  FastAPI is running on http://localhost:8000")
    print(f"  Saving frames to: {SAVE_DIR}")
    yield
    # Runs on shutdown
    print("─" * 50)


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

# Allow React frontend (Vite runs on 5173 by default) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route ─────────────────────────────────────────────────────────────────────

@app.post("/api/detect")
async def detect(chunk: UploadFile = File(...)):
    """
    Receives a video chunk (blob) from the React frontend every 2 seconds.
    Saves it as a .webm file inside the frames/ folder.
    """

    # Read raw bytes from the uploaded blob
    blob_bytes = await chunk.read()

    # ── Validation ────────────────────────────────────────────────────────────

    # Log what we received in the terminal
    print(f"[RECEIVED] size: {len(blob_bytes)} bytes | type: {chunk.content_type}")

    # Reject empty chunks
    if len(blob_bytes) == 0:
        print("[ERROR] Empty chunk — skipping save")
        return JSONResponse(
            content={"error": "Empty chunk received"},
            status_code=400
        )

    # ── Save file ─────────────────────────────────────────────────────────────

    # Use timestamp in filename so chunks don't overwrite each other
    # Example filename: chunk_20241201_123001_456789.webm
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename  = f"chunk_{timestamp}.webm"

    # os.path.join works correctly on Windows
    save_path = os.path.join(SAVE_DIR, filename)

    # Write bytes to disk
    with open(save_path, "wb") as f:
        f.write(blob_bytes)

    # Confirm the file actually landed on disk and get its size
    saved_size = os.path.getsize(save_path)
    print(f"[SAVED] {save_path} ({saved_size} bytes)")

    # ── Response ──────────────────────────────────────────────────────────────
	
    # Feed the saved webm to your model to process
	
    # Replace this with your actual model inference (this is a dummy response)
    return JSONResponse(content={
        "label":      "REAL",   # "FAKE" or "REAL"
        "confidence": 0.99,     # float between 0.0 and 1.0
  
    })
