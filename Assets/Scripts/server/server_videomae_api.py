import asyncio
import glob
import os
import tempfile
import threading
import time
import cv2
from typing import Dict, List

import numpy as np
import torch
from decord import VideoReader, cpu
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import VideoMAEImageProcessor, VideoMAEModel

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
REAL_VIDEO_DIR = os.getenv(
    "REAL_VIDEO_DATASET_DIR",
    "grayscale_real_video_dataset",
)
MODEL_NAME = os.getenv("VIDEOMAE_MODEL_NAME", "MCG-NJU/videomae-base")
NUM_FRAMES = int(os.getenv("VIDEOMAE_FRAME_COUNT", "16"))
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
UPLOAD_TIMEOUT = 120  # seconds

# --------------------------------------------------------------------------------------
# Global state (model + dataset cache)
# --------------------------------------------------------------------------------------
_processor: VideoMAEImageProcessor | None = None
_model: VideoMAEModel | None = None
_device: torch.device | None = None
_model_lock = threading.Lock()

_dataset_vectors: Dict[str, torch.Tensor] = {}
_dataset_signature: tuple | None = None
_dataset_lock = threading.Lock()


# --------------------------------------------------------------------------------------
# Video helpers
# --------------------------------------------------------------------------------------
def load_video_frames(path: str, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Load evenly sampled frames from a video using decord."""
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"Video {path} has no frames")

    if total_frames < num_frames:
        indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = vr.get_batch(indices).asnumpy()
    frames = frames.transpose(0, 3, 1, 2)
    return frames


def ensure_model_loaded() -> None:
    """Lazy-load the VideoMAE processor and model once."""
    global _processor, _model, _device
    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        model = VideoMAEModel.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()

        _processor = processor
        _model = model
        _device = device


def video_to_vector(video_path: str) -> torch.Tensor:
    ensure_model_loaded()
    assert _processor is not None and _model is not None and _device is not None

    frames = load_video_frames(video_path, NUM_FRAMES)
    frames = grayscale_opencv(frames)

    inputs = _processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
    return vec


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()


def _scan_dataset() -> List[str]:
    return sorted(glob.glob(os.path.join(REAL_VIDEO_DIR, "*.mp4")))


def ensure_real_vectors() -> Dict[str, torch.Tensor]:
    """Load and cache vectors for the real video dataset."""
    global _dataset_vectors, _dataset_signature

    video_paths = _scan_dataset()
    signature = tuple((path, os.path.getmtime(path)) for path in video_paths)

    with _dataset_lock:
        if not video_paths:
            _dataset_vectors = {}
            _dataset_signature = signature
            return _dataset_vectors

        if _dataset_signature == signature and _dataset_vectors:
            return _dataset_vectors

        vectors: Dict[str, torch.Tensor] = {}
        for path in video_paths:
            vectors[path] = video_to_vector(path)

        _dataset_vectors = vectors
        _dataset_signature = signature
        return _dataset_vectors

def grayscale_opencv(frames: np.ndarray) -> np.ndarray:
    """
    frames: (T, 3, H, W) RGB
    return: (T, 3, H, W) grayscale (OpenCV方式)
    """
    gray_frames = []

    for frame in frames:
        # (C, H, W) → (H, W, C)
        frame_hwc = frame.transpose(1, 2, 0)

        # RGB → BGR（OpenCV用）
        frame_bgr = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)

        # BGR → Gray（あなたのコードと完全一致）
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1ch → 3ch
        gray_3ch = np.stack([gray, gray, gray], axis=-1)

        # (H, W, C) → (C, H, W)
        gray_chw = gray_3ch.transpose(2, 0, 1)

        gray_frames.append(gray_chw)

    return np.stack(gray_frames, axis=0).astype(frames.dtype)

# --------------------------------------------------------------------------------------
# FastAPI setup
# --------------------------------------------------------------------------------------
app = FastAPI(title="VideoMAE Similarity Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
async def health_check():
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    dataset_files = _scan_dataset()
    return {
        "status": "ok",
        "gpu": {
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
        },
        "dataset_count": len(dataset_files),
        "model": MODEL_NAME,
        "frame_count": NUM_FRAMES,
        "real_video_dir": os.path.abspath(REAL_VIDEO_DIR),
    }


@app.post("/upload/video")
async def upload_video(
    episode_number: int = Form(...),
    file: UploadFile = File(...),
    attempt_number: int = Form(default=1),
    file_size: int = Form(default=0),
):
    start_time = time.time()

    # file type / size validation
    if file_size and file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds server limit")
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")

    # persist temporary file
    tmp_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
            while True:
                try:
                    chunk = await asyncio.wait_for(file.read(1024 * 1024), timeout=10)
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=408, detail="Upload timed out") from None

                if not chunk:
                    break
                tmp.write(chunk)
                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds server limit")

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty video upload")

        # ensure dataset vectors cached (offload heavy work)
        real_vectors = await asyncio.to_thread(ensure_real_vectors)
        if not real_vectors:
            raise HTTPException(status_code=500, detail="real_video_dataset has no mp4 files")

        # vectorize uploaded clip off the main thread
        sim_vec = await asyncio.to_thread(video_to_vector, tmp_path)

        similarities = [cosine_similarity(vec, sim_vec) for vec in real_vectors.values()]
        sims_np = np.array(similarities, dtype=np.float32)
        score = float(sims_np.mean())

        duration = time.time() - start_time
        print(
            f"[VideoMAE] Episode={episode_number} Attempt={attempt_number} Score={score:.4f}"
        )
        return {
            "status": "ok",
            "episode_number": episode_number,
            "attempt_number": attempt_number,
            "score": score,
            "stats": {
                "count": len(similarities),
                "max": float(sims_np.max()),
                "min": float(sims_np.min()),
                "mean": score,
                "var": float(sims_np.var()),
            },
            "processing_seconds": duration,
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    print("Starting VideoMAE similarity server...")
    print(f"Dataset directory: {os.path.abspath(REAL_VIDEO_DIR)}")
    uvicorn.run(
        "server_videomae_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
