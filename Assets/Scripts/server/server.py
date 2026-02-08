from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import tempfile
from typing import Optional
import asyncio
import time
import logging
from datetime import datetime
import threading
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - environment may lack torch
    torch = None

try:
    import cv2
except Exception:  # pragma: no cover - environment may lack opencv
    cv2 = None

try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - transformers may be missing
    VideoMAEImageProcessor = None
    VideoMAEForVideoClassification = None
    TRANSFORMERS_AVAILABLE = False

"""SAM2 video segmentation support is disabled; classify raw clips directly."""
SAM2_AVAILABLE = False
build_sam2_video_predictor_hf = None
SAM2AutomaticMaskGenerator = None

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fish Agent Reward Server", version="1.0")

# CORS設定 - Unity からのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル設定
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
UPLOAD_TIMEOUT = 120  # 120秒

# ログファイル設定
LOG_DIR = "video_analysis_logs"
SCORE_LOG_FILE = os.path.join(LOG_DIR, "video_scores.txt")

# ログディレクトリを作成
os.makedirs(LOG_DIR, exist_ok=True)

"""SAM2 segmentation parameters disabled
SEGMENT_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-large")
SEGMENT_MAX_WIDTH = int(os.getenv("SAM2_MAX_WIDTH", "640"))
SEGMENT_BACKGROUND_RATIO = float(os.getenv("SAM2_BACKGROUND_RATIO", "0.1"))
SEGMENT_OBJECT_COLOR = (0, 255, 0)
"""

VIDEO_CLASS_MODEL_PATH = os.getenv(
    "VIDEO_CLASS_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "finetune_output_tuna_vs_other/checkpoint-825"),
)
VIDEO_CLASS_BASE_MODEL = os.getenv("VIDEO_CLASS_BASE_MODEL", "MCG-NJU/videomae-base")
VIDEO_CLASS_FRAME_COUNT = int(os.getenv("VIDEO_CLASS_FRAME_COUNT", "16"))
VIDEO_CLASS_NAMES = ["simulator1","simulator2","simulator3", "real"]

# モデルキャッシュとロック
sam2_predictor = None
sam2_mask_generator = None
sam2_device = None
sam2_init_lock = threading.Lock()
sam2_infer_lock = threading.Lock()

video_processor = None
video_classifier = None
video_device = None
video_init_lock = threading.Lock()
video_infer_lock = threading.Lock()


def get_device_summary() -> dict:
    """現在利用可能なGPU/デバイス情報を返す"""
    summary = {
        "torch_available": torch is not None,
        "cuda_available": False,
        "gpu_name": None,
        "sam2_device": str(sam2_device) if sam2_device is not None else None,
        "video_classifier_device": str(video_device) if video_device is not None else None,
    }

    if torch is None:
        return summary

    cuda_available = torch.cuda.is_available()
    summary["cuda_available"] = cuda_available

    if cuda_available:
        try:
            summary["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            summary["gpu_name"] = f"cuda device (name unavailable: {exc})"

    return summary

def log_video_received(
    episode_number: int,
    file_size: int,
    attempt_number: int = 1,
    filename: str | None = None,
):
    """受信した映像のメタ情報をスコアログに残す"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"{timestamp} | Episode: {episode_number:4d} | Attempt: {attempt_number} | "
            f"Size: {file_size/1024/1024:6.2f}MB | Event: RECEIVED"
        )
        if filename:
            entry += f" | File: {filename}"
        entry += "\n"

        with open(SCORE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.info(
            "Log entry added for received video (episode=%s, attempt=%s, size=%.2fMB)",
            episode_number,
            attempt_number,
            file_size / 1024 / 1024,
        )
    except Exception as exc:
        logger.error(f"Failed to log video reception: {exc}")


def log_video_score(episode_number: int, file_size: int, score: float, analysis_time: float, attempt_number: int = 1):
    """映像スコアをテキストファイルに記録"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"{timestamp} | Episode: {episode_number:4d} | "
            f"Attempt: {attempt_number} | Size: {file_size/1024/1024:6.2f}MB | "
            f"Score: {score:6.4f} | Analysis: {analysis_time:6.2f}s\n"
        )
        
        # ログファイルに追記
        with open(SCORE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        logger.info(f"Score logged: Episode {episode_number}, Score {score:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to log score: {e}")

def initialize_log_file():
    """ログファイルの初期化（ヘッダー書き込み）"""
    if not os.path.exists(SCORE_LOG_FILE):
        try:
            with open(SCORE_LOG_FILE, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("Fish Agent Video Analysis Score Log\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
                f.write("Timestamp           | Episode | Attempt | Size(MB) | Score  | Analysis(s)\n")
                f.write("-" * 80 + "\n")
            logger.info(f"Initialized score log file: {SCORE_LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to initialize log file: {e}")


"""SAM2 segmentation helpers are disabled.

def get_sam2_components():
    ...

def classify_masks_by_area(...):
    ...

def segment_uploaded_video(...):
    ...

"""


def get_video_classifier_components():
    if not TRANSFORMERS_AVAILABLE or torch is None:
        raise RuntimeError("Video classifier dependencies are not available")

    global video_processor, video_classifier, video_device
    with video_init_lock:
        if video_classifier is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading VideoMAE classifier on {device}...")
            video_processor = VideoMAEImageProcessor.from_pretrained(VIDEO_CLASS_BASE_MODEL)
            video_classifier = VideoMAEForVideoClassification.from_pretrained(
                VIDEO_CLASS_MODEL_PATH,
                local_files_only=True,
            )
            video_classifier.to(device)
            video_classifier.eval()
            video_device = device
            logger.info("Video classifier ready")

    return video_processor, video_classifier, video_device


def load_video_frame_sequences(
    video_path: str,
    frame_count: int = VIDEO_CLASS_FRAME_COUNT,
    sequence_count: int = 4,
    size: tuple[int, int] = (224, 224),
):
    if cv2 is None:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    processed_frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(frame)
    finally:
        cap.release()

    if not processed_frames:
        return []

    total_frames = len(processed_frames)

    if total_frames < frame_count:
        padding_frame = processed_frames[-1]
        processed_frames.extend([padding_frame] * (frame_count - total_frames))
        total_frames = len(processed_frames)

    if sequence_count <= 1 or total_frames <= frame_count:
        start_indices = [0]
    else:
        max_start = max(total_frames - frame_count, 0)
        if sequence_count == 1 or max_start == 0:
            start_indices = [0]
        else:
            start_indices = [
                int(round(i * max_start / (sequence_count - 1)))
                for i in range(sequence_count)
            ]

    sequences: list[list[np.ndarray]] = []
    for start in start_indices:
        end = start + frame_count
        seq = processed_frames[start:end]
        if len(seq) < frame_count and seq:
            seq.extend([seq[-1]] * (frame_count - len(seq)))
        sequences.append(seq)

    return sequences


def classify_segmented_video(video_path: str) -> float:
    if not TRANSFORMERS_AVAILABLE or torch is None:
        logger.warning("Video classifier not available; returning 0 score")
        return 0.0

    frame_sequences = load_video_frame_sequences(
        video_path,
        frame_count=VIDEO_CLASS_FRAME_COUNT,
        sequence_count=4,
    )
    if not frame_sequences:
        logger.warning("No frame sequences available for classification; returning 0 score")
        return 0.0

    try:
        processor, classifier, device = get_video_classifier_components()
    except Exception as exc:
        logger.error(f"Failed to prepare video classifier: {exc}")
        return 0.0

    with video_infer_lock:
        try:
            scores: list[float] = []
            for frames in frame_sequences:
                if not frames:
                    continue
                inputs = processor(frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = classifier(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                if "real" in VIDEO_CLASS_NAMES:
                    real_idx = VIDEO_CLASS_NAMES.index("real")
                    score = float(probs[0, real_idx].cpu().item())
                else:
                    score = float(probs.max().cpu().item())
                scores.append(score)

            if scores:
                return float(sum(scores) / len(scores))

            logger.warning("Video classifier produced no scores; returning 0")
            return 0.0
        except Exception as exc:
            logger.error(f"Video classification failed: {exc}")
            return 0.0


initialize_log_file()


class SpeedRequest(BaseModel):
    speed: float = Field(..., ge=0.0, description="Current agent speed magnitude")
    min_speed: float = Field(..., ge=0.0, description="Minimum desired speed threshold")
    penalty: float = Field(-0.1, description="Negative reward to apply when below min_speed")

class RewardResponse(BaseModel):
    reward: float
    reason: Optional[str] = None

# --- Episode finalize ---
class FinalizeRequest(BaseModel):
    episode_id: Optional[int] = None
    average_speed: Optional[float] = None

class FinalizeResponse(BaseModel):
    status: str = "done"

# --- Video upload ---
class UploadResponse(BaseModel):
    status: str
    filename: str | None = None
    episode_number: int | None = None
    reward: float = 0.0


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント - Unity側の接続テスト用"""
    return {
        "status": "ok", 
        "message": "Server is running",
        "timestamp": time.time(),
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "timeout_seconds": UPLOAD_TIMEOUT,
        "gpu": get_device_summary(),
    }

@app.post("/upload/video", response_model=UploadResponse)
async def upload_video(
    episode_number: int = Form(...),
    file: UploadFile = File(...),
    attempt_number: int = Form(default=1),
    file_size: int = Form(default=0)
):
    """改善された動画アップロードエンドポイント"""
    try:
        start_time = time.time()
        logger.info(f"Receiving upload: episode={episode_number}, attempt={attempt_number}, size={file_size}")
        
        # ファイルサイズチェック
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})"
            )
        
        # ファイルタイプチェック
        if file.content_type and not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}"
            )
        
        # 動画は永続保存せず、一時ファイルにストリーム書き込み→解析後に削除
        tmp_path = None
        total_size = 0
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp_path = tmp.name
                # チャンクで読み取り（メモリ使用量を抑える）
                while True:
                    try:
                        # タイムアウト付きでチャンク読み込み
                        chunk = await asyncio.wait_for(file.read(1024 * 1024), timeout=10)
                        if not chunk:
                            break
                        tmp.write(chunk)
                        total_size += len(chunk)
                        
                        # サイズ制限チェック
                        if total_size > MAX_FILE_SIZE:
                            raise HTTPException(status_code=413, detail="File too large during upload")
                            
                    except asyncio.TimeoutError:
                        raise HTTPException(status_code=408, detail="File upload timeout")

            logger.info(f"File saved to temp: {tmp_path}, size: {total_size} bytes")
            log_video_received(
                episode_number=episode_number,
                file_size=total_size,
                attempt_number=attempt_number,
                filename=file.filename,
            )

            reward_value = 0.0
            classification_target = tmp_path
            analysis_start = time.time()
            try:
                logger.info("SAM segmentation disabled; classifying raw clip directly")
                reward_value = await asyncio.to_thread(classify_segmented_video, classification_target)
            except Exception as e:
                logger.error(f"Video processing pipeline failed: {e}")
                reward_value = 0.0
            finally:
                analysis_time = time.time() - analysis_start
                logger.info(
                    "Video analysis completed (target=raw): score=%.4f, time=%.2fs",
                    reward_value,
                    analysis_time,
                )
                log_video_score(episode_number, total_size, reward_value, analysis_time, attempt_number)
            
            upload_duration = time.time() - start_time
            logger.info(f"Upload processed successfully in {upload_duration:.2f}s")
            
            # ファイル名は返さない（サーバには残さないため）
            return UploadResponse(
                status="ok", 
                filename=None, 
                episode_number=episode_number, 
                reward=reward_value
            )
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.info(f"Temp file cleaned up: {tmp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 新しいエンドポイント: サーバー状態確認用
@app.get("/status")
async def get_server_status():
    """サーバー状態確認用"""
    # ログファイルの統計情報を取得
    log_stats = {"total_episodes": 0, "log_file_exists": False, "log_file_size": 0}
    try:
        if os.path.exists(SCORE_LOG_FILE):
            log_stats["log_file_exists"] = True
            log_stats["log_file_size"] = os.path.getsize(SCORE_LOG_FILE)
            # エピソード数をカウント（簡易）
            with open(SCORE_LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                log_stats["total_episodes"] = len([line for line in lines if "Episode:" in line])
    except Exception as e:
        logger.warning(f"Failed to read log stats: {e}")
    
    return {
        "server": "Fish Agent Reward Server",
        "status": "running",
        "uptime": time.time(),
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "timeout_seconds": UPLOAD_TIMEOUT,
        "log_directory": LOG_DIR,
        "score_log_file": SCORE_LOG_FILE,
        "log_statistics": log_stats,
        "gpu": get_device_summary(),
        "endpoints": ["/health", "/upload/video", "/status", "/logs"]
    }

@app.get("/logs")
async def get_recent_logs(lines: int = 50):
    """最近のログエントリを取得"""
    try:
        if not os.path.exists(SCORE_LOG_FILE):
            return {"message": "No log file found", "logs": []}
        
        with open(SCORE_LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
        return {
            "total_lines": len(all_lines),
            "showing_lines": len(recent_lines),
            "logs": [line.strip() for line in recent_lines if line.strip()]
        }
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


# For local testing: uvicorn server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    print("Starting Fish Agent Reward Server...")
    print(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB")
    print(f"Upload timeout: {UPLOAD_TIMEOUT} seconds")
    print(f"Score log file: {SCORE_LOG_FILE}")
    print("Available endpoints:")
    print("  GET  /health  - Health check")
    print("  GET  /status  - Server status")
    print("  POST /upload/video - Video upload and analysis")
    
    # ログファイルを初期化
    initialize_log_file()
    
    # 直接実行時に起動（reloadはOFF：importパス不要で安定）
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        timeout_keep_alive=UPLOAD_TIMEOUT,
        access_log=True
    )

