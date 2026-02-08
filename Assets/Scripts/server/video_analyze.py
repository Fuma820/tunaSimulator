"""
Lightweight video analysis to compute a simple motion-based score.
Returns a float in [0,1] based on average frame-to-frame difference.
If OpenCV is unavailable or the video cannot be read, returns 0.0.
"""
from __future__ import annotations
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from typing import Optional
import torch
import numpy as np
import cv2
import os
import glob

def analyze_video(path: str, max_frames: int = 600) -> float:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return 0.0

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0

    prev_gray: Optional["np.ndarray"] = None
    diffs_sum = 0.0
    diffs_count = 0

    frames_read = 0
    try:
        while frames_read < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                # Normalize to [0,1]
                mean_norm = float(diff.mean()) / 255.0
                diffs_sum += mean_norm
                diffs_count += 1
            prev_gray = gray
    finally:
        cap.release()

    if diffs_count == 0:
        return 0.0

    score = diffs_sum / diffs_count
    # Clamp to [0,1]
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return float(score)

def load_video_frames(video_path, frame_count=16, size=(224, 224)):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    while len(frames) < frame_count and len(frames) > 0:
        frames.append(frames[-1])
    cap.release()
    return frames

def extract_cls_feature(video_path, processor, model, device):
    frames = load_video_frames(video_path)
    if len(frames) == 0:
        return None
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # [CLS]トークン特徴量（1, トークン数, 768）→ (768,)
        cls_feature = outputs.hidden_states[-1][0, 0].cpu().numpy()
    return cls_feature

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def main():
    model_path = "server/finetune_output_tuna_vs_other/checkpoint-207"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEForVideoClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()

    # 全mp4動画を推論
    # test_dir = "videos/test_real_video"
    test_dir = "videos/test_simulator_video"
    video_files = glob.glob(os.path.join(test_dir, "*.mp4"))
    class_names = ["simulator", "real"]

    for target_video in video_files:
        frames = load_video_frames(target_video)
        if len(frames) == 0:
            print(f"Could not extract frames from {target_video}.")
            continue
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            pred_idx = int(np.argmax(logits))
            score = float(torch.softmax(outputs.logits, dim=-1)[0, pred_idx].cpu().numpy())
        print(f"推論結果: {os.path.basename(target_video)} の予測クラス: {class_names[pred_idx]} (スコア: {score:.6f})")

if __name__ == "__main__":
    main()
