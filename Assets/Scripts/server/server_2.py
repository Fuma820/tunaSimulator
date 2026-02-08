import glob
import os

import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEModel


# --------------------------
# 動画を読み込んでフレーム抽出
# --------------------------
def load_video_frames(path, num_frames=16):
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)

    # 均等サンプリング
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
    frames = frames.transpose(0, 3, 1, 2)     # (T, 3, H, W)

    return frames


# --------------------------
# モデルのロード
# --------------------------
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# --------------------------
# 映像 → ベクトル化
# --------------------------
def video_to_vector(video_path):
    frames = load_video_frames(video_path)

    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # last_hidden_state: (batch, num_tokens, hidden_size)
        # トークン方向に平均をとって768次元の特徴ベクトルを得る
        vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

    return vec


# --------------------------
# コサイン類似度
# --------------------------
def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()


# --------------------------
# ディレクトリ内の動画をベクトル化
# --------------------------
def collect_vectors(dir_path):
    video_paths = sorted(glob.glob(os.path.join(dir_path, "*.mp4")))
    if not video_paths:
        print(f"No mp4 files found in {dir_path}")
        return {}

    vectors = {}
    for path in video_paths:
        print(f"Vectorizing {path} ...")
        vec = video_to_vector(path)
        vectors[path] = vec
    return vectors


# --------------------------
# 実行例
# --------------------------
if __name__ == "__main__":
    real_dir = "Assets/Scripts/server/real_video_dataset"
    sim_video = "Assets/Scripts/server/test.mp4"

    real_vectors = collect_vectors(real_dir)
    if not real_vectors:
        print("real_video_dataset 内にベクトル化対象がありませんでした。")
        raise SystemExit(0)

    if not os.path.exists(sim_video):
        print(f"シミュレーション動画が見つかりません: {sim_video}")
        raise SystemExit(0)

    print(f"Vectorizing simulation video: {sim_video}")
    sim_vec = video_to_vector(sim_video)

    similarities = []
    for real_path, real_vec in real_vectors.items():
        similarity = cosine_similarity(real_vec, sim_vec)
        similarities.append(similarity)
        print("=" * 80)
        print(f"Real video: {os.path.basename(real_path)}")
        print(f"  vs {os.path.basename(sim_video)} -> cosine: {similarity:.4f}")

    if similarities:
        sims = np.array(similarities)
        print("=" * 80)
        print("Summary statistics (real vs simulation):")
        print(f"  max : {sims.max():.4f}")
        print(f"  mean: {sims.mean():.4f}")
        print(f"  min : {sims.min():.4f}")
        print(f"  var : {sims.var():.4f}")
