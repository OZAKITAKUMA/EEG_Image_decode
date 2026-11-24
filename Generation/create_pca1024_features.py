from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder_utils import stratified_condition_split


# ==========================================
# 1. 設定
# ==========================================

FEATURE_DIR = Path("../features")

INPUT_PATH = (
    FEATURE_DIR
    / "ViT-H-14_preprojection_features_train.pt"
)

OUTPUT_PATH = (
    FEATURE_DIR
    / "ViT-H-14_pca1024_features_train.pt"
)

SEED = 42
OUTPUT_DIM = 1024


# ==========================================
# 2. Projection前の特徴を読み込む
# ==========================================

saved = torch.load(
    INPUT_PATH,
    map_location="cpu",
    weights_only=False,
)

R = saved["img_features"].double()

print("Original features:", R.shape)

assert R.shape == (16540, 1280)


# ==========================================
# 3. 学習用画像のインデックスを取得
# ==========================================

train_indices, val_indices = stratified_condition_split(
    n_classes=1654,
    conditions_per_class=10,
    trials_per_condition=1,
    val_ratio=0.1,
    seed=SEED,
)

train_indices = torch.as_tensor(
    train_indices,
    dtype=torch.long,
)


# ==========================================
# 4. 学習画像だけでPCAの基底を求める
# ==========================================

R_train = R[train_indices]

mean_R = R_train.mean(
    dim=0,
    keepdim=True,
)

R_centered = R_train - mean_R

print("PCA fitting data:", R_centered.shape)

# 共分散行列に比例する行列
C = R_centered.T @ R_centered

# 固有値分解
eigenvalues, eigenvectors = torch.linalg.eigh(C)

# 固有値が大きい順に並び替える
eigenvalues = eigenvalues.flip(0)
eigenvectors = eigenvectors.flip(1)

# 上位1024方向を取得
V_1024 = eigenvectors[:, :OUTPUT_DIM]

print("PCA basis:", V_1024.shape)


# ==========================================
# 5. 全画像を1024次元に変換
# ==========================================

R_all_centered = R - mean_R

Z_pca = R_all_centered @ V_1024

print("PCA features:", Z_pca.shape)

assert Z_pca.shape == (16540, 1024)


# ==========================================
# 6. 特徴を保存
# ==========================================

output = {
    "img_features": Z_pca.float(),
    "text_features": saved.get("text_features"),
    "feature_type": "clip_pca1024",
    "source": "CLIP preprojection 1280D",
    "pca_dim": OUTPUT_DIM,
    "pca_mean": mean_R.float(),
    "pca_components": V_1024.float(),
    "seed": SEED,
    "val_ratio": 0.1,
}

torch.save(output, OUTPUT_PATH)

print()
print("Saved:", OUTPUT_PATH)


# ==========================================
# 7. 特異値の確認
# ==========================================

singular_values = torch.sqrt(
    eigenvalues.clamp(min=0)
)

singular_values = singular_values[:OUTPUT_DIM]

normalized = (
    singular_values / singular_values[0]
)

print()
print("===== PCA1024 Spectrum =====")

for rank in [1, 200, 400, 600, 700, 800, 900, 1024]:
    print(
        f"Rank {rank:4d}: "
        f"{normalized[rank - 1].item():.6f}"
    )