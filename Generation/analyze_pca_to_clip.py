import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eegdatasets import EEGDataset
from models.atms import ATMS, extract_id_from_string


# ==========================================
# 1. Settings
# ==========================================

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

DATA_PATH = "/home/moepy/ozakitakuma/data_eeg"

IMG_TEST = "/home/moepy/ozakitakuma/data_image/test_images"

SUBJECT = "sub-01"

FEATURE_DIR = Path("../features")

PCA_TRAIN_PATH = (
    FEATURE_DIR / "ViT-H-14_pca1024_features_train.pt"
)

PCA_TEST_PATH = (
    FEATURE_DIR / "ViT-H-14_pca1024_features_test.pt"
)

CLIP_TEST_PATH = (
    FEATURE_DIR / "ViT-H-14_features_test.pt"
)

PREPROJECTION_TEST_PATH = (
    FEATURE_DIR / "ViT-H-14_preprojection_features_test.pt"
)

ENCODER_PATH = Path(
    "./models/pca1024_20261116_135400/"
    "encoder/sub-01/11-16_13-54/best.pth"
)


# ==========================================
# 2. Load features
# ==========================================

pca_train = torch.load(
    PCA_TRAIN_PATH,
    map_location="cpu",
    weights_only=False,
)

pca_test = torch.load(
    PCA_TEST_PATH,
    map_location="cpu",
    weights_only=False,
)

clip_test = torch.load(
    CLIP_TEST_PATH,
    map_location="cpu",
    weights_only=False,
)

preprojection_test = torch.load(
    PREPROJECTION_TEST_PATH,
    map_location="cpu",
    weights_only=False,
)

# PCA parameters
mu = pca_train["pca_mean"].float()
V = pca_train["pca_components"].float()

# CLIP projection
W = preprojection_test["visual_projection"].float()

# Ground-truth image features
Z_pca_true = pca_test["img_features"].float()
Z_clip_true = clip_test["img_features"].float()

assert mu.shape == (1, 1280)
assert V.shape == (1280, 1024)
assert W.shape == (1280, 1024)

assert Z_pca_true.shape == (200, 1024)
assert Z_clip_true.shape == (200, 1024)


# ==========================================
# 3. PCA1024 -> CLIP1024
# ==========================================

def pca_to_clip(features):

    # PCA1024 -> approximate 1280D
    reconstructed_r = (
        features @ V.T + mu
    )

    # 1280D -> original CLIP1024 space
    reconstructed_clip = reconstructed_r @ W

    return reconstructed_clip


# ==========================================
# 4. Case A: Ground-truth PCA features
# ==========================================

Z_clip_A = pca_to_clip(Z_pca_true)

print("Case A:", Z_clip_A.shape)


# ==========================================
# 5. Case B: EEG predicted PCA features
# ==========================================

dataset = EEGDataset(
    DATA_PATH,
    img_dir_test=IMG_TEST,
    feature_type="clip",
    subjects=[SUBJECT],
    train=False,
    feature_space="clip",
    preloaded_features=pca_test,
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

model = ATMS(outputs_dim=1024)

model.load_state_dict(
    torch.load(
        ENCODER_PATH,
        map_location="cpu",
        weights_only=False,
    )
)

model = model.to(DEVICE)
model.eval()

subject_id = extract_id_from_string(SUBJECT)

eeg_features = []

with torch.no_grad():

    for batch in loader:

        eeg = batch[0].to(DEVICE)

        subject_ids = torch.full(
            (eeg.shape[0],),
            subject_id,
            dtype=torch.long,
            device=DEVICE,
        )

        output = model(eeg, subject_ids)

        eeg_features.append(
            output.detach().float().cpu()
        )

Z_pca_eeg = torch.cat(
    eeg_features,
    dim=0,
)

assert Z_pca_eeg.shape == (200, 1024)

Z_clip_B = pca_to_clip(Z_pca_eeg)

print("Case B:", Z_clip_B.shape)


# ==========================================
# 6. Cosine similarity
# ==========================================

def calculate_cosine(pred, target):

    cosine = F.cosine_similarity(
        pred,
        target,
        dim=1,
    )

    return cosine.mean().item(), cosine.std().item()


mean_A, std_A = calculate_cosine(
    Z_clip_A,
    Z_clip_true,
)

mean_B, std_B = calculate_cosine(
    Z_clip_B,
    Z_clip_true,
)

# Reference: EEG prediction in PCA space
mean_PCA, std_PCA = calculate_cosine(
    Z_pca_eeg,
    Z_pca_true,
)


# ==========================================
# 7. Results
# ==========================================

print("\n========== PCA -> CLIP ANALYSIS ==========")

print("\n[A] Ground-truth PCA -> CLIP")
print(f"Mean cosine: {mean_A:.4f}")
print(f"Std cosine:  {std_A:.4f}")

print("\n[B] EEG-predicted PCA -> CLIP")
print(f"Mean cosine: {mean_B:.4f}")
print(f"Std cosine:  {std_B:.4f}")

print("\n[Reference] EEG vs Image in PCA space")
print(f"Mean cosine: {mean_PCA:.4f}")
print(f"Std cosine:  {std_PCA:.4f}")

print("\n========================================")