import torch
from pathlib import Path


# =====================================
# 1. File paths
# =====================================

FEATURE_DIR = Path("../features")

TRAIN_PCA_PATH = (
    FEATURE_DIR / "ViT-H-14_pca1024_features_train.pt"
)

TEST_PREPROJECTION_PATH = (
    FEATURE_DIR / "ViT-H-14_preprojection_features_test.pt"
)

OUTPUT_PATH = (
    FEATURE_DIR / "ViT-H-14_pca1024_features_test.pt"
)


# =====================================
# 2. Load features
# =====================================

train_pca = torch.load(
    TRAIN_PCA_PATH,
    map_location="cpu",
    weights_only=False,
)

test_preprojection = torch.load(
    TEST_PREPROJECTION_PATH,
    map_location="cpu",
    weights_only=False,
)


# =====================================
# 3. Get PCA parameters
# =====================================

pca_mean = train_pca["pca_mean"].float()

pca_components = train_pca["pca_components"].float()

R_test = test_preprojection["img_features"].float()


print("R_test:", R_test.shape)
print("PCA mean:", pca_mean.shape)
print("PCA components:", pca_components.shape)

assert R_test.shape == (200, 1280)
assert pca_mean.shape == (1, 1280)
assert pca_components.shape == (1280, 1024)


# =====================================
# 4. Transform test features
# =====================================

Z_test_pca = (
    R_test - pca_mean
) @ pca_components

print("Z_test_pca:", Z_test_pca.shape)

assert Z_test_pca.shape == (200, 1024)


# =====================================
# 5. Save
# =====================================

torch.save(
    {
        "img_features": Z_test_pca,
        "text_features": test_preprojection["text_features"],
        "feature_type": "pca1024",
        "source": "ViT-H-14_preprojection_features_test.pt",
    },
    OUTPUT_PATH,
)

print("Saved:", OUTPUT_PATH)