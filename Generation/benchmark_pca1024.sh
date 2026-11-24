#!/usr/bin/env bash
set -e

# PCA1024 Encoder-only Benchmark
# Reuses benchmark_encoder_only_within_subject.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Subjects
export SUBJECTS="${SUBJECTS:-sub-01}"

# Feature space
export IMAGE_ENCODER="clip"
export FEATURE_SPACE="clip"
export METHOD="pca1024"

# Feature caches
export TRAIN_FEATURES_PATH="../features/ViT-H-14_pca1024_features_train.pt"
export TEST_FEATURES_PATH="../features/ViT-H-14_pca1024_features_test.pt"
export PCA_TRAIN_FEATURES_PATH="../features/ViT-H-14_pca1024_features_train.pt"

# Training
export ENCODER_EPOCHS=100
export BATCH_SIZE=64
export LR_ENCODER=3e-4
export PATIENCE=20
export SAVE_INTERVAL=50
export VAL_RATIO=0.1
export CHECKPOINT_CRITERION="val_rsa_pearson"

export AVG_SIGNAL_TRAINING=true
export RSA_WEIGHT=0.0
export SVD_MSE_WEIGHTING=false
export SEED=42

# Retrieval by default; set RETRIEVAL_ONLY=false for generation
export RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-true}"

# Separate output directories
export MODEL_SAVE_DIR="./models/benchmark_encoder_only/clip/pca1024"
export OUTPUT_DIR="./outputs/benchmark_encoder_only/clip/pca1024"

# Run
exec bash "${SCRIPT_DIR}/benchmark_encoder_only_within_subject.sh"