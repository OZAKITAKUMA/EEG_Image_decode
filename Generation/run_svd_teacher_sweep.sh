#!/bin/bash
###############################################################################
# SVD-teacher sweep for the encoder-only within-subject benchmark.
#
# 1. Create low-rank reconstructed CLIP training caches if they are missing.
# 2. Train/evaluate the encoder with each reconstructed teacher.
#
# Defaults:
#   subject: sub-01
#   ranks:   1008 992 960 896 768
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBJECTS="${SUBJECTS:-sub-01}"
SVD_RANKS="${SVD_RANKS:- 1}"

SOURCE_FEATURES="${SOURCE_FEATURES:-${REPO_ROOT}/features/ViT-H-14_features_train.pt}"
SVD_FEATURES_DIR="${SVD_FEATURES_DIR:-${REPO_ROOT}/features}"

ENCODER_EPOCHS="${ENCODER_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR_ENCODER="${LR_ENCODER:-3e-4}"
RSA_WEIGHT="${RSA_WEIGHT:-0.0}"
RSA_LOSS_TYPE="${RSA_LOSS_TYPE:-pearson}"
ENCODER_SELECTION_METRIC="${ENCODER_SELECTION_METRIC:-rsa}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PATIENCE="${PATIENCE:-50}"
AVG_SIGNAL_TRAINING="${AVG_SIGNAL_TRAINING:-true}"
GPU="${GPU:-cuda:0}"
SEED="${SEED:-42}"

if [ ! -f "${SOURCE_FEATURES}" ]; then
    echo "[ERROR] Source CLIP feature cache not found:"
    echo "  ${SOURCE_FEATURES}"
    exit 1
fi

echo "============================================================"
echo " SVD Teacher sweep"
echo "============================================================"
echo " Subjects:       ${SUBJECTS}"
echo " Ranks:          ${SVD_RANKS}"
echo " Source feature: ${SOURCE_FEATURES}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Create missing SVD-reconstructed caches.
# -----------------------------------------------------------------------------
MISSING_RANKS=()

for RANK in ${SVD_RANKS}; do
    FEATURE_FILE="${SVD_FEATURES_DIR}/ViT-H-14_features_train_svd${RANK}.pt"

    if [ ! -f "${FEATURE_FILE}" ]; then
        MISSING_RANKS+=("${RANK}")
    fi
done

if [ ${#MISSING_RANKS[@]} -gt 0 ]; then
    echo ""
    echo "[STEP 0] Creating missing SVD feature caches:"
    echo "  ${MISSING_RANKS[*]}"

    python "${SCRIPT_DIR}/create_svd_clip_features.py" \
        --train_features_path "${SOURCE_FEATURES}" \
        --output_dir "${SVD_FEATURES_DIR}" \
        --ranks "${MISSING_RANKS[@]}" \
        --seed "${SEED}" \
        --val_ratio "${VAL_RATIO}"
fi

# -----------------------------------------------------------------------------
# Run one benchmark per SVD rank.
# -----------------------------------------------------------------------------
for RANK in ${SVD_RANKS}; do
    FEATURE_FILE="${SVD_FEATURES_DIR}/ViT-H-14_features_train_svd${RANK}.pt"

    EXP_NAME="svd${RANK}"

    echo ""
    echo "============================================================"
    echo " SVD Teacher experiment: ${EXP_NAME}"
    echo "============================================================"

    SUBJECTS="${SUBJECTS}" \
    ENCODER_EPOCHS="${ENCODER_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    LR_ENCODER="${LR_ENCODER}" \
    RSA_WEIGHT="${RSA_WEIGHT}" \
    RSA_LOSS_TYPE="${RSA_LOSS_TYPE}" \
    ENCODER_SELECTION_METRIC="${ENCODER_SELECTION_METRIC}" \
    VAL_RATIO="${VAL_RATIO}" \
    PATIENCE="${PATIENCE}" \
    AVG_SIGNAL_TRAINING="${AVG_SIGNAL_TRAINING}" \
    FEATURE_SPACE="clip" \
    TRAIN_FEATURES_PATH="${FEATURE_FILE}" \
    GPU="${GPU}" \
    SEED="${SEED}" \
    MODEL_SAVE_DIR="${SCRIPT_DIR}/models/benchmark_svd_teacher/clip/${ENCODER_SELECTION_METRIC}/${EXP_NAME}" \
    OUTPUT_DIR="${SCRIPT_DIR}/outputs/benchmark_svd_teacher/clip/${ENCODER_SELECTION_METRIC}/${EXP_NAME}" \
    bash "${SCRIPT_DIR}/benchmark_encoder_only_within_subject.sh"
done

echo ""
echo "============================================================"
echo " SVD Teacher sweep finished"
echo "============================================================"


ENCODER_SELECTION_METRIC=val_loss \
RSA_WEIGHT=0.0 \
SVD_RANKS="1008 992 960 896 768 640 512 384 256 128 64 16 1" \ 
bash run_svd_teacher_sweep.sh