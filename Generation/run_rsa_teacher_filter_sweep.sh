#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# Common settings
# ============================================================

export SUBJECTS="sub-01"

export ENCODER_EPOCHS=100
export BATCH_SIZE=64

export FEATURE_SPACE="clip"
export AVG_SIGNAL_TRAINING=true

export RSA_WEIGHT=0.0
export RSA_LOSS_TYPE="pearson"

export SEED=42

FILTER_MODES=(
    "nearest"
    "farthest"
    "random"
)

FILTER_COUNTS=(
    1
    2
    3
)

# ============================================================
# Sweep
# ============================================================

for FILTER_MODE in "${FILTER_MODES[@]}"; do

    for FILTER_COUNT in "${FILTER_COUNTS[@]}"; do

        EXP_NAME="${FILTER_MODE}_${FILTER_COUNT}"

        echo ""
        echo "============================================================"
        echo " Teacher sample filter experiment"
        echo "============================================================"
        echo " Mode:  ${FILTER_MODE}"
        echo " Count: ${FILTER_COUNT}"
        echo " Name:  ${EXP_NAME}"
        echo "============================================================"

        export TRAIN_SAMPLE_FILTER="${FILTER_MODE}"
        export TRAIN_SAMPLE_FILTER_COUNT="${FILTER_COUNT}"

        export MODEL_SAVE_DIR="${SCRIPT_DIR}/models/benchmark_teacher_filter/clip/${EXP_NAME}"
        export OUTPUT_DIR="${SCRIPT_DIR}/outputs/benchmark_teacher_filter/clip/${EXP_NAME}"
        bash "${SCRIPT_DIR}/benchmark_encoder_only_within_subject.sh"

        STATUS=$?

        if [ ${STATUS} -ne 0 ]; then
            echo ""
            echo "[WARN] Failed: ${EXP_NAME}"
            echo ""
        else
            echo ""
            echo "[DONE] ${EXP_NAME}"
            echo ""
        fi

    done

done

echo ""
echo "============================================================"
echo " All teacher-filter experiments finished"
echo "============================================================"