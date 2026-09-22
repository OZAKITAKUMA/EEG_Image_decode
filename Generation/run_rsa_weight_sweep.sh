#!/bin/bash

# 1つのweightで失敗しても、残りの実験は続ける
set +e

# ============================================================
# Common settings
# ============================================================

export SUBJECTS="sub-01"
export ENCODER_EPOCHS=100
export BATCH_SIZE=64
export FEATURE_SPACE="clip"
export AVG_SIGNAL_TRAINING=true
export SEED=42

# すでに 0.1～0.5 は実験済み
RSA_WEIGHTS=(
    0.0
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_DIR="${SCRIPT_DIR}/outputs/rsa_weight_sweep_logs"
mkdir -p "${LOG_DIR}"

# ============================================================
# Sweep
# ============================================================

for RSA_WEIGHT in "${RSA_WEIGHTS[@]}"; do

    export RSA_WEIGHT
    export RSA_LOSS_TYPE="rdm_mse"
    export MODEL_SAVE_DIR="./models/benchmark_encoder_only_rsa_mse/${FEATURE_SPACE}/rsa_${RSA_WEIGHT}"
    export OUTPUT_DIR="./outputs/benchmark_encoder_only_rsa_mse/${FEATURE_SPACE}/rsa_${RSA_WEIGHT}"

    LOG_FILE="${LOG_DIR}/rsa_${RSA_WEIGHT}.log"

    echo ""
    echo "============================================================"
    echo "  RSA weight experiment"
    echo "  Subject:       ${SUBJECTS}"
    echo "  RSA weight:    ${RSA_WEIGHT}"
    echo "  RSA loss type: ${RSA_LOSS_TYPE}"
    echo "  Model dir:     ${MODEL_SAVE_DIR}"
    echo "  Output dir:    ${OUTPUT_DIR}"
    echo "  Log:           ${LOG_FILE}"
    echo "============================================================"

    bash "${SCRIPT_DIR}/benchmark_encoder_only_within_subject.sh" \
        2>&1 | tee "${LOG_FILE}"

    STATUS=${PIPESTATUS[0]}

    if [ ${STATUS} -eq 0 ]; then
        echo "[OK] RSA weight ${RSA_WEIGHT} finished."
    else
        echo "[ERROR] RSA weight ${RSA_WEIGHT} failed with status ${STATUS}."
        echo "[INFO] Continue to next weight."
    fi

done

echo ""
echo "============================================================"
echo "  RSA weight sweep finished"
echo "============================================================"