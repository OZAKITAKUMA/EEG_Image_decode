#!/bin/bash

set -e

# ============================================================
# RSA loss experiment settings
# ============================================================

export SUBJECTS="sub-01"

export ENCODER_EPOCHS=100
export BATCH_SIZE=64
export FEATURE_SPACE="clip"
export AVG_SIGNAL_TRAINING=true

export RSA_WEIGHT=0.0

export SEED=42

export MODEL_SAVE_DIR="./models/benchmark_encoder_only_rsa/clip/rsa_0.0"
export OUTPUT_DIR="./outputs/benchmark_encoder_only_rsa/clip/rsa_0.0"

# ============================================================
# Run
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/benchmark_encoder_only_within_subject.sh"