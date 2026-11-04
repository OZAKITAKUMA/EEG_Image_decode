#!/usr/bin/env bash
# Deep CCA vs. identical sub-01 encoder-only baseline configuration.
# Run from Generation/: bash run_deepcca_sub01.sh
#
# Defaults to TRAIN + retrieval-only EVALUATE + matched PCA arrow comparison.
# Optional: RUN_GENERATION=1 bash run_deepcca_sub01.sh (200 SDXL images).
set -euo pipefail

cd "$(dirname "$0")"

SUBJECT="sub-01"
DATA_PATH="${DATA_PATH:-/home/moepy/ozakitakuma/data_eeg}"
IMG_DIR_TRAINING="${IMG_DIR_TRAINING:-/home/moepy/ozakitakuma/data_image/training_images}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/home/moepy/ozakitakuma/data_image/test_images}"
FEATURES_DIR="${FEATURES_DIR:-../features}"
BASELINE_PATH="${BASELINE_PATH:-models/baseline/clip/full/mse_contrastive/val_rdm_mse/encoder/sub-01/10-10_07-51/best.pth}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/deepcca/clip}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-models/deepcca/clip}"
DCCA_WEIGHT="${DCCA_WEIGHT:-0.1}"
DCCA_DIM="${DCCA_DIM:-16}"
GPU="${GPU:-cuda:0}"
RUN_GENERATION="${RUN_GENERATION:-0}"
SDXL_MODEL_PATH="${SDXL_MODEL_PATH:-stabilityai/sdxl-turbo}"
IP_ADAPTER_PATH="${IP_ADAPTER_PATH:-h94/IP-Adapter}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

if [[ ! -f "$BASELINE_PATH" ]]; then
  echo "[ERROR] Baseline checkpoint not found: $BASELINE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MODEL_SAVE_DIR"
marker="$(mktemp "$OUTPUT_DIR/.deepcca_start.XXXXXXXX")"
trap 'rm -f "$marker"' EXIT

echo "Training sub-01 from scratch (encoder-only, avg_trials, clip 1024, seed=42)"
python -u train.py \
  --encoder_only \
  --data_path "$DATA_PATH" \
  --img_dir_training "$IMG_DIR_TRAINING" \
  --img_dir_test "$IMG_DIR_TEST" \
  --features_dir "$FEATURES_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_save_dir "$MODEL_SAVE_DIR" \
  --subject "$SUBJECT" \
  --image_encoder clip \
  --feature_space clip \
  --avg_trials \
  --method deepcca \
  --experiment_type deepcca \
  --total_epochs 100 \
  --batch_size 64 \
  --lr_encoder 3e-4 \
  --val_ratio 0.1 \
  --patience 50 \
  --save_interval 10 \
  --rsa_weight 0.0 \
  --checkpoint_criterion val_rdm_mse \
  --deep_cca_weight "$DCCA_WEIGHT" \
  --deep_cca_dim "$DCCA_DIM" \
  --gpu "$GPU" \
  --seed 42

# Use ONLY the newly created run, not any prior timestamped run.
RUN_INFO="$(find "$OUTPUT_DIR/$SUBJECT" -name paths_info.txt -type f -newer "$marker" | sort | tail -n 1)"
if [[ -z "$RUN_INFO" ]]; then
  echo "[ERROR] No new paths_info.txt was produced" >&2
  exit 1
fi
ENCODER_PATH="$(sed -n 's/^encoder_path=//p' "$RUN_INFO" | head -n 1)"
RUN_DIR="$(dirname "$RUN_INFO")"
if [[ ! -f "$ENCODER_PATH" ]]; then
  echo "[ERROR] Trained best encoder not found: $ENCODER_PATH" >&2
  exit 1
fi
echo "Best Deep CCA encoder: $ENCODER_PATH"
echo "Logs: $RUN_DIR"

echo "Retrieval-only evaluate.py (no image generation)"
python -u evaluate.py \
  --retrieval_only \
  --data_path "$DATA_PATH" \
  --img_directory_test "$IMG_DIR_TEST" \
  --img_dir_training "$IMG_DIR_TRAINING" \
  --features_dir "$FEATURES_DIR" \
  --output_dir "$RUN_DIR" \
  --encoder_path "$ENCODER_PATH" \
  --subject "$SUBJECT" \
  --feature_space clip \
  --image_encoder clip \
  --gpu "$GPU" \
  --seed 42

echo "Draw matched baseline vs. Deep CCA validation error arrows"
python -u compare_deepcca_arrows.py \
  --baseline-path "$BASELINE_PATH" \
  --deepcca-path "$ENCODER_PATH" \
  --subject "$SUBJECT" \
  --data-path "$DATA_PATH" \
  --img-dir-training "$IMG_DIR_TRAINING" \
  --features-dir "$FEATURES_DIR" \
  --output-dir "$RUN_DIR/deepcca_arrow_comparison" \
  --pc-x 600 \
  --pc-y 601 \
  --n-arrows 45 \
  --device "$GPU" \
  --seed 42

if [[ "$RUN_GENERATION" == "1" ]]; then
  echo "Optional SDXL generation: 200 images (1 per test class)"
  python -u evaluate.py \
    --encoder_only \
    --data_path "$DATA_PATH" \
    --img_directory_test "$IMG_DIR_TEST" \
    --img_dir_training "$IMG_DIR_TRAINING" \
    --features_dir "$FEATURES_DIR" \
    --output_dir "$RUN_DIR" \
    --encoder_path "$ENCODER_PATH" \
    --subject "$SUBJECT" \
    --feature_space clip \
    --image_encoder clip \
    --num_gen_per_class 1 \
    --gen_batch_size 1 \
    --sdxl_steps 4 \
    --sdxl_model_path "$SDXL_MODEL_PATH" \
    --ip_adapter_path "$IP_ADAPTER_PATH" \
    --gpu "$GPU" \
    --seed 42
fi

echo "COMPLETE: $RUN_DIR"
