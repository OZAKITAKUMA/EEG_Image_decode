#!/bin/bash
###############################################################################
# Encoder-only benchmark for EEG-to-Image reconstruction.
#
# Per subject:
#   1. Train only the ATMS EEG encoder (default: 100 epochs maximum).
#   2. Evaluate retrieval from the encoder output.
#   3. Generate images directly from the encoder output.
#   4. Compute and aggregate encoder-only reconstruction metrics.
#
# The Diffusion Prior is never created, trained, loaded, or evaluated.
# The learnable EEG CLS -> CLIP Adapter is not used.
#
# Examples:
#   bash benchmark_encoder_only.sh
#   SUBJECTS="sub-01 sub-02" bash benchmark_encoder_only.sh
#   RESUME=08-16_02-30 SUBJECTS=sub-01 bash benchmark_encoder_only.sh
###############################################################################

set -e

#==============================================================================
# Paths
#==============================================================================
DATA_PATH="${DATA_PATH:-/home/moepy/ozakitakuma/data_eeg}"
IMG_DIR_TRAINING="${IMG_DIR_TRAINING:-/home/moepy/ozakitakuma/data_image/training_images}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/home/moepy/ozakitakuma/data_image/test_images}"

SDXL_MODEL_PATH="${SDXL_MODEL_PATH:-stabilityai/sdxl-turbo}"
IP_ADAPTER_PATH="${IP_ADAPTER_PATH:-h94/IP-Adapter}"
FEATURES_DIR="${FEATURES_DIR:-}"
VISION_MODELS_DIR="${VISION_MODELS_DIR:-./vision_models}"

# Keep encoder-only artifacts separate from the original benchmark.
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-./models/benchmark_encoder_only_loso_idfree/cls}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/benchmark_encoder_only_loso_idfree/cls}"

#==============================================================================
# Experiment settings
#==============================================================================
ALL_SUBJECTS=(
    sub-01
    sub-02
    sub-03
    sub-04
    sub-05
    sub-06
    sub-07
    sub-08
    sub-09
    sub-10
)
# 除外被験者(test用)
SUBJECTS="${ALL_SUBJECTS[*]}"
RESUME="${RESUME:-}"

ENCODER_EPOCHS="${ENCODER_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR_ENCODER="${LR_ENCODER:-3e-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PATIENCE="${PATIENCE:-50}"
AVG_SIGNAL_TRAINING="${AVG_SIGNAL_TRAINING:-true}"

# CLIP projected feature space (1024-D) is the default baseline.
FEATURE_SPACE="cls"                                                                #
if [ "${FEATURE_SPACE}" != "clip" ] && [ "${FEATURE_SPACE}" != "cls" ]; then
    echo "[ERROR] FEATURE_SPACE must be clip or cls"
    exit 1
fi

NUM_GEN_PER_CLASS="${NUM_GEN_PER_CLASS:-1}"
SDXL_INFERENCE_STEPS="${SDXL_INFERENCE_STEPS:-4}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"

GPU="${GPU:-cuda:0}"
SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

mkdir -p "${VISION_MODELS_DIR}"
export OPEN_CLIP_CACHE_DIR="${VISION_MODELS_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  ATMS Encoder-only EEG-to-Image Benchmark"
echo "============================================================"
echo "  Held-out subjects: ${SUBJECTS}"
echo "  Training mode:     LOSO (9 train / 1 test)"
echo "  Subject ID:        disabled (shared token)"
echo "  Data path:       ${DATA_PATH}"
echo "  Encoder epochs:  ${ENCODER_EPOCHS} (max)"
echo "  Feature space:   ${FEATURE_SPACE}"
echo "  Avg trials:      ${AVG_SIGNAL_TRAINING}"
echo "  Adapter:         false"
echo "  Diffusion Prior: disabled"
echo "  Seed:            ${SEED}"
echo "  GPU:             ${GPU}"
echo "  Output dir:      ${OUTPUT_DIR}"
if [ -n "${RESUME}" ]; then
    echo "  RESUME:          ${RESUME} (skip training)"
fi
echo "============================================================"

COMPLETED_SUBJECTS=()
METRIC_FILES=()

# Continue with other subjects if one subject fails.
set +e

for SUBJECT in ${SUBJECTS}; do
    TRAIN_SUBJECTS=()

    for CANDIDATE_SUBJECT in "${ALL_SUBJECTS[@]}"; do
        if [ "${CANDIDATE_SUBJECT}" != "${SUBJECT}" ]; then
            TRAIN_SUBJECTS+=("${CANDIDATE_SUBJECT}")
        fi
    done

    if [ "${#TRAIN_SUBJECTS[@]}" -ne 9 ]; then
        echo "[ERROR] Expected 9 training subjects, got ${#TRAIN_SUBJECTS[@]}."
        continue
    fi

    echo ""
    echo "############################################################"
    echo "  Held-out subject: ${SUBJECT}"
    echo "  Train subjects:   ${TRAIN_SUBJECTS[*]}"
    echo "  Subject ID:       disabled (shared token)"
    echo "############################################################"
        
    if [ -n "${RESUME}" ]; then
        TIMESTAMP="${RESUME}"
        EVAL_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${TIMESTAMP}"
        PATHS_INFO="${EVAL_OUTPUT_DIR}/paths_info.txt"

        if [ -f "${PATHS_INFO}" ]; then
            ENCODER_PATH=$(grep "encoder_path=" "${PATHS_INFO}" | cut -d= -f2)
        else
            ENCODER_PATH="${MODEL_SAVE_DIR}/encoder/${SUBJECT}/${TIMESTAMP}/best.pth"
        fi

        if [ ! -f "${ENCODER_PATH}" ]; then
            echo "[WARN] Encoder not found: ${ENCODER_PATH}. Skipping ${SUBJECT}."
            continue
        fi
    else
        TRAIN_CMD=(
            python -u train.py
            --encoder_only
            --data_path "${DATA_PATH}"
            --img_dir_training "${IMG_DIR_TRAINING}"
            --img_dir_test "${IMG_DIR_TEST}"
            --output_dir "${OUTPUT_DIR}"
            --model_save_dir "${MODEL_SAVE_DIR}"
            --subject "${SUBJECT}"
            --train_subjects "${TRAIN_SUBJECTS[@]}"
            --exclude_subject "${SUBJECT}"
            --no_subject_id
            --total_epochs "${ENCODER_EPOCHS}"
            --batch_size "${BATCH_SIZE}"
            --lr_encoder "${LR_ENCODER}"
            --gpu "${GPU}"
            --seed "${SEED}"
            --save_interval "${SAVE_INTERVAL}"
            --val_ratio "${VAL_RATIO}"
            --patience "${PATIENCE}"
            --feature_space "${FEATURE_SPACE}"
        )

        if [ -n "${FEATURES_DIR}" ]; then
            TRAIN_CMD+=(--features_dir "${FEATURES_DIR}")
        fi

        if [ "${AVG_SIGNAL_TRAINING}" = true ]; then
            TRAIN_CMD+=(--avg_trials)
        fi

        echo "  [STEP 1] Encoder-only training ..."
        "${TRAIN_CMD[@]}"
        TRAIN_STATUS=$?

        if [ ${TRAIN_STATUS} -ne 0 ]; then
            echo "[WARN] Training failed for ${SUBJECT} (exit ${TRAIN_STATUS})."
            continue
        fi

        PATHS_INFO=$(find "${OUTPUT_DIR}/${SUBJECT}" -name paths_info.txt -type f | sort | tail -1)
        if [ -z "${PATHS_INFO}" ]; then
            echo "[WARN] paths_info.txt not found for ${SUBJECT}."
            continue
        fi

        ENCODER_PATH=$(grep "encoder_path=" "${PATHS_INFO}" | cut -d= -f2)
        TIMESTAMP=$(grep "timestamp=" "${PATHS_INFO}" | cut -d= -f2)
        EVAL_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${TIMESTAMP}"
    fi

    if [ ! -f "${ENCODER_PATH}" ]; then
        echo "[WARN] Encoder checkpoint missing: ${ENCODER_PATH}"
        continue
    fi

    EVAL_CMD=(
        python -u evaluate.py
        --encoder_only
        --data_path "${DATA_PATH}"
        --img_directory_test "${IMG_DIR_TEST}"
        --img_dir_training "${IMG_DIR_TRAINING}"
        --output_dir "${EVAL_OUTPUT_DIR}"
        --encoder_path "${ENCODER_PATH}"
        --subject "${SUBJECT}"
        --no_subject_id
        --num_gen_per_class "${NUM_GEN_PER_CLASS}"
        --sdxl_steps "${SDXL_INFERENCE_STEPS}"
        --gen_batch_size "${GEN_BATCH_SIZE}"
        --feature_space "${FEATURE_SPACE}"
        --gpu "${GPU}"
        --seed "${SEED}"
    )

    if [ -n "${FEATURES_DIR}" ]; then
        EVAL_CMD+=(--features_dir "${FEATURES_DIR}")
    fi
    if [ -n "${SDXL_MODEL_PATH}" ]; then
        EVAL_CMD+=(--sdxl_model_path "${SDXL_MODEL_PATH}")
    fi
    if [ -n "${IP_ADAPTER_PATH}" ]; then
        EVAL_CMD+=(--ip_adapter_path "${IP_ADAPTER_PATH}")
    fi

    echo "  [STEP 2] Encoder-only generation and evaluation ..."
    "${EVAL_CMD[@]}"
    EVAL_STATUS=$?

    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "[WARN] Evaluation failed for ${SUBJECT} (exit ${EVAL_STATUS})."
        continue
    fi

    METRIC_FILE="${EVAL_OUTPUT_DIR}/reconstruction_metrics_${SUBJECT}_encoder_only.csv"
    if [ ! -f "${METRIC_FILE}" ]; then
        echo "[WARN] Encoder-only metrics missing: ${METRIC_FILE}"
        continue
    fi

    COMPLETED_SUBJECTS+=("${SUBJECT}")
    METRIC_FILES+=("${METRIC_FILE}")
    echo "  [INFO] ${SUBJECT} complete: ${METRIC_FILE}"
done

set -e

N_DONE=${#COMPLETED_SUBJECTS[@]}
if [ ${N_DONE} -eq 0 ]; then
    echo "[ERROR] No subjects completed successfully."
    exit 1
fi

METRIC_LIST="${METRIC_FILES[*]}"
SUBJECT_LIST="${COMPLETED_SUBJECTS[*]}"
SUMMARY_CSV="${OUTPUT_DIR}/summary_encoder_only_$(date +%m-%d_%H-%M).csv"

python3 - "${METRIC_LIST}" "${SUBJECT_LIST}" "${SUMMARY_CSV}" <<'PYEOF'
import csv
import os
import sys
from collections import defaultdict

import numpy as np

metric_files = sys.argv[1].split()
subjects = sys.argv[2].split()
summary_path = sys.argv[3]

per_metric = defaultdict(list)
subject_metrics = {}
metric_order = []
valid_subjects = []

for path, subject in zip(metric_files, subjects):
    if not os.path.isfile(path):
        print(f"[WARN] Missing metrics for {subject}: {path}")
        continue

    values = {}
    with open(path) as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        for row in reader:
            try:
                values[row['Metric']] = float(row['Mean'])
            except (KeyError, TypeError, ValueError):
                continue

    if not values:
        continue

    valid_subjects.append(subject)
    subject_metrics[subject] = values
    for metric, value in values.items():
        if metric not in metric_order:
            metric_order.append(metric)
        per_metric[metric].append(value)

if not per_metric:
    raise SystemExit("No valid encoder-only metrics found")

print("\nEncoder-only results by subject")
headers = ["Subject"] + metric_order
table_rows = []
for subject in valid_subjects:
    table_rows.append([
        subject,
        *[
            f"{subject_metrics[subject][metric]:.4f}"
            if metric in subject_metrics[subject]
            else "-"
            for metric in metric_order
        ],
    ])

widths = [
    max(len(headers[i]), *(len(row[i]) for row in table_rows))
    for i in range(len(headers))
]
print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
print("  ".join("-" * width for width in widths))
for row in table_rows:
    print("  ".join(value.ljust(width) for value, width in zip(row, widths)))

rows = []
aggregate_rows = {}
print("\nCross-subject aggregate")
print("-" * 60)
for metric in metric_order:
    values = [
        subject_metrics[subject][metric]
        for subject in valid_subjects
        if metric in subject_metrics[subject]
    ]
    mean = float(np.mean(values))
    std = float(np.std(values))
    aggregate_rows[metric] = (mean, std)
    print(f"{metric:<16} mean={mean:.4f} std={std:.4f}")
    rows.append({
        'Metric': metric,
        **{
            subject: (
                f"{subject_metrics[subject][metric]:.4f}"
                if metric in subject_metrics[subject]
                else ''
            )
            for subject in valid_subjects
        },
        'Mean': f"{mean:.4f}",
        'Std': f"{std:.4f}",
    })

os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, 'w', newline='') as stream:
    fieldnames = ['Metric'] + valid_subjects + ['Mean', 'Std']
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

per_subject_path = os.path.join(
    os.path.dirname(summary_path),
    os.path.basename(summary_path).replace(
        'summary_encoder_only_', 'per_subject_encoder_only_', 1
    ),
)
with open(per_subject_path, 'w', newline='') as stream:
    fieldnames = ['Subject'] + metric_order
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for subject in valid_subjects:
        writer.writerow({
            'Subject': subject,
            **{
                metric: (
                    f"{subject_metrics[subject][metric]:.4f}"
                    if metric in subject_metrics[subject]
                    else ''
                )
                for metric in metric_order
            },
        })
    for label, index in [('Mean', 0), ('Std', 1)]:
        writer.writerow({
            'Subject': label,
            **{
                metric: f"{aggregate_rows[metric][index]:.4f}"
                for metric in metric_order
            },
        })

print(f"Summary saved to: {summary_path}")
print(f"Per-subject table saved to: {per_subject_path}")
PYEOF

echo ""
echo "============================================================"
echo "  Encoder-only Benchmark Complete"
echo "============================================================"
echo "  Subjects run: ${COMPLETED_SUBJECTS[*]}"
echo "  Summary:      ${SUMMARY_CSV}"
echo "  Outputs:      ${OUTPUT_DIR}/<subject>/<timestamp>/"
echo "============================================================"