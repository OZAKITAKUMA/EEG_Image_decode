#!/bin/bash

set -e

BASE=/home/moepy/ozakitakuma/EEG_Image_decode_develop

RESULTS_FILE="${BASE}/Generation/outputs/train9_clip_consistency/summary_variance.txt"
mkdir -p "$(dirname "${RESULTS_FILE}")"
: > "${RESULTS_FILE}"

while read NAME FEATURE CHECKPOINT; do
    echo "========================================"
    echo "START: ${NAME}"
    echo "========================================"

    TMP_LOG=$(mktemp)

    python "${BASE}/Generation/analyze_train9_clip_consistency.py" \
        --features_path "${BASE}/features/${FEATURE}" \
        --checkpoint "${CHECKPOINT}" \
        --output_dir "${BASE}/Generation/outputs/train9_clip_consistency/${NAME}" \
        | tee "${TMP_LOG}"

    VARIANCE_MEAN=$(
        awk '
            /TEACHER TRAIN-9 CLASS VARIANCE/ {in_block=1; next}
            in_block && /mean:/ {print $2; exit}
        ' "${TMP_LOG}"
    )

    echo "${NAME} ${VARIANCE_MEAN}" >> "${RESULTS_FILE}"

    rm -f "${TMP_LOG}"

    echo "========================================"
    echo "END: ${NAME}"
    echo "========================================"
    echo

done <<EOF
class_bottom64 ViT-H-14_features_train_svd_class_remove_bottom64.pt ${BASE}/Generation/models/baseline/clip/svd_class_remove_bottom_64/mse_contrastive/val_rdm_mse/encoder/sub-01/10-07_03-21/best.pth
class_top64 ViT-H-14_features_train_svd_class_remove_top64.pt ${BASE}/Generation/models/baseline/clip/svd_class_remove_top_64/mse_contrastive/val_rdm_mse/encoder/sub-01/10-07_03-12/best.pth
global_bottom64 ViT-H-14_features_train_svd_global_remove_bottom64.pt ${BASE}/Generation/models/baseline/clip/svd_global_remove_bottom_64/mse_contrastive/val_rdm_mse/encoder/sub-01/10-07_02-53/best.pth
global_top64 ViT-H-14_features_train_svd_global_remove_top64.pt ${BASE}/Generation/models/baseline/clip/svd_global_remove_top_64/mse_contrastive/val_rdm_mse/encoder/sub-01/10-07_03-02/best.pth
EOF


echo ""
echo "========================================"
echo "TEACHER CLASS VARIANCE SUMMARY"
echo "========================================"
printf "%-20s %s\n" "Condition" "Mean variance"
printf "%-20s %s\n" "--------------------" "-------------"

while read NAME VARIANCE; do
    printf "%-20s %s\n" "${NAME}" "${VARIANCE}"
done < "${RESULTS_FILE}"

echo "========================================"
