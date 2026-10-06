import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from torch.utils.data import DataLoader, Subset

from eegdatasets import EEGDataset

from analyze_eeg_sample_difficulty import (
    load_model_checkpoint,
    make_subject_ids,
)

from encoder_utils import (
    stratified_condition_split,
)


N_CLASSES = 1654
CONDITIONS_PER_CLASS = 10
FEATURE_DIM = 1024

def load_clip_features(
    features_path,
):
    saved = torch.load(
        features_path,
        map_location="cpu",
        weights_only=False,
    )

    if "img_features" not in saved:
        raise KeyError(
            f"'img_features' was not found in: "
            f"{features_path}"
        )

    features = (
        saved[
            "img_features"
        ]
        .float()
    )

    expected_shape = (
        N_CLASSES
        * CONDITIONS_PER_CLASS,
        FEATURE_DIM,
    )

    if features.shape != expected_shape:
        raise RuntimeError(
            "Unexpected CLIP feature shape: "
            f"{tuple(features.shape)} "
            f"expected={expected_shape}"
        )

    return features

def get_train_features_by_class(
    all_features,
    seed=42,
    val_ratio=0.1,
):
    """
    train.pyと同じ9:1 splitを再現し、
    各classのtrain 9画像を取り出す。

    Returns:
        train_features:
            (1654, 9, 1024)

        val_features:
            (1654, 1, 1024)
    """

    train_indices, val_indices = (
        stratified_condition_split(
            n_classes=N_CLASSES,
            conditions_per_class=CONDITIONS_PER_CLASS,
            trials_per_condition=1,
            val_ratio=val_ratio,
            seed=seed,
        )
    )

    train_features = all_features[
        train_indices
    ].reshape(
        N_CLASSES,
        9,
        FEATURE_DIM,
    )

    val_features = all_features[
        val_indices
    ].reshape(
        N_CLASSES,
        1,
        FEATURE_DIM,
    )

    return (
        train_features,
        val_features,
    )

@torch.no_grad()
def extract_train_prediction_features(
    args,
    device,
):
    """
    train.pyと同じ9:1 splitを再現して、
    train 9刺激に対するEEG predictionを取得する。

    Returns:
        prediction_features:
            (1654, 9, 1024)
    """

    dataset = EEGDataset(
        args.data_path,
        img_dir_training=(
            args.img_dir_training
        ),
        img_dir_test=(
            args.img_dir_test
        ),
        features_dir=(
            args.features_dir
        ),
        subjects=[
            args.subject
        ],
        train=True,
        avg_trials=True,
        feature_space="clip",
    )

    train_indices, _ = (
        stratified_condition_split(
            n_classes=N_CLASSES,
            conditions_per_class=CONDITIONS_PER_CLASS,
            trials_per_condition=1,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    )

    train_subset = Subset(
        dataset,
        train_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model_checkpoint(
        args.checkpoint,
        device,
    )

    prediction_list = []

    for batch in train_loader:
        (
            eeg_data,
            labels,
            text,
            text_features,
            img,
            img_features,
        ) = batch

        eeg_data = eeg_data.to(
            device
        )

        batch_size = (
            eeg_data.size(0)
        )

        subject_ids = make_subject_ids(
            model=model,
            subject=args.subject,
            batch_size=batch_size,
            device=device,
            use_subject_id=True,
        )

        prediction = model(
            eeg_data,
            subject_ids,
        ).float()

        prediction_list.append(
            prediction
            .detach()
            .cpu()
        )

    prediction_features = torch.cat(
        prediction_list,
        dim=0,
    )

    expected_samples = (
        N_CLASSES
        * 9
    )

    if prediction_features.shape != (
        expected_samples,
        FEATURE_DIM,
    ):
        raise RuntimeError(
            "Unexpected train prediction shape: "
            f"{tuple(prediction_features.shape)}"
        )

    prediction_features = (
        prediction_features.reshape(
            N_CLASSES,
            9,
            FEATURE_DIM,
        )
    )

    return prediction_features

def compute_train9_consistency(
    train_features,
):
    """
    各classについて、
    L2正規化後のtrain 9画像のクラス内分散を計算する。

    train_features:
        (1654, 9, 1024)

    Returns:
        class_variances:
            (1654,)
    """

    # 各特徴ベクトルをL2正規化
    # (1654, 9, 1024)
    normalized_features = F.normalize(
        train_features.float(),
        dim=-1,
    )

    # 各クラスの平均特徴量 μ
    # (1654, 9, 1024)
    #          ↓
    # (1654, 1, 1024)
    class_means = normalized_features.mean(
        dim=1,
        keepdim=True,
    )

    # x_i - μ
    # (1654, 9, 1024)
    centered_features = (
        normalized_features
        - class_means
    )

    # ||x_i - μ||^2
    # (1654, 9, 1024)
    #          ↓
    # (1654, 9)
    squared_distances = (
        centered_features
        .pow(2)
        .sum(dim=-1)
    )

    # V_c = 1/N Σ ||x_i - μ||^2
    # (1654, 9)
    #      ↓
    # (1654,)
    class_variances = (
        squared_distances
        .mean(dim=1)
    )

    return (
        class_variances
        .cpu()
        .numpy()
    )

def compute_train_gt_prediction_distance(
    teacher_features,
    prediction_features,
):
    """
    train 9刺激それぞれについて、
    Teacher CLIPとEEG predictionの
    cosine distanceを計算する。

    teacher_features:
        (1654, 9, 1024)

    prediction_features:
        (1654, 9, 1024)

    Returns:
        cosine_distances:
            (1654, 9)

        class_mean_distances:
            (1654,)
    """

    teacher_n = F.normalize(
        teacher_features.float(),
        dim=-1,
    )

    prediction_n = F.normalize(
        prediction_features.float(),
        dim=-1,
    )

    cosine_similarities = (
        teacher_n
        * prediction_n
    ).sum(
        dim=-1
    )

    cosine_distances = (
        1.0
        - cosine_similarities
    )

    class_mean_distances = (
        cosine_distances.mean(
            dim=1
        )
    )

    return (
        cosine_distances.cpu().numpy(),
        class_mean_distances.cpu().numpy(),
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze CLIP-space consistency "
            "of the 9 training images per class."
        )
    )

    parser.add_argument(
        "--features_path",
        type=str,
        default=(
            "./features/"
            "ViT-H-14_features_train.pt"
        ),
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=(
            "/home/moepy/"
            "ozakitakuma/"
            "data_eeg"
        ),
    )

    parser.add_argument(
        "--img_dir_training",
        type=str,
        default=(
            "/home/moepy/"
            "ozakitakuma/"
            "data_image/"
            "training_images"
        ),
    )

    parser.add_argument(
        "--img_dir_test",
        type=str,
        default=(
            "/home/moepy/"
            "ozakitakuma/"
            "data_image/"
            "test_images"
        ),
    )

    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to trained encoder best.pth"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "./outputs/"
            "train9_clip_consistency"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    all_features = load_clip_features(
        args.features_path
    )

    (
        train_features,
        val_features,
    ) = get_train_features_by_class(
        all_features,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    print(
        "Train features:",
        train_features.shape,
    )

    print(
        "Validation features:",
        val_features.shape,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    prediction_features = (
        extract_train_prediction_features(
            args,
            device,
        )
    )

    print(
        "Train EEG predictions:",
        prediction_features.shape,
    )

    prediction_class_variances = (
        compute_train9_consistency(
            prediction_features
        )
    )


    (
        train_gt_pred_distances,
        train_gt_pred_class_mean,
    ) = compute_train_gt_prediction_distance(
        train_features,
        prediction_features,
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "TRAIN GT-PREDICTION DISTANCE"
    )

    print(
        "========================================"
    )

    print(
        "All train stimuli"
    )

    print(
        "  mean:",
        f"{train_gt_pred_distances.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(train_gt_pred_distances):.4f}",
    )

    print(
        "  min:",
        f"{train_gt_pred_distances.min():.4f}",
    )

    print(
        "  max:",
        f"{train_gt_pred_distances.max():.4f}",
    )

    print()

    print(
        "Class mean distance"
    )

    print(
        "  mean:",
        f"{train_gt_pred_class_mean.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(train_gt_pred_class_mean):.4f}",
    )

    print(
        "  min:",
        f"{train_gt_pred_class_mean.min():.4f}",
    )

    print(
        "  max:",
        f"{train_gt_pred_class_mean.max():.4f}",
    )
    
    teacher_class_variances = (
        compute_train9_consistency(
            train_features
        )
    )

    
    # ========================================================
    # Class-wise results
    # ========================================================

    variance_ratio = (
        prediction_class_variances
        / teacher_class_variances
    )

    results_df = pd.DataFrame({
        "class_index":
            np.arange(
                N_CLASSES
            ),

        "teacher_class_variance":
            teacher_class_variances,

        "prediction_class_variance":
            prediction_class_variances,

        "train_gt_prediction_mean_distance":
            train_gt_pred_class_mean,

        "variance_ratio":
            variance_ratio,
    })

    output_csv = os.path.join(
        args.output_dir,
        "train9_clip_consistency.csv",
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    # ========================================================
    # Stimulus-wise GT vs Prediction
    # ========================================================

    stimulus_rows = []

    for class_idx in range(
        N_CLASSES
    ):
        for train_idx in range(
            9
        ):
            stimulus_rows.append({
                "class_index":
                    class_idx,

                "train_index":
                    train_idx,

                "gt_prediction_distance":
                    train_gt_pred_distances[
                        class_idx,
                        train_idx,
                    ],
            })

    stimulus_df = pd.DataFrame(
        stimulus_rows
    )

    stimulus_csv = os.path.join(
        args.output_dir,
        "train9_gt_prediction_distances.csv",
    )

    stimulus_df.to_csv(
        stimulus_csv,
        index=False,
    )

    # ========================================================
    # Teacher CLIP class variance
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "TEACHER TRAIN-9 CLASS VARIANCE"
    )

    print(
        "========================================"
    )

    print(
        "Class variance"
    )

    print(
        "  mean:",
        f"{teacher_class_variances.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(teacher_class_variances):.4f}",
    )

    print(
        "  min:",
        f"{teacher_class_variances.min():.4f}",
    )

    print(
        "  max:",
        f"{teacher_class_variances.max():.4f}",
    )

        # ========================================================
    # Prediction class variance
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "EEG PREDICTION TRAIN-9 CLASS VARIANCE"
    )

    print(
        "========================================"
    )

    print(
        "Class variance"
    )

    print(
        "  mean:",
        f"{prediction_class_variances.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(prediction_class_variances):.4f}",
    )

    print(
        "  min:",
        f"{prediction_class_variances.min():.4f}",
    )

    print(
        "  max:",
        f"{prediction_class_variances.max():.4f}",
    )

    # ========================================================
    # Teacher vs Prediction class variance
    # ========================================================

    n_prediction_lower_variance = int(
        (
            prediction_class_variances
            < teacher_class_variances
        ).sum()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "TEACHER VS PREDICTION CLASS VARIANCE"
    )

    print(
        "========================================"
    )

    print(
        "Mean variance ratio:",
        f"{variance_ratio.mean():.4f}",
    )

    print(
        "Median variance ratio:",
        f"{np.median(variance_ratio):.4f}",
    )

    print(
        "Prediction variance lower than teacher:",
        f"{n_prediction_lower_variance}/"
        f"{N_CLASSES} "
        f"({100.0 * n_prediction_lower_variance / N_CLASSES:.1f}%)",
    )

    # ========================================================
    # GT vs Prediction direct distance
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "TRAIN GT-PREDICTION DISTANCE"
    )

    print(
        "========================================"
    )

    print(
        "All train stimuli"
    )

    print(
        "  mean:",
        f"{train_gt_pred_distances.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(train_gt_pred_distances):.4f}",
    )

    print(
        "  min:",
        f"{train_gt_pred_distances.min():.4f}",
    )

    print(
        "  max:",
        f"{train_gt_pred_distances.max():.4f}",
    )

    print()

    print(
        "Class mean GT-Prediction distance"
    )

    print(
        "  mean:",
        f"{train_gt_pred_class_mean.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(train_gt_pred_class_mean):.4f}",
    )

    print(
        "  min:",
        f"{train_gt_pred_class_mean.min():.4f}",
    )

    print(
        "  max:",
        f"{train_gt_pred_class_mean.max():.4f}",
    )

        # ========================================================
    # Relation:
    # teacher class variance vs fitting difficulty
    # ========================================================

    teacher_variance_series = pd.Series(
        teacher_class_variances
    )

    gt_pred_series = pd.Series(
        train_gt_pred_class_mean
    )

    variance_fit_corr = (
        teacher_variance_series.corr(
            gt_pred_series,
            method="spearman",
        )
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "TEACHER CLASS VARIANCE VS GT-PREDICTION DISTANCE"
    )

    print(
        "========================================"
    )

    print(
        "Spearman rho:",
        f"{variance_fit_corr:.4f}",
    )


if __name__ == "__main__":
    main()


# python Generation/analyze_train9_clip_consistency.py --checkpoint /home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/models/baseline/clip/full/mse_contrastive/val_rdm_mse/encoder/sub-01/10-06_05-03/best.pth