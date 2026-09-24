#!/usr/bin/env python3
"""
Analyze the dimensional spread of ground-truth CLIP image features with SVD.

Reports SVD statistics for:
  1) all training-image CLIP features:      (16540, 1024)
  2) validation teacher CLIP features only: (1654, 1024)

The validation subset is reproduced with the same stratified split used in
Generation/train.py when avg_trials=True, val_ratio=0.1, seed=42.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from encoder_utils import stratified_condition_split


N_CLASSES = 1654
CONDITIONS_PER_CLASS = 10


def load_clip_features(features_path):
    saved = torch.load(
        features_path,
        map_location="cpu",
        weights_only=False,
    )

    if "img_features" not in saved:
        raise KeyError(
            f"'img_features' was not found in: {features_path}"
        )

    features = saved["img_features"].float()

    if features.ndim != 2:
        raise ValueError(
            "Expected img_features to have shape (N, D), "
            f"but got {tuple(features.shape)}"
        )

    expected_samples = N_CLASSES * CONDITIONS_PER_CLASS

    if features.shape[0] != expected_samples:
        raise ValueError(
            "Unexpected number of CLIP training-image features: "
            f"expected={expected_samples}, actual={features.shape[0]}"
        )

    return features


def get_validation_teacher_features(
    all_features,
    seed=42,
    val_ratio=0.1,
):
    _, val_indices = stratified_condition_split(
        n_classes=N_CLASSES,
        conditions_per_class=CONDITIONS_PER_CLASS,
        trials_per_condition=1,
        val_ratio=val_ratio,
        seed=seed,
    )

    val_features = all_features[val_indices].clone()

    if val_features.shape[0] != N_CLASSES:
        raise RuntimeError(
            "Validation split should contain exactly one sample per class: "
            f"expected={N_CLASSES}, actual={val_features.shape[0]}"
        )

    return val_features, val_indices


def preprocess_features(
    features,
    l2_normalize=True,
    center=True,
):
    x = features.float().clone()

    if l2_normalize:
        x = F.normalize(
            x,
            dim=-1,
        )

    if center:
        x = x - x.mean(
            dim=0,
            keepdim=True,
        )

    return x


def compute_svd_statistics(features):
    _, singular_values, _ = torch.linalg.svd(
        features,
        full_matrices=False,
    )

    singular_values = (
        singular_values
        .cpu()
        .numpy()
    )

    variance = singular_values ** 2

    variance_ratio = (
        variance
        / variance.sum()
    )

    cumulative_ratio = np.cumsum(
        variance_ratio
    )

    return (
        singular_values,
        variance_ratio,
        cumulative_ratio,
    )


def dimension_for_threshold(
    cumulative_ratio,
    threshold,
):
    return int(
        np.searchsorted(
            cumulative_ratio,
            threshold,
        )
        + 1
    )


def write_svd_csv(
    path,
    singular_values,
    variance_ratio,
    cumulative_ratio,
):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "component",
                "singular_value",
                "explained_variance_ratio",
                "cumulative_explained_variance_ratio",
            ]
        )

        for i, (s, r, c) in enumerate(
            zip(
                singular_values,
                variance_ratio,
                cumulative_ratio,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    i,
                    s,
                    r,
                    c,
                ]
            )


def plot_singular_values(
    path,
    singular_values,
    title,
):
    components = np.arange(
        1,
        len(singular_values) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        components,
        singular_values,
    )

    ax.set_xlabel(
        "SVD component"
    )

    ax.set_ylabel(
        "Singular value"
    )

    ax.set_title(
        title
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_cumulative_variance(
    path,
    cumulative_ratio,
    title,
):
    components = np.arange(
        1,
        len(cumulative_ratio) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        components,
        cumulative_ratio * 100.0,
    )

    for threshold in (
        90,
        95,
        99,
    ):
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
        )

    ax.set_xlabel(
        "Number of SVD components"
    )

    ax.set_ylabel(
        "Cumulative explained variance (%)"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        100.5,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_summary(
    path,
    name,
    shape,
    variance_ratio,
    cumulative_ratio,
    l2_normalize,
    center,
):
    d90 = dimension_for_threshold(
        cumulative_ratio,
        0.90,
    )

    d95 = dimension_for_threshold(
        cumulative_ratio,
        0.95,
    )

    d99 = dimension_for_threshold(
        cumulative_ratio,
        0.99,
    )

    top10 = (
        variance_ratio[:10].sum()
        * 100.0
    )

    top50 = (
        variance_ratio[:50].sum()
        * 100.0
    )

    top100 = (
        variance_ratio[:100].sum()
        * 100.0
    )

    lines = [
        f"Dataset: {name}",
        f"Shape: {shape}",
        f"L2 normalized: {l2_normalize}",
        f"Mean centered: {center}",
        "",
        f"Dimensions for 90% variance: {d90}",
        f"Dimensions for 95% variance: {d95}",
        f"Dimensions for 99% variance: {d99}",
        "",
        f"Variance captured by first 10 dims:  {top10:.2f}%",
        f"Variance captured by first 50 dims:  {top50:.2f}%",
        f"Variance captured by first 100 dims: {top100:.2f}%",
    ]

    text = "\n".join(lines)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            text + "\n"
        )

    print("")
    print(
        "=" * 65
    )

    print(
        text
    )

    print(
        "=" * 65
    )


def analyze_one_feature_set(
    name,
    features,
    output_dir,
    l2_normalize,
    center,
):
    x = preprocess_features(
        features,
        l2_normalize=l2_normalize,
        center=center,
    )

    (
        singular_values,
        variance_ratio,
        cumulative_ratio,
    ) = compute_svd_statistics(
        x
    )

    prefix = (
        name
        .replace(" ", "_")
        .lower()
    )

    csv_path = os.path.join(
        output_dir,
        f"{prefix}_svd.csv",
    )

    spectrum_path = os.path.join(
        output_dir,
        f"{prefix}_singular_value_spectrum.png",
    )

    cumulative_path = os.path.join(
        output_dir,
        f"{prefix}_cumulative_variance.png",
    )

    summary_path = os.path.join(
        output_dir,
        f"{prefix}_summary.txt",
    )

    write_svd_csv(
        csv_path,
        singular_values,
        variance_ratio,
        cumulative_ratio,
    )

    plot_singular_values(
        spectrum_path,
        singular_values,
        title=(
            f"{name}: "
            "singular-value spectrum"
        ),
    )

    plot_cumulative_variance(
        cumulative_path,
        cumulative_ratio,
        title=(
            f"{name}: "
            "cumulative explained variance"
        ),
    )

    save_summary(
        summary_path,
        name=name,
        shape=tuple(
            features.shape
        ),
        variance_ratio=variance_ratio,
        cumulative_ratio=cumulative_ratio,
        l2_normalize=l2_normalize,
        center=center,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SVD analysis of ground-truth "
            "CLIP image features."
        )
    )

    parser.add_argument(
        "--features_path",
        default=(
            "../features/"
            "ViT-H-14_features_train.pt"
        ),
        help=(
            "Path to ViT-H-14 training "
            "feature cache. "
            "Default assumes this script "
            "is run from Generation/."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="./clip_teacher_svd",
        help=(
            "Directory for SVD "
            "analysis outputs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Validation split seed."
        ),
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help=(
            "Validation ratio."
        ),
    )

    parser.add_argument(
        "--no_l2_normalize",
        action="store_true",
        help=(
            "Disable per-feature L2 "
            "normalization before SVD. "
            "Default is L2-normalized "
            "because the project uses "
            "cosine geometry."
        ),
    )

    parser.add_argument(
        "--no_center",
        action="store_true",
        help=(
            "Disable mean-centering "
            "before SVD. "
            "Default is centered SVD."
        ),
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
        val_features,
        val_indices,
    ) = get_validation_teacher_features(
        all_features,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    print(
        "All teacher CLIP features:",
        tuple(
            all_features.shape
        ),
    )

    print(
        "Validation teacher CLIP features:",
        tuple(
            val_features.shape
        ),
    )

    val_index_path = os.path.join(
        args.output_dir,
        "validation_feature_indices.txt",
    )

    with open(
        val_index_path,
        "w",
        encoding="utf-8",
    ) as f:
        for index in val_indices:
            f.write(
                f"{index}\n"
            )

    l2_normalize = (
        not args.no_l2_normalize
    )

    center = (
        not args.no_center
    )

    analyze_one_feature_set(
        name="all_teacher_clip",
        features=all_features,
        output_dir=args.output_dir,
        l2_normalize=l2_normalize,
        center=center,
    )

    analyze_one_feature_set(
        name="validation_teacher_clip",
        features=val_features,
        output_dir=args.output_dir,
        l2_normalize=l2_normalize,
        center=center,
    )

    print("")
    print(
        "Saved validation indices:"
    )

    print(
        f"  {val_index_path}"
    )

    print("")
    print(
        "SVD analysis finished."
    )


if __name__ == "__main__":
    main()