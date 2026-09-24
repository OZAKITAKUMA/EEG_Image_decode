import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from analyze_eeg_sample_difficulty import (
    load_category_mapping,
    compute_eeg_prediction_quality,
    spearman_rho,
)


# ============================================================
# Category-specific center distance
# ============================================================

def compute_within_category_rows(
    teacher_features,
    eeg_features,
    labels,
    class_to_categories,
    categories,
    min_category_size,
):
    """
    各カテゴリの内部だけで、

        「そのカテゴリ中心からの距離」
            vs
        「EEG予測性能」

    を調べる。

    multi-labelの刺激は、
    所属する各カテゴリについて1行ずつ解析する。

    例:
        aardvark = animal, mammal

    なら、

        animal中心からの距離
        mammal中心からの距離

    をそれぞれ別に計算する。
    """

    teacher_n = F.normalize(
        teacher_features.float(),
        dim=-1,
    )

    gt_cosine, gt_rank = (
        compute_eeg_prediction_quality(
            eeg_features,
            teacher_features,
        )
    )

    labels_list = labels.tolist()

    rows = []

    for category in categories:

        members = []

        for row_idx, class_idx in enumerate(
            labels_list
        ):
            class_categories = (
                class_to_categories.get(
                    int(class_idx),
                    set(),
                )
            )

            if category in class_categories:
                members.append(row_idx)

        n_members = len(members)

        # 小さすぎるカテゴリの相関は
        # かなり不安定になるので除外
        if n_members < min_category_size:
            continue

        member_tensor = torch.tensor(
            members,
            dtype=torch.long,
        )

        category_vectors = (
            teacher_n[
                member_tensor
            ]
        )

        # leave-one-out中心を効率よく作るため
        # まずカテゴリ全体のベクトル和を取る
        category_sum = (
            category_vectors.sum(
                dim=0
            )
        )

        for row_idx in members:

            # 自分自身をカテゴリ中心から除外
            center = (
                category_sum
                - teacher_n[row_idx]
            )

            center = (
                center
                / (n_members - 1)
            )

            center = F.normalize(
                center.unsqueeze(0),
                dim=-1,
            ).squeeze(0)

            similarity = torch.dot(
                teacher_n[row_idx],
                center,
            ).item()

            center_distance = (
                1.0
                - similarity
            )

            rows.append({
                "category":
                    category,

                "row_index":
                    row_idx,

                "class_index":
                    int(
                        labels[
                            row_idx
                        ].item()
                    ),

                "category_size":
                    n_members,

                "center_distance":
                    center_distance,

                "gt_cosine":
                    float(
                        gt_cosine[
                            row_idx
                        ]
                    ),

                "gt_rank":
                    int(
                        gt_rank[
                            row_idx
                        ]
                    ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Per-category correlation
# ============================================================

def analyze_each_category(
    long_df,
):
    rows = []

    for category, group in (
        long_df.groupby(
            "category"
        )
    ):
        rho_cosine, n_cosine = (
            spearman_rho(
                group[
                    "center_distance"
                ],
                group[
                    "gt_cosine"
                ],
            )
        )

        rho_rank, n_rank = (
            spearman_rho(
                group[
                    "center_distance"
                ],
                group[
                    "gt_rank"
                ],
            )
        )

        # --------------------------------------------
        # 同じカテゴリ内で
        # 中心に近い25% vs 遠い25%
        # --------------------------------------------

        sorted_group = (
            group.sort_values(
                "center_distance"
            )
            .reset_index(
                drop=True
            )
        )

        n = len(
            sorted_group
        )

        quarter_n = max(
            1,
            n // 4,
        )

        near = sorted_group.iloc[
            :quarter_n
        ]

        far = sorted_group.iloc[
            -quarter_n:
        ]

        rows.append({
            "category":
                category,

            "n":
                n,

            "rho_distance_gt_cosine":
                rho_cosine,

            "rho_distance_gt_rank":
                rho_rank,

            "near_center_distance_mean":
                near[
                    "center_distance"
                ].mean(),

            "far_center_distance_mean":
                far[
                    "center_distance"
                ].mean(),

            "near_gt_cosine_mean":
                near[
                    "gt_cosine"
                ].mean(),

            "far_gt_cosine_mean":
                far[
                    "gt_cosine"
                ].mean(),

            "near_gt_rank_median":
                near[
                    "gt_rank"
                ].median(),

            "far_gt_rank_median":
                far[
                    "gt_rank"
                ].median(),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Pooled within-category analysis
# ============================================================

def add_within_category_percentiles(
    long_df,
):
    """
    カテゴリ間の差を消すため、

    各カテゴリの内部で
    0～1の順位に変換する。

    例:
        animalの中で中心距離が上位90%
        clothingの中で中心距離が上位90%

    を同じ基準で扱える。

    multi-labelの刺激は複数カテゴリに現れるので、
    これは補助的な集約指標として使う。
    """

    df = long_df.copy()

    df[
        "distance_percentile"
    ] = (
        df.groupby(
            "category"
        )[
            "center_distance"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    df[
        "gt_rank_percentile"
    ] = (
        df.groupby(
            "category"
        )[
            "gt_rank"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    df[
        "gt_cosine_percentile"
    ] = (
        df.groupby(
            "category"
        )[
            "gt_cosine"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    return df


# ============================================================
# Plot
# ============================================================

def save_rho_histogram(
    values,
    title,
    xlabel,
    output_path,
):
    values = pd.Series(
        values
    ).dropna()

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        values,
        bins=15,
    )

    plt.axvline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Number of categories"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_cache",
        type=str,
        default=(
            "./outputs/"
            "eeg_sample_difficulty/"
            "sub-01_val_features.pt"
        ),
    )

    parser.add_argument(
        "--category_tsv",
        type=str,
        default=(
            "Generation/"
            "category53_long-format.tsv"
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
        "--min_category_size",
        type=int,
        default=10,
        help=(
            "Minimum number of samples "
            "required for within-category "
            "correlation."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "./outputs/"
            "eeg_sample_difficulty/"
            "within_category"
        ),
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load already-extracted EEG / CLIP features
    # ========================================================

    print(
        "Load cached features:"
    )

    print(
        args.feature_cache
    )

    cache = torch.load(
        args.feature_cache,
        map_location="cpu",
    )

    eeg_features = (
        cache[
            "eeg_features"
        ].float()
    )

    teacher_features = (
        cache[
            "teacher_features"
        ].float()
    )

    labels = (
        cache[
            "labels"
        ].long()
    )

    print(
        "EEG:",
        eeg_features.shape,
    )

    print(
        "Teacher:",
        teacher_features.shape,
    )

    # ========================================================
    # 2. THINGSplus
    # ========================================================

    (
        class_to_categories,
        categories,
        folders,
    ) = load_category_mapping(
        args.category_tsv,
        args.img_dir_training,
    )

    # ========================================================
    # 3. Category-specific distances
    # ========================================================

    print(
        "\nCompute within-category "
        "center distances..."
    )

    long_df = (
        compute_within_category_rows(
            teacher_features=(
                teacher_features
            ),
            eeg_features=(
                eeg_features
            ),
            labels=labels,
            class_to_categories=(
                class_to_categories
            ),
            categories=categories,
            min_category_size=(
                args.min_category_size
            ),
        )
    )

    long_path = os.path.join(
        args.output_dir,
        (
            "within_category_"
            "samples.csv"
        ),
    )

    long_df.to_csv(
        long_path,
        index=False,
    )

    # ========================================================
    # 4. Correlation within each category
    # ========================================================

    category_df = (
        analyze_each_category(
            long_df
        )
    )

    category_path = os.path.join(
        args.output_dir,
        (
            "within_category_"
            "correlations.csv"
        ),
    )

    category_df.to_csv(
        category_path,
        index=False,
    )

    # ========================================================
    # 5. Summary
    # ========================================================

    valid_cosine = (
        category_df[
            "rho_distance_gt_cosine"
        ].dropna()
    )

    valid_rank = (
        category_df[
            "rho_distance_gt_rank"
        ].dropna()
    )

    n_categories = len(
        category_df
    )

    cosine_correct = int(
        (
            valid_cosine < 0
        ).sum()
    )

    rank_correct = int(
        (
            valid_rank > 0
        ).sum()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "WITHIN-CATEGORY RESULT"
    )

    print(
        "========================================"
    )

    print(
        "Analyzed categories:",
        n_categories,
    )

    print()

    print(
        "center distance vs GT cosine"
    )

    print(
        "  median rho:",
        f"{valid_cosine.median():.4f}",
    )

    print(
        "  expected direction (rho < 0):",
        f"{cosine_correct}/"
        f"{len(valid_cosine)}",
    )

    print()

    print(
        "center distance vs GT rank"
    )

    print(
        "  median rho:",
        f"{valid_rank.median():.4f}",
    )

    print(
        "  expected direction (rho > 0):",
        f"{rank_correct}/"
        f"{len(valid_rank)}",
    )

    # ========================================================
    # 6. Pooled within-category rank analysis
    # ========================================================

    percentile_df = (
        add_within_category_percentiles(
            long_df
        )
    )

    pooled_rank = (
        percentile_df[
            "distance_percentile"
        ].corr(
            percentile_df[
                "gt_rank_percentile"
            ],
            method="pearson",
        )
    )

    pooled_cosine = (
        percentile_df[
            "distance_percentile"
        ].corr(
            percentile_df[
                "gt_cosine_percentile"
            ],
            method="pearson",
        )
    )

    print()

    print(
        "Pooled within-category rank relation"
    )

    print(
        "  distance vs GT cosine:",
        f"{pooled_cosine:.4f}",
    )

    print(
        "  distance vs GT rank:",
        f"{pooled_rank:.4f}",
    )

    print(
        "  NOTE: multi-label samples can "
        "appear in multiple categories."
    )

    # ========================================================
    # 7. Strongest categories
    # ========================================================

    display_columns = [
        "category",
        "n",
        "rho_distance_gt_cosine",
        "rho_distance_gt_rank",
        "near_gt_cosine_mean",
        "far_gt_cosine_mean",
        "near_gt_rank_median",
        "far_gt_rank_median",
    ]

    print(
        "\n"
        "=== Strongest expected-direction "
        "categories (GT rank) ==="
    )

    print(
        category_df.sort_values(
            "rho_distance_gt_rank",
            ascending=False,
        )[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        "=== Opposite / weak categories "
        "(GT rank) ==="
    )

    print(
        category_df.sort_values(
            "rho_distance_gt_rank",
            ascending=True,
        )[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 8. Histograms
    # ========================================================

    save_rho_histogram(
        valid_rank,
        (
            "Within-category correlation: "
            "center distance vs GT rank"
        ),
        "Spearman rho",
        os.path.join(
            args.output_dir,
            "rho_gt_rank_histogram.png",
        ),
    )

    save_rho_histogram(
        valid_cosine,
        (
            "Within-category correlation: "
            "center distance vs GT cosine"
        ),
        "Spearman rho",
        os.path.join(
            args.output_dir,
            "rho_gt_cosine_histogram.png",
        ),
    )

    print(
        "\nSaved:"
    )

    print(
        long_path
    )

    print(
        category_path
    )

    print(
        "\nAnalysis finished."
    )


if __name__ == "__main__":
    main()