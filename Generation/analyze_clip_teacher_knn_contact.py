#!/usr/bin/env python3

"""
Analyze category-to-category local contact structure
in teacher CLIP feature space using k-nearest neighbors.

For each validation teacher CLIP feature:
    - find k nearest other CLIP features
    - count which THINGSplus categories appear in its neighborhood

Outputs:
    1. directed kNN edge counts
    2. category-to-category contact rates
    3. random-expected contact rates
    4. contact enrichment
    5. mutual-kNN category pair counts

Important:
    THINGSplus categories are multi-label.

Therefore one CLIP point can contribute to multiple category pairs.
For example:
    aardvark -> {"animal", "mammal"}

This is intentional because categories are treated as sets.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from encoder_utils import stratified_condition_split


N_CLASSES = 1654
CONDITIONS_PER_CLASS = 10

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REPO_ROOT = os.path.dirname(
    SCRIPT_DIR
)


# ============================================================
# Feature loading
# ============================================================

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
            f"'img_features' not found in: "
            f"{features_path}"
        )

    features = (
        saved["img_features"]
        .float()
    )

    expected = (
        N_CLASSES
        * CONDITIONS_PER_CLASS
    )

    if features.ndim != 2:
        raise ValueError(
            "img_features must be 2-D, "
            f"but got {tuple(features.shape)}"
        )

    if features.shape[0] != expected:
        raise ValueError(
            "Unexpected number of features: "
            f"expected={expected}, "
            f"actual={features.shape[0]}"
        )

    return features


# ============================================================
# Validation split
# ============================================================

def get_validation_features(
    all_features,
    seed=42,
    val_ratio=0.1,
):
    _, val_indices = (
        stratified_condition_split(
            n_classes=N_CLASSES,
            conditions_per_class=CONDITIONS_PER_CLASS,
            trials_per_condition=1,
            val_ratio=val_ratio,
            seed=seed,
        )
    )

    val_features = (
        all_features[
            val_indices
        ]
        .clone()
    )

    if val_features.shape[0] != N_CLASSES:
        raise RuntimeError(
            "Expected one validation feature "
            f"per class ({N_CLASSES}), "
            f"but got {val_features.shape[0]}"
        )

    return (
        val_features,
        val_indices,
    )


# ============================================================
# Class names
# ============================================================

def load_class_names(
    img_dir_training,
):
    folders = sorted(
        folder
        for folder in os.listdir(
            img_dir_training
        )
        if os.path.isdir(
            os.path.join(
                img_dir_training,
                folder,
            )
        )
    )

    if len(folders) != N_CLASSES:
        raise ValueError(
            f"Expected {N_CLASSES} class folders, "
            f"but found {len(folders)}"
        )

    class_names = []

    for folder in folders:

        if "_" not in folder:
            raise ValueError(
                "Unexpected folder name: "
                f"{folder}"
            )

        class_name = (
            folder[
                folder.index("_") + 1:
            ]
        )

        class_names.append(
            class_name
        )

    return class_names


# ============================================================
# THINGSplus categories
# ============================================================

def load_category_map(
    category_tsv,
):
    concept_to_categories = (
        defaultdict(set)
    )

    category_order = []

    seen = set()

    with open(
        category_tsv,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        required = {
            "category",
            "uniqueID",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                "Missing TSV columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            category = (
                row["category"]
                .strip()
            )

            unique_id = (
                row["uniqueID"]
                .strip()
            )

            if (
                not category
                or not unique_id
            ):
                continue

            concept_to_categories[
                unique_id
            ].add(
                category
            )

            if category not in seen:

                seen.add(
                    category
                )

                category_order.append(
                    category
                )

    return (
        dict(concept_to_categories),
        category_order,
    )


def build_point_category_sets(
    class_names,
    concept_to_categories,
):
    point_categories = []

    unmapped = []

    for class_name in class_names:

        categories = (
            concept_to_categories
            .get(
                class_name,
                set(),
            )
        )

        point_categories.append(
            set(categories)
        )

        if not categories:
            unmapped.append(
                class_name
            )

    return (
        point_categories,
        unmapped,
    )


# ============================================================
# kNN
# ============================================================

def compute_knn(
    features,
    k,
):
    """
    features:
        [1654, 1024]

    Returns:
        knn_indices:
            [1654, k]

    Self-neighbor is excluded.
    """

    normalized = F.normalize(
        features,
        dim=-1,
    )

    similarity = (
        normalized
        @ normalized.T
    )

    # Do not retrieve itself.
    similarity.fill_diagonal_(
        -float("inf")
    )

    _, knn_indices = torch.topk(
        similarity,
        k=k,
        dim=1,
    )

    return knn_indices.cpu()


# ============================================================
# Category contact
# ============================================================

def compute_contact_matrices(
    point_categories,
    category_order,
    knn_indices,
):
    """
    Count directed category-to-category kNN edges.

    Example:

        source point categories:
            {animal, mammal}

        neighbor categories:
            {animal, bird}

    This edge contributes to:

        animal -> animal
        animal -> bird
        mammal -> animal
        mammal -> bird
    """

    n_categories = len(
        category_order
    )

    category_to_id = {
        category: i
        for i, category
        in enumerate(category_order)
    }

    directed_counts = np.zeros(
        (
            n_categories,
            n_categories,
        ),
        dtype=np.int64,
    )

    source_point_counts = np.zeros(
        n_categories,
        dtype=np.int64,
    )

    category_point_counts = np.zeros(
        n_categories,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Number of points belonging to each category
    # --------------------------------------------------------

    for categories in point_categories:

        for category in categories:

            category_id = (
                category_to_id[
                    category
                ]
            )

            category_point_counts[
                category_id
            ] += 1

    # --------------------------------------------------------
    # Directed kNN edges
    # --------------------------------------------------------

    for source_index in range(
        len(point_categories)
    ):

        source_categories = (
            point_categories[
                source_index
            ]
        )

        if not source_categories:
            continue

        for source_category in source_categories:

            source_id = (
                category_to_id[
                    source_category
                ]
            )

            source_point_counts[
                source_id
            ] += 1

        for neighbor_index in (
            knn_indices[
                source_index
            ]
            .tolist()
        ):

            target_categories = (
                point_categories[
                    neighbor_index
                ]
            )

            if not target_categories:
                continue

            for source_category in (
                source_categories
            ):

                source_id = (
                    category_to_id[
                        source_category
                    ]
                )

                for target_category in (
                    target_categories
                ):

                    target_id = (
                        category_to_id[
                            target_category
                        ]
                    )

                    directed_counts[
                        source_id,
                        target_id,
                    ] += 1

    return (
        directed_counts,
        source_point_counts,
        category_point_counts,
        category_to_id,
    )


# ============================================================
# Contact rate
# ============================================================

def compute_contact_rates(
    directed_counts,
    source_point_counts,
    k,
):
    """
    contact_rate[A, B]

    = number of A -> B category edges
      --------------------------------
             |A| * k

    Because target points can have multiple labels,
    one row does NOT necessarily sum to 1.
    """

    denominator = (
        source_point_counts[:, None]
        * k
    )

    rates = np.divide(
        directed_counts,
        denominator,
        out=np.zeros_like(
            directed_counts,
            dtype=np.float64,
        ),
        where=denominator != 0,
    )

    return rates


# ============================================================
# Random expectation
# ============================================================

def compute_random_expected_rates(
    point_categories,
    category_order,
    category_point_counts,
    category_to_id,
):
    """
    Expected probability of seeing category B
    if a neighbor were chosen randomly from all other points.

    Self-exclusion and multi-label overlap are handled exactly.

    expected[A, B]
    =
    average over source points i in A:

        (# B points excluding i if i is also B)
        ----------------------------------------
                      N - 1
    """

    n_points = len(
        point_categories
    )

    n_categories = len(
        category_order
    )

    expected = np.zeros(
        (
            n_categories,
            n_categories,
        ),
        dtype=np.float64,
    )

    for source_category in (
        category_order
    ):

        source_id = (
            category_to_id[
                source_category
            ]
        )

        source_indices = [
            i
            for i, categories
            in enumerate(
                point_categories
            )
            if source_category
            in categories
        ]

        if not source_indices:
            continue

        for target_category in (
            category_order
        ):

            target_id = (
                category_to_id[
                    target_category
                ]
            )

            target_count = (
                category_point_counts[
                    target_id
                ]
            )

            probabilities = []

            for source_index in (
                source_indices
            ):

                source_is_target = (
                    target_category
                    in point_categories[
                        source_index
                    ]
                )

                available_target_count = (
                    target_count
                    - int(
                        source_is_target
                    )
                )

                probability = (
                    available_target_count
                    / (n_points - 1)
                )

                probabilities.append(
                    probability
                )

            expected[
                source_id,
                target_id,
            ] = np.mean(
                probabilities
            )

    return expected


# ============================================================
# Enrichment
# ============================================================

def compute_enrichment(
    observed_rates,
    expected_rates,
):
    enrichment = np.divide(
        observed_rates,
        expected_rates,
        out=np.zeros_like(
            observed_rates,
            dtype=np.float64,
        ),
        where=expected_rates > 0,
    )

    excess = (
        observed_rates
        - expected_rates
    )

    return (
        enrichment,
        excess,
    )


# ============================================================
# Mutual kNN
# ============================================================

def compute_mutual_knn_counts(
    point_categories,
    category_order,
    knn_indices,
):
    """
    Count mutual-kNN point pairs.

    A pair i-j is mutual if:

        j is in kNN(i)

    AND

        i is in kNN(j)

    Each unordered point pair is counted once.
    """

    n_categories = len(
        category_order
    )

    category_to_id = {
        category: i
        for i, category
        in enumerate(category_order)
    }

    mutual_counts = np.zeros(
        (
            n_categories,
            n_categories,
        ),
        dtype=np.int64,
    )

    neighbor_sets = [
        set(
            row.tolist()
        )
        for row in knn_indices
    ]

    n_points = len(
        point_categories
    )

    for i in range(n_points):

        for j in neighbor_sets[i]:

            # Count each point pair once.
            if i >= j:
                continue

            if i not in neighbor_sets[j]:
                continue

            categories_i = (
                point_categories[i]
            )

            categories_j = (
                point_categories[j]
            )

            if (
                not categories_i
                or not categories_j
            ):
                continue

            # A multi-label point pair may generate
            # several category pairs.
            # Use a set so the same category pair is
            # counted only once for this point pair.
            category_pairs = set()

            for category_i in (
                categories_i
            ):

                id_i = (
                    category_to_id[
                        category_i
                    ]
                )

                for category_j in (
                    categories_j
                ):

                    id_j = (
                        category_to_id[
                            category_j
                        ]
                    )

                    low = min(
                        id_i,
                        id_j,
                    )

                    high = max(
                        id_i,
                        id_j,
                    )

                    category_pairs.add(
                        (
                            low,
                            high,
                        )
                    )

            for low, high in (
                category_pairs
            ):

                mutual_counts[
                    low,
                    high,
                ] += 1

                if low != high:

                    mutual_counts[
                        high,
                        low,
                    ] += 1

    return mutual_counts


# ============================================================
# Save matrix CSV
# ============================================================

def save_matrix_csv(
    matrix,
    category_order,
    path,
):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "source_category",
                *category_order,
            ]
        )

        for i, category in enumerate(
            category_order
        ):

            writer.writerow(
                [
                    category,
                    *matrix[i].tolist(),
                ]
            )


# ============================================================
# Save long-format CSV
# ============================================================

def save_long_csv(
    category_order,
    directed_counts,
    contact_rates,
    expected_rates,
    enrichment,
    excess,
    mutual_counts,
    source_point_counts,
    path,
):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "source_category",
                "target_category",
                "source_n",
                "directed_knn_edge_count",
                "contact_rate",
                "random_expected_rate",
                "enrichment",
                "excess_rate",
                "mutual_knn_pair_count",
            ]
        )

        for i, source_category in enumerate(
            category_order
        ):

            for j, target_category in enumerate(
                category_order
            ):

                writer.writerow(
                    [
                        source_category,
                        target_category,
                        source_point_counts[i],
                        directed_counts[i, j],
                        contact_rates[i, j],
                        expected_rates[i, j],
                        enrichment[i, j],
                        excess[i, j],
                        mutual_counts[i, j],
                    ]
                )


# ============================================================
# Heatmap
# ============================================================

def plot_heatmap(
    matrix,
    category_order,
    title,
    colorbar_label,
    path,
):
    fig, ax = plt.subplots(
        figsize=(18, 16)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(category_order)
        )
    )

    ax.set_yticks(
        np.arange(
            len(category_order)
        )
    )

    ax.set_xticklabels(
        category_order,
        rotation=90,
        fontsize=6,
    )

    ax.set_yticklabels(
        category_order,
        fontsize=6,
    )

    ax.set_xlabel(
        "Neighbor category"
    )

    ax.set_ylabel(
        "Source category"
    )

    ax.set_title(
        title
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        colorbar_label
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Console summary
# ============================================================

def print_top_contacts(
    category_order,
    directed_counts,
    contact_rates,
    enrichment,
    mutual_counts,
):
    pairs = []

    n_categories = len(
        category_order
    )

    for i in range(
        n_categories
    ):

        for j in range(
            i + 1,
            n_categories,
        ):

            total_edges = (
                directed_counts[i, j]
                + directed_counts[j, i]
            )

            symmetric_enrichment = (
                enrichment[i, j]
                + enrichment[j, i]
            ) / 2.0

            mutual = (
                mutual_counts[
                    i,
                    j,
                ]
            )

            pairs.append(
                {
                    "a": category_order[i],
                    "b": category_order[j],
                    "edges": total_edges,
                    "enrichment": (
                        symmetric_enrichment
                    ),
                    "mutual": mutual,
                    "a_to_b": (
                        contact_rates[
                            i,
                            j,
                        ]
                    ),
                    "b_to_a": (
                        contact_rates[
                            j,
                            i,
                        ]
                    ),
                }
            )

    print("")
    print(
        "=" * 80
    )

    print(
        "Top category pairs by total kNN edge count"
    )

    print(
        "=" * 80
    )

    for row in sorted(
        pairs,
        key=lambda x: x["edges"],
        reverse=True,
    )[:15]:

        print(
            f"{row['a']:<24} "
            f"<-> {row['b']:<24} "
            f"edges={row['edges']:>4}  "
            f"mutual={row['mutual']:>3}  "
            f"enrich={row['enrichment']:.2f}"
        )

    print("")
    print(
        "=" * 80
    )

    print(
        "Top category pairs by contact enrichment"
    )

    print(
        "(only pairs with at least 5 directed edges)"
    )

    print(
        "=" * 80
    )

    eligible = [
        row
        for row in pairs
        if row["edges"] >= 5
    ]

    for row in sorted(
        eligible,
        key=lambda x: x["enrichment"],
        reverse=True,
    )[:15]:

        print(
            f"{row['a']:<24} "
            f"<-> {row['b']:<24} "
            f"edges={row['edges']:>4}  "
            f"mutual={row['mutual']:>3}  "
            f"enrich={row['enrichment']:.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Category-to-category kNN contact "
            "analysis in teacher CLIP space."
        )
    )

    parser.add_argument(
        "--features_path",
        default=os.path.join(
            REPO_ROOT,
            "features",
            "ViT-H-14_features_train.pt",
        ),
    )

    parser.add_argument(
        "--img_dir_training",
        required=True,
    )

    parser.add_argument(
        "--category_tsv",
        default=os.path.join(
            SCRIPT_DIR,
            "category53_long-format.tsv",
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            REPO_ROOT,
            "outputs",
            "clip_teacher_knn_contact",
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
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

    # --------------------------------------------------------
    # CLIP teacher features
    # --------------------------------------------------------

    all_features = (
        load_clip_features(
            args.features_path
        )
    )

    (
        val_features,
        val_indices,
    ) = get_validation_features(
        all_features,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    print(
        "Validation teacher CLIP features:",
        tuple(
            val_features.shape
        ),
    )

    # --------------------------------------------------------
    # Category labels
    # --------------------------------------------------------

    class_names = (
        load_class_names(
            args.img_dir_training
        )
    )

    (
        concept_to_categories,
        category_order,
    ) = load_category_map(
        args.category_tsv
    )

    (
        point_categories,
        unmapped,
    ) = build_point_category_sets(
        class_names,
        concept_to_categories,
    )

    print(
        "Number of categories:",
        len(category_order),
    )

    print(
        "Mapped classes:",
        sum(
            bool(categories)
            for categories
            in point_categories
        ),
    )

    print(
        "Unmapped classes:",
        len(unmapped),
    )

    # --------------------------------------------------------
    # kNN
    # --------------------------------------------------------

    knn_indices = compute_knn(
        val_features,
        k=args.k,
    )

    print(
        "kNN indices:",
        tuple(
            knn_indices.shape
        ),
    )

    # --------------------------------------------------------
    # Contact matrices
    # --------------------------------------------------------

    (
        directed_counts,
        source_point_counts,
        category_point_counts,
        category_to_id,
    ) = compute_contact_matrices(
        point_categories,
        category_order,
        knn_indices,
    )

    contact_rates = (
        compute_contact_rates(
            directed_counts,
            source_point_counts,
            args.k,
        )
    )

    expected_rates = (
        compute_random_expected_rates(
            point_categories,
            category_order,
            category_point_counts,
            category_to_id,
        )
    )

    (
        enrichment,
        excess,
    ) = compute_enrichment(
        contact_rates,
        expected_rates,
    )

    mutual_counts = (
        compute_mutual_knn_counts(
            point_categories,
            category_order,
            knn_indices,
        )
    )

    # --------------------------------------------------------
    # Save paths
    # --------------------------------------------------------

    counts_path = os.path.join(
        args.output_dir,
        "teacher_clip_knn_edge_counts.csv",
    )

    rate_path = os.path.join(
        args.output_dir,
        "teacher_clip_knn_contact_rates.csv",
    )

    expected_path = os.path.join(
        args.output_dir,
        "teacher_clip_knn_random_expected_rates.csv",
    )

    enrichment_path = os.path.join(
        args.output_dir,
        "teacher_clip_knn_enrichment.csv",
    )

    mutual_path = os.path.join(
        args.output_dir,
        "teacher_clip_mutual_knn_counts.csv",
    )

    long_path = os.path.join(
        args.output_dir,
        "teacher_clip_knn_contact_long.csv",
    )

    rate_heatmap = os.path.join(
        args.output_dir,
        "teacher_clip_knn_contact_rate_heatmap.png",
    )

    enrichment_heatmap = os.path.join(
        args.output_dir,
        "teacher_clip_knn_enrichment_heatmap.png",
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_matrix_csv(
        directed_counts,
        category_order,
        counts_path,
    )

    save_matrix_csv(
        contact_rates,
        category_order,
        rate_path,
    )

    save_matrix_csv(
        expected_rates,
        category_order,
        expected_path,
    )

    save_matrix_csv(
        enrichment,
        category_order,
        enrichment_path,
    )

    save_matrix_csv(
        mutual_counts,
        category_order,
        mutual_path,
    )

    save_long_csv(
        category_order,
        directed_counts,
        contact_rates,
        expected_rates,
        enrichment,
        excess,
        mutual_counts,
        source_point_counts,
        long_path,
    )

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------

    plot_heatmap(
        contact_rates * 100.0,
        category_order,
        title=(
            f"Teacher CLIP category contact "
            f"(k={args.k})"
        ),
        colorbar_label=(
            "Neighbor slots containing category (%)"
        ),
        path=rate_heatmap,
    )

    plot_heatmap(
        enrichment,
        category_order,
        title=(
            f"Teacher CLIP category contact enrichment "
            f"(k={args.k})"
        ),
        colorbar_label=(
            "Observed / random expectation"
        ),
        path=enrichment_heatmap,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_top_contacts(
        category_order,
        directed_counts,
        contact_rates,
        enrichment,
        mutual_counts,
    )

    print("")
    print(
        "Saved outputs:"
    )

    print(
        f"  {counts_path}"
    )

    print(
        f"  {rate_path}"
    )

    print(
        f"  {expected_path}"
    )

    print(
        f"  {enrichment_path}"
    )

    print(
        f"  {mutual_path}"
    )

    print(
        f"  {long_path}"
    )

    print(
        f"  {rate_heatmap}"
    )

    print(
        f"  {enrichment_heatmap}"
    )


if __name__ == "__main__":
    main()