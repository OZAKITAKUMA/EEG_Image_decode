#!/usr/bin/env python3

"""
Analyze category-to-category contact in teacher CLIP space
after removing samples shared by both categories.

For each category pair A, B:

    intersection = A ∩ B

    A_only = A - B
    B_only = B - A

Then measure kNN contact only between A_only and B_only.

This separates:

1. category overlap caused by THINGSplus multi-label annotations
2. genuine spatial contact between different point sets
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
# CLIP features
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
            f"got {tuple(features.shape)}"
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
            f"got {val_features.shape[0]}"
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
            f"Expected {N_CLASSES} folders, "
            f"found {len(folders)}"
        )

    class_names = []

    for folder in folders:

        if "_" not in folder:
            raise ValueError(
                f"Unexpected folder name: "
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
                f"Missing TSV columns: "
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


def build_category_sets(
    class_names,
    concept_to_categories,
    category_order,
):
    """
    category -> set of CLIP point indices
    """

    category_sets = {
        category: set()
        for category in category_order
    }

    unmapped = []

    for index, class_name in enumerate(
        class_names
    ):

        categories = (
            concept_to_categories
            .get(
                class_name,
                set(),
            )
        )

        if not categories:
            unmapped.append(
                class_name
            )
            continue

        for category in categories:

            if category in category_sets:

                category_sets[
                    category
                ].add(
                    index
                )

    return (
        category_sets,
        unmapped,
    )


# ============================================================
# Global kNN graph
# ============================================================

def compute_knn(
    features,
    k=5,
):
    """
    Compute k nearest neighbors for every CLIP point.

    The neighborhood is computed in the full 1654-point space.
    """

    normalized = F.normalize(
        features,
        dim=-1,
    )

    similarity = (
        normalized
        @ normalized.T
    )

    # Never retrieve itself.
    similarity.fill_diagonal_(
        -float("inf")
    )

    _, knn_indices = torch.topk(
        similarity,
        k=k,
        dim=1,
    )

    neighbor_sets = [
        set(
            row.tolist()
        )
        for row in knn_indices.cpu()
    ]

    return neighbor_sets


# ============================================================
# Contact for a category pair
# ============================================================

def compute_pair_contact(
    category_a,
    category_b,
    category_sets,
    neighbor_sets,
    n_points,
    k,
):
    """
    Remove the shared samples first.

        intersection = A & B
        A_only = A - B
        B_only = B - A

    Then count global-kNN edges crossing between
    A_only and B_only.
    """

    set_a = category_sets[
        category_a
    ]

    set_b = category_sets[
        category_b
    ]

    intersection = (
        set_a
        & set_b
    )

    a_only = (
        set_a
        - set_b
    )

    b_only = (
        set_b
        - set_a
    )

    # --------------------------------------------------------
    # Directed A -> B contact
    # --------------------------------------------------------

    a_to_b_edges = 0

    for source in a_only:

        a_to_b_edges += sum(
            neighbor in b_only
            for neighbor
            in neighbor_sets[source]
        )

    # --------------------------------------------------------
    # Directed B -> A contact
    # --------------------------------------------------------

    b_to_a_edges = 0

    for source in b_only:

        b_to_a_edges += sum(
            neighbor in a_only
            for neighbor
            in neighbor_sets[source]
        )

    # --------------------------------------------------------
    # Contact rates
    # --------------------------------------------------------

    if len(a_only) > 0:

        a_to_b_rate = (
            a_to_b_edges
            /
            (
                len(a_only)
                * k
            )
        )

    else:

        a_to_b_rate = np.nan

    if len(b_only) > 0:

        b_to_a_rate = (
            b_to_a_edges
            /
            (
                len(b_only)
                * k
            )
        )

    else:

        b_to_a_rate = np.nan

    # --------------------------------------------------------
    # Random expectation
    #
    # Since A_only and B_only are disjoint,
    # a source point can never itself belong to
    # the target set.
    # --------------------------------------------------------

    if len(a_only) > 0:

        expected_a_to_b = (
            len(b_only)
            / (n_points - 1)
        )

    else:

        expected_a_to_b = np.nan

    if len(b_only) > 0:

        expected_b_to_a = (
            len(a_only)
            / (n_points - 1)
        )

    else:

        expected_b_to_a = np.nan

    # --------------------------------------------------------
    # Enrichment
    # --------------------------------------------------------

    if (
        np.isfinite(a_to_b_rate)
        and expected_a_to_b > 0
    ):

        enrichment_a_to_b = (
            a_to_b_rate
            / expected_a_to_b
        )

    else:

        enrichment_a_to_b = np.nan

    if (
        np.isfinite(b_to_a_rate)
        and expected_b_to_a > 0
    ):

        enrichment_b_to_a = (
            b_to_a_rate
            / expected_b_to_a
        )

    else:

        enrichment_b_to_a = np.nan

    # --------------------------------------------------------
    # Mutual kNN pairs
    #
    # i in A_only, j in B_only
    #
    # j in kNN(i)
    # AND
    # i in kNN(j)
    # --------------------------------------------------------

    mutual_pairs = 0

    for point_a in a_only:

        for point_b in (
            neighbor_sets[
                point_a
            ]
        ):

            if point_b not in b_only:
                continue

            if point_a in (
                neighbor_sets[
                    point_b
                ]
            ):

                mutual_pairs += 1

    # Each A-B pair is visited from A only,
    # so no divide-by-two is needed.

    # --------------------------------------------------------
    # Symmetric summary
    # --------------------------------------------------------

    total_edges = (
        a_to_b_edges
        + b_to_a_edges
    )

    valid_enrichments = [
        value
        for value in (
            enrichment_a_to_b,
            enrichment_b_to_a,
        )
        if np.isfinite(value)
    ]

    if valid_enrichments:

        symmetric_enrichment = (
            np.mean(
                valid_enrichments
            )
        )

    else:

        symmetric_enrichment = np.nan

    # Jaccard overlap before removing intersection.
    union = (
        set_a
        | set_b
    )

    if len(union) > 0:

        jaccard = (
            len(intersection)
            / len(union)
        )

    else:

        jaccard = np.nan

    return {
        "category_a": category_a,
        "category_b": category_b,

        "a_total_n": len(set_a),
        "b_total_n": len(set_b),

        "intersection_n": (
            len(intersection)
        ),

        "jaccard_overlap": (
            jaccard
        ),

        "a_only_n": len(a_only),
        "b_only_n": len(b_only),

        "a_to_b_edges": (
            a_to_b_edges
        ),

        "b_to_a_edges": (
            b_to_a_edges
        ),

        "total_cross_edges": (
            total_edges
        ),

        "a_to_b_contact_rate": (
            a_to_b_rate
        ),

        "b_to_a_contact_rate": (
            b_to_a_rate
        ),

        "expected_a_to_b_rate": (
            expected_a_to_b
        ),

        "expected_b_to_a_rate": (
            expected_b_to_a
        ),

        "a_to_b_enrichment": (
            enrichment_a_to_b
        ),

        "b_to_a_enrichment": (
            enrichment_b_to_a
        ),

        "symmetric_enrichment": (
            symmetric_enrichment
        ),

        "mutual_knn_pairs": (
            mutual_pairs
        ),
    }


# ============================================================
# All category pairs
# ============================================================

def analyze_all_pairs(
    category_order,
    category_sets,
    neighbor_sets,
    n_points,
    k,
):
    results = []

    for i in range(
        len(category_order)
    ):

        for j in range(
            i + 1,
            len(category_order),
        ):

            result = compute_pair_contact(
                category_order[i],
                category_order[j],
                category_sets,
                neighbor_sets,
                n_points,
                k,
            )

            results.append(
                result
            )

    return results


# ============================================================
# Save long-format CSV
# ============================================================

def save_results_csv(
    results,
    output_path,
):
    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                row
            )


# ============================================================
# Build heatmap matrices
# ============================================================

def build_pair_matrix(
    results,
    category_order,
    value_key,
):
    n = len(
        category_order
    )

    matrix = np.full(
        (
            n,
            n,
        ),
        np.nan,
        dtype=np.float64,
    )

    category_to_id = {
        category: index
        for index, category
        in enumerate(category_order)
    }

    for row in results:

        i = category_to_id[
            row["category_a"]
        ]

        j = category_to_id[
            row["category_b"]
        ]

        value = row[
            value_key
        ]

        matrix[
            i,
            j,
        ] = value

        matrix[
            j,
            i,
        ] = value

    # Diagonal has no A-vs-B meaning.
    np.fill_diagonal(
        matrix,
        np.nan,
    )

    return matrix


# ============================================================
# Heatmap
# ============================================================

def plot_heatmap(
    matrix,
    category_order,
    title,
    colorbar_label,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(18, 16)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
    )

    ticks = np.arange(
        len(category_order)
    )

    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
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
        "Category B"
    )

    ax.set_ylabel(
        "Category A"
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
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Terminal summary
# ============================================================

def print_summary(
    results,
):
    print("")
    print(
        "=" * 90
    )

    print(
        "Largest category overlaps"
    )

    print(
        "=" * 90
    )

    for row in sorted(
        results,
        key=lambda x: x[
            "intersection_n"
        ],
        reverse=True,
    )[:15]:

        print(
            f"{row['category_a']:<24} "
            f"<-> {row['category_b']:<24} "
            f"intersection={row['intersection_n']:>3}  "
            f"Aonly={row['a_only_n']:>3}  "
            f"Bonly={row['b_only_n']:>3}  "
            f"Jaccard={row['jaccard_overlap']:.3f}"
        )

    print("")
    print(
        "=" * 90
    )

    print(
        "Strongest spatial contacts AFTER removing shared classes"
    )

    print(
        "(pairs require >= 5 total cross-kNN edges)"
    )

    print(
        "=" * 90
    )

    eligible = [
        row
        for row in results
        if (
            row["total_cross_edges"]
            >= 5
            and np.isfinite(
                row[
                    "symmetric_enrichment"
                ]
            )
        )
    ]

    for row in sorted(
        eligible,
        key=lambda x: x[
            "symmetric_enrichment"
        ],
        reverse=True,
    )[:20]:

        print(
            f"{row['category_a']:<24} "
            f"<-> {row['category_b']:<24} "
            f"overlap={row['intersection_n']:>3}  "
            f"edges={row['total_cross_edges']:>3}  "
            f"mutual={row['mutual_knn_pairs']:>3}  "
            f"enrich={row['symmetric_enrichment']:.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Teacher CLIP category contact analysis "
            "after removing multi-label intersections."
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
            "clip_teacher_knn_disjoint_contact",
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
    # Teacher CLIP features
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
        "Validation CLIP features:",
        tuple(
            val_features.shape
        ),
    )

    # --------------------------------------------------------
    # Categories
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
        category_sets,
        unmapped,
    ) = build_category_sets(
        class_names,
        concept_to_categories,
        category_order,
    )

    print(
        "Categories:",
        len(category_order),
    )

    print(
        "Unmapped classes:",
        len(unmapped),
    )

    # --------------------------------------------------------
    # Global kNN
    # --------------------------------------------------------

    neighbor_sets = compute_knn(
        val_features,
        k=args.k,
    )

    # --------------------------------------------------------
    # Pairwise disjoint analysis
    # --------------------------------------------------------

    results = analyze_all_pairs(
        category_order,
        category_sets,
        neighbor_sets,
        n_points=len(
            val_features
        ),
        k=args.k,
    )

    print(
        "Category pairs:",
        len(results),
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        args.output_dir,
        "teacher_clip_disjoint_knn_pairs.csv",
    )

    save_results_csv(
        results,
        csv_path,
    )

    # --------------------------------------------------------
    # Matrices
    # --------------------------------------------------------

    intersection_matrix = (
        build_pair_matrix(
            results,
            category_order,
            "intersection_n",
        )
    )

    enrichment_matrix = (
        build_pair_matrix(
            results,
            category_order,
            "symmetric_enrichment",
        )
    )

    edge_matrix = (
        build_pair_matrix(
            results,
            category_order,
            "total_cross_edges",
        )
    )

    mutual_matrix = (
        build_pair_matrix(
            results,
            category_order,
            "mutual_knn_pairs",
        )
    )

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------

    intersection_path = os.path.join(
        args.output_dir,
        "category_intersection_heatmap.png",
    )

    enrichment_path = os.path.join(
        args.output_dir,
        "disjoint_contact_enrichment_heatmap.png",
    )

    edge_path = os.path.join(
        args.output_dir,
        "disjoint_cross_edge_heatmap.png",
    )

    mutual_path = os.path.join(
        args.output_dir,
        "disjoint_mutual_knn_heatmap.png",
    )

    plot_heatmap(
        intersection_matrix,
        category_order,
        title=(
            "THINGSplus category overlap"
        ),
        colorbar_label=(
            "Number of shared classes"
        ),
        output_path=intersection_path,
    )

    plot_heatmap(
        enrichment_matrix,
        category_order,
        title=(
            f"Teacher CLIP spatial contact "
            f"after removing shared classes "
            f"(k={args.k})"
        ),
        colorbar_label=(
            "Observed / random expected contact"
        ),
        output_path=enrichment_path,
    )

    plot_heatmap(
        edge_matrix,
        category_order,
        title=(
            f"Teacher CLIP cross-category "
            f"kNN edges after removing overlap "
            f"(k={args.k})"
        ),
        colorbar_label=(
            "Cross-kNN edge count"
        ),
        output_path=edge_path,
    )

    plot_heatmap(
        mutual_matrix,
        category_order,
        title=(
            f"Teacher CLIP mutual kNN contact "
            f"after removing overlap "
            f"(k={args.k})"
        ),
        colorbar_label=(
            "Mutual kNN pair count"
        ),
        output_path=mutual_path,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_summary(
        results
    )

    print("")
    print(
        "Saved:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {intersection_path}"
    )

    print(
        f"  {enrichment_path}"
    )

    print(
        f"  {edge_path}"
    )

    print(
        f"  {mutual_path}"
    )


if __name__ == "__main__":
    main()