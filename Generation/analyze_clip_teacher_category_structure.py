#!/usr/bin/env python3

"""
Analyze category-wise structure of teacher CLIP features.

For each THINGSplus category, this script measures:

1. intra-category cosine distance
   -> how spread out samples inside the category are

2. inter-category cosine distance
   -> how far category samples are from samples outside the category

3. separation
   -> inter_distance - intra_distance

A larger separation means:
    same-category samples are relatively close
    and different-category samples are relatively far.

The analysis uses the same validation split as Generation/train.py:
    1654 classes
    10 conditions per class
    1 validation condition per class
    seed=42
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


def load_clip_features(features_path):
    """
    Load cached CLIP image features.

    Expected:
        img_features.shape = (16540, 1024)
    """

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

    img_features = (
        saved["img_features"]
        .float()
    )

    expected_samples = (
        N_CLASSES
        * CONDITIONS_PER_CLASS
    )

    if img_features.ndim != 2:
        raise ValueError(
            "img_features must be 2-D, "
            f"but got {tuple(img_features.shape)}"
        )

    if img_features.shape[0] != expected_samples:
        raise ValueError(
            "Unexpected number of image features: "
            f"expected={expected_samples}, "
            f"actual={img_features.shape[0]}"
        )

    return img_features


def get_validation_features(
    all_features,
    seed=42,
    val_ratio=0.1,
):
    """
    Reproduce the validation split used in train.py.

    Returns:
        val_features: (1654, 1024)
    """

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
            "Validation features should contain "
            f"{N_CLASSES} samples, "
            f"but got {val_features.shape[0]}"
        )

    return (
        val_features,
        val_indices,
    )


def load_class_names(
    img_dir_training,
):
    """
    Read the 1654 class names from the THINGS image folders.

    Example folder:
        00001_aardvark

    becomes:
        aardvark
    """

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
            "Expected "
            f"{N_CLASSES} training class folders, "
            f"but found {len(folders)}"
        )

    class_names = []

    for folder in folders:

        if "_" not in folder:
            raise ValueError(
                "Unexpected class folder name: "
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


def load_category_map(
    category_tsv,
):
    """
    Load THINGSplus multi-label categories.

    Returns:
        concept_to_categories

    Example:
        aardvark -> {"animal", "mammal"}
    """

    concept_to_categories = (
        defaultdict(set)
    )

    category_order = []

    seen_categories = set()

    with open(
        category_tsv,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        required_columns = {
            "category",
            "uniqueID",
        }

        missing = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                "Missing columns in category TSV: "
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

            if category not in seen_categories:

                seen_categories.add(
                    category
                )

                category_order.append(
                    category
                )

    return (
        dict(concept_to_categories),
        category_order,
    )


def build_category_indices(
    class_names,
    concept_to_categories,
):
    """
    Build category -> sample-index mapping.

    Only classes that have a THINGSplus mapping are used.

    This is important because an unmapped class should not
    automatically be treated as "not animal", "not food", etc.
    """

    category_to_indices = (
        defaultdict(list)
    )

    mapped_indices = []

    unmapped_classes = []

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

            unmapped_classes.append(
                class_name
            )

            continue

        mapped_indices.append(
            index
        )

        for category in categories:

            category_to_indices[
                category
            ].append(
                index
            )

    return (
        dict(category_to_indices),
        mapped_indices,
        unmapped_classes,
    )


def compute_cosine_distance_matrix(
    features,
):
    """
    features:
        (1654, 1024)

    First L2-normalize each feature.

    similarity:
        (1654, 1654)

    distance:
        1 - cosine similarity
    """

    normalized = F.normalize(
        features,
        dim=-1,
    )

    similarity = (
        normalized
        @ normalized.T
    )

    distance = (
        1.0
        - similarity
    )

    return (
        normalized,
        distance,
    )


def compute_category_statistics(
    normalized_features,
    distance_matrix,
    category_to_indices,
    mapped_indices,
    category_order,
):
    """
    Compute intra/inter distances for every category.
    """

    results = []

    mapped_set = set(
        mapped_indices
    )

    for category in category_order:

        positive_indices = (
            category_to_indices
            .get(
                category,
                [],
            )
        )

        if len(positive_indices) < 2:
            continue

        positive_set = set(
            positive_indices
        )

        negative_indices = sorted(
            mapped_set
            - positive_set
        )

        if len(negative_indices) == 0:
            continue

        pos = torch.tensor(
            positive_indices,
            dtype=torch.long,
        )

        neg = torch.tensor(
            negative_indices,
            dtype=torch.long,
        )

        # --------------------------------------------------
        # 1. Intra-category distance
        # --------------------------------------------------
        #
        # Example:
        # animal:
        #
        # dog <-> cat
        # dog <-> horse
        # cat <-> horse
        # ...
        #
        # Diagonal (self-self) is excluded.
        #

        intra_matrix = (
            distance_matrix[
                pos[:, None],
                pos[None, :],
            ]
        )

        upper_indices = torch.triu_indices(
            intra_matrix.shape[0],
            intra_matrix.shape[1],
            offset=1,
        )

        intra_distances = (
            intra_matrix[
                upper_indices[0],
                upper_indices[1],
            ]
        )

        # --------------------------------------------------
        # 2. Inter-category distance
        # --------------------------------------------------
        #
        # animal vs non-animal
        #

        inter_distances = (
            distance_matrix[
                pos[:, None],
                neg[None, :],
            ]
            .reshape(-1)
        )

        # --------------------------------------------------
        # 3. Distance to category centroid
        # --------------------------------------------------
        #
        # This measures category compactness in another way.
        #

        category_features = (
            normalized_features[
                pos
            ]
        )

        centroid = (
            category_features
            .mean(
                dim=0,
                keepdim=True,
            )
        )

        centroid = F.normalize(
            centroid,
            dim=-1,
        )

        centroid_similarity = (
            category_features
            @ centroid.T
        ).squeeze(1)

        centroid_distances = (
            1.0
            - centroid_similarity
        )

        intra_mean = (
            intra_distances
            .mean()
            .item()
        )

        intra_std = (
            intra_distances
            .std()
            .item()
        )

        inter_mean = (
            inter_distances
            .mean()
            .item()
        )

        inter_std = (
            inter_distances
            .std()
            .item()
        )

        centroid_mean = (
            centroid_distances
            .mean()
            .item()
        )

        centroid_std = (
            centroid_distances
            .std()
            .item()
        )

        # Larger = better separated
        separation = (
            inter_mean
            - intra_mean
        )

        results.append(
            {
                "category": category,

                "n_category": (
                    len(positive_indices)
                ),

                "n_rest": (
                    len(negative_indices)
                ),

                "intra_mean_cosine_distance": (
                    intra_mean
                ),

                "intra_std_cosine_distance": (
                    intra_std
                ),

                "inter_mean_cosine_distance": (
                    inter_mean
                ),

                "inter_std_cosine_distance": (
                    inter_std
                ),

                "centroid_mean_cosine_distance": (
                    centroid_mean
                ),

                "centroid_std_cosine_distance": (
                    centroid_std
                ),

                "separation": (
                    separation
                ),
            }
        )

    return results


def save_results_csv(
    results,
    output_path,
):
    fieldnames = [
        "category",
        "n_category",
        "n_rest",
        "intra_mean_cosine_distance",
        "intra_std_cosine_distance",
        "inter_mean_cosine_distance",
        "inter_std_cosine_distance",
        "centroid_mean_cosine_distance",
        "centroid_std_cosine_distance",
        "separation",
    ]

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


def plot_separation(
    results,
    output_path,
):
    """
    Horizontal bar plot.

    Larger separation means cleaner category structure.
    """

    sorted_results = sorted(
        results,
        key=lambda x: x["separation"],
    )

    categories = [
        row["category"]
        for row in sorted_results
    ]

    values = [
        row["separation"]
        for row in sorted_results
    ]

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                10,
                0.3 * len(categories),
            ),
        )
    )

    y = np.arange(
        len(categories)
    )

    ax.barh(
        y,
        values,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        categories,
        fontsize=8,
    )

    ax.set_xlabel(
        "Inter distance - Intra distance"
    )

    ax.set_ylabel(
        "THINGSplus category"
    )

    ax.set_title(
        "Teacher CLIP category separation"
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_intra_inter(
    results,
    output_path,
):
    """
    Compare intra-category and inter-category distances.
    """

    sorted_results = sorted(
        results,
        key=lambda x: x["separation"],
        reverse=True,
    )

    categories = [
        row["category"]
        for row in sorted_results
    ]

    intra = np.array(
        [
            row[
                "intra_mean_cosine_distance"
            ]
            for row in sorted_results
        ]
    )

    inter = np.array(
        [
            row[
                "inter_mean_cosine_distance"
            ]
            for row in sorted_results
        ]
    )

    y = np.arange(
        len(categories)
    )

    height = 0.38

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(
                10,
                0.32 * len(categories),
            ),
        )
    )

    ax.barh(
        y - height / 2,
        intra,
        height=height,
        label="Intra-category",
    )

    ax.barh(
        y + height / 2,
        inter,
        height=height,
        label="Inter-category",
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        categories,
        fontsize=8,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Mean cosine distance"
    )

    ax.set_ylabel(
        "THINGSplus category"
    )

    ax.set_title(
        "Teacher CLIP: intra vs inter category distance"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(
    results,
):
    print("")
    print(
        "=" * 80
    )

    print(
        "Teacher CLIP category structure"
    )

    print(
        "=" * 80
    )

    print("")
    print(
        "Top 10 categories by separation:"
    )

    print(
        ""
    )

    sorted_results = sorted(
        results,
        key=lambda x: x["separation"],
        reverse=True,
    )

    for row in sorted_results[:10]:

        print(
            f"{row['category']:<28} "
            f"n={row['n_category']:>4}  "
            f"intra={row['intra_mean_cosine_distance']:.4f}  "
            f"inter={row['inter_mean_cosine_distance']:.4f}  "
            f"sep={row['separation']:.4f}"
        )

    print("")
    print(
        "Bottom 10 categories by separation:"
    )

    print("")

    for row in sorted_results[-10:]:

        print(
            f"{row['category']:<28} "
            f"n={row['n_category']:>4}  "
            f"intra={row['intra_mean_cosine_distance']:.4f}  "
            f"inter={row['inter_mean_cosine_distance']:.4f}  "
            f"sep={row['separation']:.4f}"
        )

    print("")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Category-wise structure analysis "
            "of teacher CLIP features."
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
        help=(
            "Path to THINGS training image folders."
        ),
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
            "clip_teacher_category_structure",
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

    # --------------------------------------------------
    # Load teacher CLIP features
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Class/category information
    # --------------------------------------------------

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
        category_to_indices,
        mapped_indices,
        unmapped_classes,
    ) = build_category_indices(
        class_names,
        concept_to_categories,
    )

    print(
        "Mapped classes:",
        len(mapped_indices),
    )

    print(
        "Unmapped classes:",
        len(unmapped_classes),
    )

    # --------------------------------------------------
    # Cosine distance
    # --------------------------------------------------

    (
        normalized_features,
        distance_matrix,
    ) = compute_cosine_distance_matrix(
        val_features
    )

    print(
        "Cosine distance matrix:",
        tuple(
            distance_matrix.shape
        ),
    )

    # --------------------------------------------------
    # Category statistics
    # --------------------------------------------------

    results = (
        compute_category_statistics(
            normalized_features,
            distance_matrix,
            category_to_indices,
            mapped_indices,
            category_order,
        )
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    csv_path = os.path.join(
        args.output_dir,
        "teacher_clip_category_structure.csv",
    )

    separation_plot_path = os.path.join(
        args.output_dir,
        "teacher_clip_category_separation.png",
    )

    intra_inter_plot_path = os.path.join(
        args.output_dir,
        "teacher_clip_intra_inter_distance.png",
    )

    save_results_csv(
        results,
        csv_path,
    )

    plot_separation(
        results,
        separation_plot_path,
    )

    plot_intra_inter(
        results,
        intra_inter_plot_path,
    )

    print_summary(
        results
    )

    print(
        "Saved:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {separation_plot_path}"
    )

    print(
        f"  {intra_inter_plot_path}"
    )


if __name__ == "__main__":
    main()