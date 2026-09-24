#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def load_category_map(category_tsv):
    concept_to_categories = defaultdict(set)
    category_order = []
    seen = set()

    with open(category_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"category", "uniqueID"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"category TSV is missing columns: {sorted(missing)}"
            )

        for row in reader:
            category = row["category"].strip()
            unique_id = row["uniqueID"].strip()

            if not category or not unique_id:
                continue

            concept_to_categories[unique_id].add(category)

            if category not in seen:
                seen.add(category)
                category_order.append(category)

    return dict(concept_to_categories), category_order


def load_class_names(img_dir_training):
    folders = sorted(
        folder
        for folder in os.listdir(img_dir_training)
        if os.path.isdir(os.path.join(img_dir_training, folder))
    )

    class_names = []

    for folder in folders:
        if "_" not in folder:
            raise ValueError(
                f"Unexpected training-class folder name: {folder}"
            )
        class_names.append(folder[folder.index("_") + 1:])

    return class_names


def load_clip_class_features(features_path, n_classes, images_per_class=10):
    saved = torch.load(
        features_path,
        map_location="cpu",
        weights_only=False,
    )

    if "img_features" not in saved:
        raise KeyError(
            f"'img_features' not found in: {features_path}"
        )

    img_features = saved["img_features"].float()
    expected = n_classes * images_per_class

    if img_features.ndim != 2:
        raise ValueError(
            f"Expected 2-D img_features, got shape={tuple(img_features.shape)}"
        )

    if img_features.shape[0] != expected:
        raise ValueError(
            "Unexpected number of image features: "
            f"expected={expected}, actual={img_features.shape[0]}"
        )

    # Match Generation/train.py:
    # img_features_per_class = img_features_all[::10]
    class_features = img_features[::images_per_class].clone()

    if class_features.shape[0] != n_classes:
        raise RuntimeError(
            f"Expected {n_classes} class features, "
            f"got {class_features.shape[0]}"
        )

    return class_features


def cosine_topk_excluding_self(class_features, k=5):
    x = F.normalize(class_features, dim=-1)

    # (N, D) @ (D, N) -> (N, N)
    similarity = x @ x.T

    # Exclude the query class itself.
    similarity.fill_diagonal_(-float("inf"))

    topk_similarity, topk_indices = torch.topk(
        similarity,
        k=k,
        dim=1,
    )

    return topk_indices.cpu(), topk_similarity.cpu()


def analyze_category_retention(
    class_names,
    topk_indices,
    concept_to_categories,
    max_k=5,
):
    gt_total = Counter()
    topk_hits = defaultdict(Counter)
    uncategorized_gt = []

    for gt_index, gt_name in enumerate(class_names):
        gt_categories = concept_to_categories.get(gt_name, set())

        if not gt_categories:
            uncategorized_gt.append(gt_name)
            continue

        neighbor_categories = []

        for rank_index in range(max_k):
            pred_index = int(topk_indices[gt_index, rank_index])
            pred_name = class_names[pred_index]
            pred_categories = concept_to_categories.get(
                pred_name,
                set(),
            )
            neighbor_categories.append(pred_categories)

        for gt_category in gt_categories:
            gt_total[gt_category] += 1

            for k in range(1, max_k + 1):
                hit = any(
                    gt_category in neighbor_categories[r]
                    for r in range(k)
                )

                if hit:
                    topk_hits[gt_category][k] += 1

    return gt_total, topk_hits, uncategorized_gt


def write_neighbors_csv(
    path,
    class_names,
    topk_indices,
    topk_similarity,
):
    fieldnames = ["gt_class"]

    for rank in range(1, topk_indices.shape[1] + 1):
        fieldnames += [
            f"top{rank}_class",
            f"top{rank}_similarity",
        ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for gt_index, gt_name in enumerate(class_names):
            row = {"gt_class": gt_name}

            for rank in range(topk_indices.shape[1]):
                pred_index = int(topk_indices[gt_index, rank])
                row[f"top{rank + 1}_class"] = class_names[pred_index]
                row[f"top{rank + 1}_similarity"] = (
                    f"{float(topk_similarity[gt_index, rank]):.6f}"
                )

            writer.writerow(row)


def write_retention_csv(
    path,
    category_order,
    gt_total,
    topk_hits,
    max_k=5,
):
    fieldnames = ["category", "num_gt_classes"] + [
        f"top{k}_same_category_rate"
        for k in range(1, max_k + 1)
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for category in category_order:
            n = gt_total[category]
            if n == 0:
                continue

            row = {
                "category": category,
                "num_gt_classes": n,
            }

            for k in range(1, max_k + 1):
                row[f"top{k}_same_category_rate"] = (
                    topk_hits[category][k] / n
                )

            writer.writerow(row)


def plot_retention_heatmap(
    path,
    category_order,
    gt_total,
    topk_hits,
    max_k=5,
):
    used_categories = [
        category
        for category in category_order
        if gt_total[category] > 0
    ]

    matrix = np.array(
        [
            [
                100.0 * topk_hits[category][k]
                / gt_total[category]
                for k in range(1, max_k + 1)
            ]
            for category in used_categories
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, max(10, 0.28 * len(used_categories)))
    )

    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(max_k))
    ax.set_xticklabels(
        [f"Top-{k}" for k in range(1, max_k + 1)]
    )

    ax.set_yticks(range(len(used_categories)))
    ax.set_yticklabels(used_categories, fontsize=7)

    ax.set_xlabel("Search depth")
    ax.set_ylabel("GT category")
    ax.set_title(
        "CLIP same-category retrieval rate "
        "(self class excluded)"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rate (%)")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    category_order,
    gt_total,
    topk_hits,
):
    rows = []

    for category in category_order:
        n = gt_total[category]

        if n == 0:
            continue

        top1 = topk_hits[category][1] / n
        top5 = topk_hits[category][5] / n
        rows.append((category, n, top1, top5))

    print("")
    print("=== CLIP category structure ===")
    print("Self class is excluded from nearest-neighbor search.")
    print("")

    print("Top 10 categories by Top-1 same-category rate:")
    for category, n, top1, top5 in sorted(
        rows,
        key=lambda x: x[2],
        reverse=True,
    )[:10]:
        print(
            f"  {category:<28} "
            f"n={n:>4} "
            f"Top1={top1 * 100:6.2f}% "
            f"Top5={top5 * 100:6.2f}%"
        )

    print("")
    print("Bottom 10 categories by Top-1 same-category rate:")
    for category, n, top1, top5 in sorted(
        rows,
        key=lambda x: x[2],
    )[:10]:
        print(
            f"  {category:<28} "
            f"n={n:>4} "
            f"Top1={top1 * 100:6.2f}% "
            f"Top5={top5 * 100:6.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure how strongly THINGSplus categories cluster "
            "in the ground-truth CLIP image-feature space."
        )
    )

    parser.add_argument(
        "--features_path",
        default="../features/ViT-H-14_features_train.pt",
        help=(
            "Path to cached training CLIP features. "
            "Default assumes execution from Generation/."
        ),
    )
    parser.add_argument(
        "--img_dir_training",
        required=True,
        help="Path to THINGS training-image class folders.",
    )
    parser.add_argument(
        "--category_tsv",
        default="category53_long-format.tsv",
        help="Path to THINGSplus category53_long-format.tsv.",
    )
    parser.add_argument(
        "--output_dir",
        default="./clip_category_structure",
        help="Directory for output CSV/PNG files.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    class_names = load_class_names(args.img_dir_training)

    if len(class_names) != 1654:
        raise ValueError(
            "Expected 1654 training classes, "
            f"found {len(class_names)}"
        )

    concept_to_categories, category_order = load_category_map(
        args.category_tsv
    )

    class_features = load_clip_class_features(
        args.features_path,
        n_classes=len(class_names),
        images_per_class=10,
    )

    print("CLIP class features:", tuple(class_features.shape))

    topk_indices, topk_similarity = cosine_topk_excluding_self(
        class_features,
        k=5,
    )

    gt_total, topk_hits, uncategorized_gt = (
        analyze_category_retention(
            class_names,
            topk_indices,
            concept_to_categories,
            max_k=5,
        )
    )

    neighbors_csv = os.path.join(
        args.output_dir,
        "clip_top5_neighbors_excluding_self.csv",
    )
    retention_csv = os.path.join(
        args.output_dir,
        "clip_category_topk_retention.csv",
    )
    heatmap_png = os.path.join(
        args.output_dir,
        "clip_category_topk_retention_heatmap.png",
    )

    write_neighbors_csv(
        neighbors_csv,
        class_names,
        topk_indices,
        topk_similarity,
    )

    write_retention_csv(
        retention_csv,
        category_order,
        gt_total,
        topk_hits,
        max_k=5,
    )

    plot_retention_heatmap(
        heatmap_png,
        category_order,
        gt_total,
        topk_hits,
        max_k=5,
    )

    print_summary(
        category_order,
        gt_total,
        topk_hits,
    )

    print("")
    print(
        "Classes without THINGSplus category mapping:",
        len(uncategorized_gt),
    )
    print("")
    print("Saved:")
    print(f"  {neighbors_csv}")
    print(f"  {retention_csv}")
    print(f"  {heatmap_png}")


if __name__ == "__main__":
    main()
