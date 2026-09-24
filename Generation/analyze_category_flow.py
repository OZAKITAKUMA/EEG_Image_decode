#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


UNCATEGORIZED = "__uncategorized__"


def load_category_map(category_tsv):
    """
    Returns
    -------
    concept_to_categories : dict[str, set[str]]
        THINGS uniqueID -> set of THINGSplus categories
    category_order : list[str]
        Categories in first-appearance order in the TSV.
    """
    concept_to_categories = defaultdict(set)
    category_order = []
    seen_categories = set()

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

            if category not in seen_categories:
                seen_categories.add(category)
                category_order.append(category)

    return dict(concept_to_categories), category_order


def load_top5_rows(csv_path):
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"gt_class_name"}
        for rank in range(1, 6):
            required.add(f"top{rank}_class_name")

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Top-5 CSV is missing columns: {sorted(missing)}"
            )

        return list(reader)


def write_topk_retention_csv(
    path,
    category_order,
    gt_total,
    topk_hits,
):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "category",
            "num_gt_samples",
            "top1_same_category_rate",
            "top2_same_category_rate",
            "top3_same_category_rate",
            "top4_same_category_rate",
            "top5_same_category_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for category in category_order:
            n = gt_total[category]
            if n == 0:
                continue

            row = {
                "category": category,
                "num_gt_samples": n,
            }

            for k in range(1, 6):
                row[f"top{k}_same_category_rate"] = (
                    topk_hits[category][k] / n
                )

            writer.writerow(row)


def write_top1_flow_csv(
    path,
    category_order,
    gt_total,
    top1_flow,
):
    destination_order = category_order + [UNCATEGORIZED]

    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "gt_category",
            "num_gt_samples",
            *destination_order,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for gt_category in category_order:
            n = gt_total[gt_category]
            if n == 0:
                continue

            row = {
                "gt_category": gt_category,
                "num_gt_samples": n,
            }

            for pred_category in destination_order:
                row[pred_category] = (
                    top1_flow[gt_category][pred_category] / n
                )

            writer.writerow(row)


def write_rank_destination_csv(
    path,
    category_order,
    gt_total,
    rank_destinations,
):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "gt_category",
            "rank",
            "pred_category",
            "count",
            "rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        destination_order = category_order + [UNCATEGORIZED]

        for gt_category in category_order:
            n = gt_total[gt_category]
            if n == 0:
                continue

            for rank in range(1, 6):
                counter = rank_destinations[gt_category][rank]

                for pred_category in destination_order:
                    count = counter[pred_category]
                    if count == 0:
                        continue

                    writer.writerow(
                        {
                            "gt_category": gt_category,
                            "rank": rank,
                            "pred_category": pred_category,
                            "count": count,
                            "rate": count / n,
                        }
                    )


def plot_topk_retention(
    path,
    category_order,
    gt_total,
    topk_hits,
):
    used_categories = [
        c for c in category_order
        if gt_total[c] > 0
    ]

    matrix = np.array(
        [
            [
                100.0 * topk_hits[c][k] / gt_total[c]
                for k in range(1, 6)
            ]
            for c in used_categories
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, max(10, 0.28 * len(used_categories)))
    )
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(5))
    ax.set_xticklabels(
        ["Top-1", "Top-2", "Top-3", "Top-4", "Top-5"]
    )

    ax.set_yticks(range(len(used_categories)))
    ax.set_yticklabels(used_categories, fontsize=7)

    ax.set_xlabel("Search depth")
    ax.set_ylabel("GT category")
    ax.set_title(
        "Same-category retrieval rate by GT category"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rate (%)")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top1_flow(
    path,
    category_order,
    gt_total,
    top1_flow,
):
    used_categories = [
        c for c in category_order
        if gt_total[c] > 0
    ]
    destination_order = category_order + [UNCATEGORIZED]

    matrix = np.array(
        [
            [
                100.0
                * top1_flow[gt][pred]
                / gt_total[gt]
                for pred in destination_order
            ]
            for gt in used_categories
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(20, max(12, 0.3 * len(used_categories)))
    )
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(len(destination_order)))
    ax.set_xticklabels(
        destination_order,
        rotation=90,
        fontsize=6,
    )

    ax.set_yticks(range(len(used_categories)))
    ax.set_yticklabels(
        used_categories,
        fontsize=7,
    )

    ax.set_xlabel("Top-1 predicted category")
    ax.set_ylabel("GT category")
    ax.set_title(
        "GT category -> Top-1 predicted category"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Fraction of GT-category samples (%)"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_focus_category(
    path,
    focus_category,
    category_order,
    gt_total,
    rank_destinations,
):
    n = gt_total[focus_category]
    if n == 0:
        raise ValueError(
            f"No GT samples found for category: {focus_category}"
        )

    destination_order = category_order + [UNCATEGORIZED]

    # Keep only destination categories that occur at least once
    # in Top-1..Top-5 for the selected GT category.
    active_destinations = []

    for pred_category in destination_order:
        total_count = sum(
            rank_destinations[focus_category][rank][pred_category]
            for rank in range(1, 6)
        )
        if total_count > 0:
            active_destinations.append(pred_category)

    matrix = np.array(
        [
            [
                100.0
                * rank_destinations[focus_category][rank][pred]
                / n
                for pred in active_destinations
            ]
            for rank in range(1, 6)
        ],
        dtype=float,
    )

    fig_width = max(
        10,
        0.45 * len(active_destinations),
    )

    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(len(active_destinations)))
    ax.set_xticklabels(
        active_destinations,
        rotation=90,
        fontsize=7,
    )

    ax.set_yticks(range(5))
    ax.set_yticklabels(
        ["Top-1", "Top-2", "Top-3", "Top-4", "Top-5"]
    )

    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Rank")
    ax.set_title(
        f"Destination categories for GT = {focus_category} "
        f"(n={n})"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Fraction of GT-category samples (%)"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_focus_summary(
    focus_category,
    gt_total,
    topk_hits,
    rank_destinations,
):
    n = gt_total[focus_category]

    print("")
    print("=" * 70)
    print(
        f"Focus category: {focus_category} "
        f"(GT samples={n})"
    )
    print("=" * 70)

    for k in range(1, 6):
        rate = (
            topk_hits[focus_category][k] / n
            if n > 0
            else 0.0
        )
        print(
            f"Top-{k} contains same category: "
            f"{rate * 100:.2f}%"
        )

    print("")
    print("Top-1 destination categories:")

    for category, count in (
        rank_destinations[focus_category][1]
        .most_common(10)
    ):
        print(
            f"  {category:<30} "
            f"{count:>4} / {n} "
            f"({count / n * 100:6.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze THINGSplus category flow in "
            "validation_top5_neighbors.csv"
        )
    )

    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to validation_top5_neighbors.csv",
    )
    parser.add_argument(
        "--category_tsv",
        default="category53_long-format.tsv",
        help="Path to category53_long-format.tsv",
    )
    parser.add_argument(
        "--output_dir",
        default="./category_flow_analysis",
        help="Directory for CSV and PNG outputs",
    )
    parser.add_argument(
        "--focus_category",
        default="animal",
        help=(
            "GT category for the rank-wise destination "
            "heatmap. Default: animal"
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    concept_to_categories, category_order = (
        load_category_map(args.category_tsv)
    )
    rows = load_top5_rows(args.csv_path)

    gt_total = Counter()

    # topk_hits[c][k]:
    # GT has category c, and at least one prediction
    # within Top-k also has category c.
    topk_hits = defaultdict(Counter)

    # top1_flow[gt_cat][pred_cat]:
    # Among samples whose GT contains gt_cat,
    # how often does the Top-1 prediction contain pred_cat?
    top1_flow = defaultdict(Counter)

    # rank_destinations[gt_cat][rank][pred_cat]
    rank_destinations = defaultdict(
        lambda: defaultdict(Counter)
    )

    gt_without_category = 0

    for row in rows:
        gt_name = row["gt_class_name"].strip()
        gt_categories = concept_to_categories.get(
            gt_name,
            set(),
        )

        if not gt_categories:
            gt_without_category += 1
            continue

        pred_categories_by_rank = {}

        for rank in range(1, 6):
            pred_name = (
                row[f"top{rank}_class_name"].strip()
            )
            pred_categories = concept_to_categories.get(
                pred_name,
                set(),
            )

            if pred_categories:
                pred_categories_by_rank[rank] = (
                    pred_categories
                )
            else:
                pred_categories_by_rank[rank] = {
                    UNCATEGORIZED
                }

        for gt_category in gt_categories:
            gt_total[gt_category] += 1

            # Top-k same-category retention.
            for k in range(1, 6):
                hit = any(
                    gt_category
                    in pred_categories_by_rank[rank]
                    for rank in range(1, k + 1)
                )

                if hit:
                    topk_hits[gt_category][k] += 1

            # Top-1 category destinations.
            for pred_category in (
                pred_categories_by_rank[1]
            ):
                top1_flow[gt_category][
                    pred_category
                ] += 1

            # Destination category for each rank.
            for rank in range(1, 6):
                for pred_category in (
                    pred_categories_by_rank[rank]
                ):
                    rank_destinations[
                        gt_category
                    ][rank][pred_category] += 1

    retention_csv = os.path.join(
        args.output_dir,
        "category_topk_retention.csv",
    )
    top1_flow_csv = os.path.join(
        args.output_dir,
        "category_top1_flow.csv",
    )
    rank_destination_csv = os.path.join(
        args.output_dir,
        "category_rank_destinations.csv",
    )

    write_topk_retention_csv(
        retention_csv,
        category_order,
        gt_total,
        topk_hits,
    )
    write_top1_flow_csv(
        top1_flow_csv,
        category_order,
        gt_total,
        top1_flow,
    )
    write_rank_destination_csv(
        rank_destination_csv,
        category_order,
        gt_total,
        rank_destinations,
    )

    retention_png = os.path.join(
        args.output_dir,
        "category_topk_retention_heatmap.png",
    )
    top1_flow_png = os.path.join(
        args.output_dir,
        "category_top1_flow_heatmap.png",
    )

    plot_topk_retention(
        retention_png,
        category_order,
        gt_total,
        topk_hits,
    )
    plot_top1_flow(
        top1_flow_png,
        category_order,
        gt_total,
        top1_flow,
    )

    if args.focus_category not in gt_total:
        available = ", ".join(
            c for c in category_order
            if gt_total[c] > 0
        )
        raise ValueError(
            f"Unknown or empty focus category: "
            f"{args.focus_category}\n"
            f"Available categories: {available}"
        )

    focus_png = os.path.join(
        args.output_dir,
        (
            f"category_{args.focus_category}"
            "_rank_destinations.png"
        ),
    )

    plot_focus_category(
        focus_png,
        args.focus_category,
        category_order,
        gt_total,
        rank_destinations,
    )

    print_focus_summary(
        args.focus_category,
        gt_total,
        topk_hits,
        rank_destinations,
    )

    print("")
    print(
        f"Rows in Top-5 CSV: {len(rows)}"
    )
    print(
        "Rows with GT class not present in "
        f"THINGSplus mapping: {gt_without_category}"
    )
    print("")
    print("Saved:")
    print(f"  {retention_csv}")
    print(f"  {top1_flow_csv}")
    print(f"  {rank_destination_csv}")
    print(f"  {retention_png}")
    print(f"  {top1_flow_png}")
    print(f"  {focus_png}")


if __name__ == "__main__":
    main()
