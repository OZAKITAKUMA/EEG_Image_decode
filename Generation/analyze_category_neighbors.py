import argparse
import csv
from collections import defaultdict


parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help="Path to validation_top5_neighbors.csv",
)

parser.add_argument(
    "--category_tsv",
    type=str,
    default="category53_long-format.tsv",
    help="Path to THINGSplus category53_long-format.tsv",
)

args = parser.parse_args()


# ------------------------------------------------------------
# Load THINGSplus categories
# ------------------------------------------------------------
concept_to_categories = defaultdict(set)

with open(args.category_tsv, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        concept_to_categories[row["uniqueID"]].add(
            row["category"]
        )


def same_category(name_a, name_b):
    categories_a = concept_to_categories.get(name_a, set())
    categories_b = concept_to_categories.get(name_b, set())

    return bool(categories_a & categories_b)


# ------------------------------------------------------------
# Load Top-5 retrieval results
# ------------------------------------------------------------
with open(args.csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


eligible = 0
top1_match = 0
top5_any_match = 0
top5_match_total = 0


for row in rows:
    gt = row["gt_class_name"]

    # GTにTHINGSplusカテゴリがない場合は評価対象外
    if gt not in concept_to_categories:
        continue

    eligible += 1

    top_names = [
        row[f"top{i}_class_name"]
        for i in range(1, 6)
    ]

    matches = [
        same_category(gt, pred)
        for pred in top_names
    ]

    if matches[0]:
        top1_match += 1

    if any(matches):
        top5_any_match += 1

    top5_match_total += sum(matches)


print("Eligible GT classes       :", eligible)
print(
    "Top1 same-category rate   :",
    f"{top1_match / eligible:.4f}",
)
print(
    "Top5 any-category rate    :",
    f"{top5_any_match / eligible:.4f}",
)
print(
    "Mean same-category in Top5:",
    f"{top5_match_total / eligible:.4f}",
    "/ 5",
)


# 実行例:
# python analyze_category_neighbors.py \
#   --csv_path outputs/benchmark_encoder_only_rsa_mse/clip/rsa_0.0/sub-01/09-22_09-49/validation_top5_neighbors.csv