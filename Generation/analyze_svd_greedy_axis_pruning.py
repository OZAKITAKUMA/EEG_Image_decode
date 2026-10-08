#!/usr/bin/env python3
"""
Greedy SVD-axis pruning analysis for EEG -> CLIP retrieval.

Goal
----
Find SVD/PCA directions that are unnecessary or harmful for validation
retrieval, without retraining the EEG encoder for every candidate axis.

At each greedy step:
  1. Evaluate removing each still-active SVD axis.
  2. Prefer the removal that gives the largest validation-accuracy gain.
  3. If gains tie, prefer the axis with larger image-feature variance.
  4. Stop when every candidate would reduce the selected metric by more than
     --max_allowed_drop, or when --max_remove axes have been removed.

Important
---------
- The SVD basis is fitted only on the 9/10 training image conditions.
- Axis selection uses validation data only.
- Test data is never used for selecting axes.
- Removing one centered PCA axis means replacing that coordinate by the
  projected global mean, which is exactly equivalent to:
      x_removed = mean + sum(kept_pc_coeff * pc_basis)
- Retrieval candidates are fixed once with --seed, so every candidate axis is
  compared under the same 200-way retrieval problem.

This script only searches for axes. It does not retrain the EEG encoder.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eegdatasets import EEGDataset, get_image_encoder_feature_dim
from encoder_utils import stratified_condition_split
from models.atms import ATMS, extract_id_from_string


N_CLASSES = 1654
CONDITIONS_PER_CLASS = 10


def find_latest_full_checkpoint(subject):
    pattern = (
        REPO_ROOT
        / "Generation"
        / "models"
        / "baseline"
        / "clip"
        / "full"
        / "mse_contrastive"
        / "val_rdm_mse"
        / "encoder"
        / subject
        / "*"
        / "best.pth"
    )

    matches = sorted(glob.glob(str(pattern)))

    if not matches:
        raise FileNotFoundError(
            "Full-CLIP val_rdm_mse checkpoint was not found automatically. "
            "Pass --encoder_path explicitly. Searched: "
            f"{pattern}"
        )

    return matches[-1]


def fit_global_svd(
    img_features_all,
    seed,
    val_ratio,
    device,
):
    fit_indices, val_condition_indices = stratified_condition_split(
        n_classes=N_CLASSES,
        conditions_per_class=CONDITIONS_PER_CLASS,
        trials_per_condition=1,
        val_ratio=val_ratio,
        seed=seed,
    )

    index_tensor = torch.as_tensor(
        fit_indices,
        dtype=torch.long,
    )

    x_fit = img_features_all[index_tensor].float()
    mean = x_fit.mean(
        dim=0,
        keepdim=True,
    )

    centered = (
        x_fit
        - mean
    ).to(device)

    print(
        "Fitting global SVD on training 9/10 conditions:",
        f"shape={tuple(centered.shape)}",
        f"device={device}",
    )

    _, singular_values, vh = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    return (
        mean.cpu(),
        singular_values.cpu(),
        vh.cpu(),
        fit_indices,
        val_condition_indices,
    )


@torch.no_grad()
def extract_validation_features(
    model,
    val_loader,
    subject,
    device,
):
    model.eval()

    eeg_features = []
    labels_all = []

    subject_id = extract_id_from_string(subject)

    for (
        eeg_data,
        labels,
        _text,
        _text_features,
        _img,
        _img_features,
    ) in tqdm(
        val_loader,
        desc="Extracting validation EEG features",
    ):
        eeg_data = eeg_data.to(device)

        batch_size = eeg_data.shape[0]

        subject_ids = torch.full(
            (batch_size,),
            subject_id,
            dtype=torch.long,
            device=device,
        )

        predicted = model(
            eeg_data,
            subject_ids,
        ).float()

        eeg_features.append(
            predicted.cpu()
        )
        labels_all.append(
            labels.cpu().long()
        )

    return (
        torch.cat(eeg_features, dim=0),
        torch.cat(labels_all, dim=0),
    )


def build_fixed_candidate_indices(
    labels,
    n_classes,
    k,
    seed,
):
    if k < 2:
        raise ValueError("k must be >= 2")

    if k > n_classes:
        raise ValueError(
            f"k={k} exceeds n_classes={n_classes}"
        )

    rng = random.Random(seed)

    all_classes = set(
        range(n_classes)
    )

    rows = []

    for label in labels.tolist():
        possible = list(
            all_classes
            - {int(label)}
        )

        selected = rng.sample(
            possible,
            k - 1,
        )

        # Keep the correct class at a fixed final position.
        selected.append(
            int(label)
        )

        rows.append(selected)

    return torch.as_tensor(
        rows,
        dtype=torch.long,
    )


def retrieval_accuracy_from_similarity(
    similarity,
    target_position,
):
    top1 = (
        similarity.argmax(dim=1)
        == target_position
    ).float().mean().item()

    top5_indices = similarity.topk(
        k=min(
            5,
            similarity.shape[1],
        ),
        dim=1,
    ).indices

    top5 = (
        top5_indices
        == target_position
    ).any(dim=1).float().mean().item()

    return top1, top5


def compute_current_similarity(
    current_dot,
    current_eeg_norm2,
    current_pool_norm2,
    candidate_indices,
    eps=1e-12,
):
    pool_norm = current_pool_norm2[
        candidate_indices
    ]

    denominator = torch.sqrt(
        current_eeg_norm2.unsqueeze(1)
        * pool_norm
    ).clamp_min(eps)

    return (
        current_dot
        / denominator
    )


def evaluate_axis_removal(
    axis,
    current_dot,
    current_eeg_norm2,
    current_pool_norm2,
    eeg_rotated,
    pool_rotated,
    candidate_indices,
    mean_rotated,
    target_position,
):
    mean_value = mean_rotated[axis]

    eeg_axis = eeg_rotated[
        :,
        axis,
    ]

    pool_axis_all = pool_rotated[
        :,
        axis,
    ]

    pool_axis_candidates = pool_axis_all[
        candidate_indices
    ]

    candidate_dot = (
        current_dot
        - eeg_axis.unsqueeze(1)
        * pool_axis_candidates
        + mean_value.pow(2)
    )

    candidate_eeg_norm2 = (
        current_eeg_norm2
        - eeg_axis.pow(2)
        + mean_value.pow(2)
    )

    candidate_pool_norm2 = (
        current_pool_norm2
        - pool_axis_all.pow(2)
        + mean_value.pow(2)
    )

    similarity = compute_current_similarity(
        current_dot=candidate_dot,
        current_eeg_norm2=candidate_eeg_norm2,
        current_pool_norm2=candidate_pool_norm2,
        candidate_indices=candidate_indices,
    )

    top1, top5 = (
        retrieval_accuracy_from_similarity(
            similarity,
            target_position,
        )
    )

    return (
        top1,
        top5,
        candidate_dot,
        candidate_eeg_norm2,
        candidate_pool_norm2,
    )


def write_candidate_scores(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step",
                "pc_one_based",
                "axis_zero_based",
                "singular_value",
                "variance",
                "variance_ratio",
                "candidate_top1",
                "candidate_top5",
                "gain_top1",
                "gain_top5",
                "selected_metric_gain",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step",
                "removed_pc_one_based",
                "removed_axis_zero_based",
                "singular_value",
                "variance_ratio",
                "previous_top1",
                "new_top1",
                "gain_top1",
                "previous_top5",
                "new_top5",
                "gain_top5",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Greedily remove SVD axes that do not hurt validation retrieval."
        )
    )

    parser.add_argument(
        "--data_path",
        default="/home/moepy/ozakitakuma/data_eeg",
    )
    parser.add_argument(
        "--img_dir_training",
        default="/home/moepy/ozakitakuma/data_image/training_images",
    )
    parser.add_argument(
        "--img_dir_test",
        default="/home/moepy/ozakitakuma/data_image/test_images",
    )
    parser.add_argument(
        "--features_dir",
        default=None,
    )
    parser.add_argument(
        "--encoder_path",
        default=None,
        help=(
            "Full-CLIP encoder checkpoint. "
            "If omitted, the latest baseline/clip/full/"
            "mse_contrastive/val_rdm_mse checkpoint is used."
        ),
    )
    parser.add_argument(
        "--subject",
        default="sub-01",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--avg_trials",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_avg_trials",
        dest="avg_trials",
        action="store_false",
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
    parser.add_argument(
        "--k_way",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--selection_metric",
        choices=[
            "top1",
            "top5",
        ],
        default="top1",
    )
    parser.add_argument(
        "--max_remove",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max_allowed_drop",
        type=float,
        default=0.0,
        help=(
            "Allow one greedy removal to reduce the selected validation "
            "accuracy by at most this absolute amount. "
            "0.0 means never accept an accuracy-decreasing step."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="cuda:0",
    )
    parser.add_argument(
        "--svd_device",
        default="cpu",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )

    args = parser.parse_args()

    if not args.avg_trials:
        raise ValueError(
            "This first implementation is intentionally restricted to "
            "--avg_trials, matching the current baseline experiments."
        )

    if args.max_remove < 1:
        raise ValueError(
            "--max_remove must be >= 1"
        )

    if args.max_allowed_drop < 0:
        raise ValueError(
            "--max_allowed_drop must be >= 0"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.gpu
        if torch.cuda.is_available()
        else "cpu"
    )

    svd_device = torch.device(
        args.svd_device
    )

    encoder_path = (
        args.encoder_path
        if args.encoder_path is not None
        else find_latest_full_checkpoint(
            args.subject
        )
    )

    if not os.path.exists(
        encoder_path
    ):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {encoder_path}"
        )

    checkpoint_stamp = (
        Path(encoder_path)
        .parent
        .name
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (
            REPO_ROOT
            / "Generation"
            / "outputs"
            / "svd_greedy_axis_pruning"
            / args.subject
            / checkpoint_stamp
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SVD Greedy Axis Pruning")
    print("=" * 72)
    print(
        "Subject:",
        args.subject,
    )
    print(
        "Encoder:",
        encoder_path,
    )
    print(
        "Selection metric:",
        args.selection_metric,
    )
    print(
        "k-way:",
        args.k_way,
    )
    print(
        "Max removals:",
        args.max_remove,
    )
    print(
        "Allowed per-step drop:",
        args.max_allowed_drop,
    )
    print(
        "Output:",
        output_dir,
    )
    print("=" * 72)

    dataset = EEGDataset(
        args.data_path,
        img_dir_training=args.img_dir_training,
        img_dir_test=args.img_dir_test,
        feature_type="clip",
        features_dir=args.features_dir,
        subjects=[
            args.subject
        ],
        train=True,
        avg_trials=True,
        feature_space="clip",
    )

    img_features_all = (
        dataset.img_features
        .detach()
        .cpu()
        .float()
    )

    expected_image_features = (
        N_CLASSES
        * CONDITIONS_PER_CLASS
    )

    if img_features_all.shape[0] != expected_image_features:
        raise RuntimeError(
            "Expected one image feature per training condition: "
            f"{expected_image_features}, "
            f"got {img_features_all.shape[0]}"
        )

    feature_dim = get_image_encoder_feature_dim(
        "clip"
    )

    if img_features_all.shape[1] != feature_dim:
        raise RuntimeError(
            "Unexpected CLIP feature dimension: "
            f"expected={feature_dim}, "
            f"actual={img_features_all.shape[1]}"
        )

    train_indices, val_indices = (
        stratified_condition_split(
            n_classes=N_CLASSES,
            conditions_per_class=CONDITIONS_PER_CLASS,
            trials_per_condition=1,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    )

    val_subset = Subset(
        dataset,
        val_indices,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ATMS(
        outputs_dim=feature_dim
    )

    model.load_state_dict(
        torch.load(
            encoder_path,
            map_location=device,
            weights_only=False,
        )
    )

    model = model.to(device)
    model.eval()

    eeg_features_val, labels_val = (
        extract_validation_features(
            model=model,
            val_loader=val_loader,
            subject=args.subject,
            device=device,
        )
    )

    del model

    if len(eeg_features_val) != len(val_indices):
        raise RuntimeError(
            "Validation EEG feature count mismatch: "
            f"expected={len(val_indices)}, "
            f"actual={len(eeg_features_val)}"
        )

    (
        global_mean,
        singular_values,
        vh,
        _fit_indices,
        _svd_val_indices,
    ) = fit_global_svd(
        img_features_all=img_features_all,
        seed=args.seed,
        val_ratio=args.val_ratio,
        device=svd_device,
    )

    image_pool = (
        img_features_all[
            ::CONDITIONS_PER_CLASS
        ]
        .clone()
    )

    if image_pool.shape[0] != N_CLASSES:
        raise RuntimeError(
            "Unexpected image-pool size: "
            f"{image_pool.shape[0]}"
        )

    candidate_indices = (
        build_fixed_candidate_indices(
            labels=labels_val,
            n_classes=N_CLASSES,
            k=args.k_way,
            seed=args.seed,
        )
    )

    search_device = device

    vh = vh.to(
        search_device
    )

    singular_values = (
        singular_values
        .to(search_device)
    )

    global_mean = (
        global_mean
        .to(search_device)
    )

    eeg_features_val = (
        eeg_features_val
        .to(search_device)
    )

    image_pool = (
        image_pool
        .to(search_device)
    )

    candidate_indices = (
        candidate_indices
        .to(search_device)
    )

    # Orthogonal rotation preserves cosine similarity.
    eeg_rotated = (
        eeg_features_val
        @ vh.T
    )

    pool_rotated = (
        image_pool
        @ vh.T
    )

    mean_rotated = (
        global_mean
        @ vh.T
    ).squeeze(0)

    current_eeg_norm2 = (
        eeg_rotated.pow(2)
        .sum(dim=1)
    )

    current_pool_norm2 = (
        pool_rotated.pow(2)
        .sum(dim=1)
    )

    full_dot = (
        eeg_rotated
        @ pool_rotated.T
    )

    current_dot = full_dot[
        torch.arange(
            len(eeg_rotated),
            device=search_device,
        ).unsqueeze(1),
        candidate_indices,
    ]

    target_position = (
        args.k_way
        - 1
    )

    current_similarity = (
        compute_current_similarity(
            current_dot=current_dot,
            current_eeg_norm2=current_eeg_norm2,
            current_pool_norm2=current_pool_norm2,
            candidate_indices=candidate_indices,
        )
    )

    current_top1, current_top5 = (
        retrieval_accuracy_from_similarity(
            current_similarity,
            target_position,
        )
    )

    baseline_top1 = current_top1
    baseline_top5 = current_top5

    print("")
    print(
        "Baseline validation retrieval:",
        f"Top-1={baseline_top1:.6f}",
        f"Top-5={baseline_top5:.6f}",
    )

    variance = (
        singular_values.pow(2)
    )

    variance_ratio = (
        variance
        / variance.sum()
    )

    remaining_axes = set(
        range(feature_dim)
    )

    selected_rows = []
    removed_axes = []

    for step in range(
        1,
        min(
            args.max_remove,
            feature_dim - 1,
        )
        + 1,
    ):
        print("")
        print(
            f"===== Greedy step {step} ====="
        )

        candidate_rows = []

        best = None

        current_metric = (
            current_top1
            if args.selection_metric == "top1"
            else current_top5
        )

        axes_sorted = sorted(
            remaining_axes
        )

        for index, axis in enumerate(
            tqdm(
                axes_sorted,
                desc=(
                    f"Testing {len(axes_sorted)} axes"
                ),
            )
        ):
            (
                top1,
                top5,
                _candidate_dot,
                _candidate_eeg_norm2,
                _candidate_pool_norm2,
            ) = evaluate_axis_removal(
                axis=axis,
                current_dot=current_dot,
                current_eeg_norm2=current_eeg_norm2,
                current_pool_norm2=current_pool_norm2,
                eeg_rotated=eeg_rotated,
                pool_rotated=pool_rotated,
                candidate_indices=candidate_indices,
                mean_rotated=mean_rotated,
                target_position=target_position,
            )

            gain_top1 = (
                top1
                - current_top1
            )

            gain_top5 = (
                top5
                - current_top5
            )

            selected_gain = (
                gain_top1
                if args.selection_metric == "top1"
                else gain_top5
            )

            row = {
                "step": step,
                "pc_one_based": axis + 1,
                "axis_zero_based": axis,
                "singular_value": (
                    singular_values[axis]
                    .item()
                ),
                "variance": (
                    variance[axis]
                    .item()
                ),
                "variance_ratio": (
                    variance_ratio[axis]
                    .item()
                ),
                "candidate_top1": top1,
                "candidate_top5": top5,
                "gain_top1": gain_top1,
                "gain_top5": gain_top5,
                "selected_metric_gain": (
                    selected_gain
                ),
            }

            candidate_rows.append(row)

            # Primary key: validation-accuracy gain.
            # Tie-break: remove the higher-variance axis.
            key = (
                selected_gain,
                variance[axis].item(),
            )

            if (
                best is None
                or key > best["key"]
            ):
                best = {
                    "axis": axis,
                    "key": key,
                    "row": row,
                }

        candidate_path = (
            output_dir
            / f"candidate_scores_step_{step:03d}.csv"
        )

        write_candidate_scores(
            candidate_path,
            candidate_rows,
        )

        if best is None:
            print(
                "No candidate axis was evaluated."
            )
            break

        best_gain = (
            best["row"][
                "selected_metric_gain"
            ]
        )

        if (
            best_gain
            < -args.max_allowed_drop
        ):
            print(
                "Stopping: every remaining axis would "
                "reduce the selected validation metric "
                "more than allowed."
            )
            print(
                f"Best available gain={best_gain:+.6f}, "
                f"allowed_drop={args.max_allowed_drop:.6f}"
            )
            break

        selected_axis = best["axis"]

        previous_top1 = current_top1
        previous_top5 = current_top5

        (
            new_top1,
            new_top5,
            current_dot,
            current_eeg_norm2,
            current_pool_norm2,
        ) = evaluate_axis_removal(
            axis=selected_axis,
            current_dot=current_dot,
            current_eeg_norm2=current_eeg_norm2,
            current_pool_norm2=current_pool_norm2,
            eeg_rotated=eeg_rotated,
            pool_rotated=pool_rotated,
            candidate_indices=candidate_indices,
            mean_rotated=mean_rotated,
            target_position=target_position,
        )

        current_top1 = new_top1
        current_top5 = new_top5

        remaining_axes.remove(
            selected_axis
        )

        removed_axes.append(
            selected_axis
        )

        summary_row = {
            "step": step,
            "removed_pc_one_based": (
                selected_axis
                + 1
            ),
            "removed_axis_zero_based": (
                selected_axis
            ),
            "singular_value": (
                singular_values[
                    selected_axis
                ].item()
            ),
            "variance_ratio": (
                variance_ratio[
                    selected_axis
                ].item()
            ),
            "previous_top1": previous_top1,
            "new_top1": current_top1,
            "gain_top1": (
                current_top1
                - previous_top1
            ),
            "previous_top5": previous_top5,
            "new_top5": current_top5,
            "gain_top5": (
                current_top5
                - previous_top5
            ),
        }

        selected_rows.append(
            summary_row
        )

        print(
            "Selected:",
            f"PC{selected_axis + 1}",
            f"variance_ratio={summary_row['variance_ratio']:.8f}",
            f"Top-1 {previous_top1:.6f} -> {current_top1:.6f}",
            f"({summary_row['gain_top1']:+.6f})",
            f"Top-5 {previous_top5:.6f} -> {current_top5:.6f}",
            f"({summary_row['gain_top5']:+.6f})",
        )

        write_summary(
            output_dir
            / "pruning_summary.csv",
            selected_rows,
        )

    keep_mask = torch.ones(
        feature_dim,
        dtype=torch.bool,
    )

    if removed_axes:
        keep_mask[
            torch.as_tensor(
                removed_axes,
                dtype=torch.long,
            )
        ] = False

    torch.save(
        {
            "subject": args.subject,
            "encoder_path": str(
                encoder_path
            ),
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "k_way": args.k_way,
            "selection_metric": (
                args.selection_metric
            ),
            "max_allowed_drop": (
                args.max_allowed_drop
            ),
            "baseline_top1": baseline_top1,
            "baseline_top5": baseline_top5,
            "final_top1": current_top1,
            "final_top5": current_top5,
            "removed_axes_zero_based": (
                torch.as_tensor(
                    removed_axes,
                    dtype=torch.long,
                )
            ),
            "removed_pcs_one_based": [
                axis + 1
                for axis in removed_axes
            ],
            "keep_mask": keep_mask,
            "global_mean": (
                global_mean.cpu()
            ),
            "singular_values": (
                singular_values.cpu()
            ),
            "vh": vh.cpu(),
        },
        output_dir
        / "selected_svd_axes.pt",
    )

    metadata = {
        "subject": args.subject,
        "encoder_path": str(
            encoder_path
        ),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "k_way": args.k_way,
        "selection_metric": (
            args.selection_metric
        ),
        "max_allowed_drop": (
            args.max_allowed_drop
        ),
        "baseline_top1": baseline_top1,
        "baseline_top5": baseline_top5,
        "final_top1": current_top1,
        "final_top5": current_top5,
        "removed_pcs_one_based": [
            axis + 1
            for axis in removed_axes
        ],
        "n_removed": len(
            removed_axes
        ),
        "n_kept": (
            feature_dim
            - len(removed_axes)
        ),
    }

    with (
        output_dir
        / "run_summary.json"
    ).open(
        "w",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
        )

    print("")
    print("=" * 72)
    print("Finished")
    print("=" * 72)
    print(
        "Removed PCs:",
        metadata[
            "removed_pcs_one_based"
        ],
    )
    print(
        "Dimensions:",
        f"{feature_dim} -> {metadata['n_kept']}",
    )
    print(
        "Validation Top-1:",
        f"{baseline_top1:.6f} -> {current_top1:.6f}",
    )
    print(
        "Validation Top-5:",
        f"{baseline_top5:.6f} -> {current_top5:.6f}",
    )
    print(
        "Saved:",
        output_dir,
    )


if __name__ == "__main__":
    main()
