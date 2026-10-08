#!/usr/bin/env python3
"""
Create a CLIP teacher cache with Greedy-selected SVD axes removed, then
retrain/evaluate the encoder with the existing benchmark pipeline.

The selected axis mask must come from analyze_svd_greedy_axis_pruning.py.
The SVD basis and global mean saved in selected_svd_axes.pt are reused exactly;
no new SVD is fitted here.

Example
-------
python Generation/run_greedy_pruned_retraining.py

By default the newest
Generation/outputs/svd_greedy_axis_pruning/sub-01/*/selected_svd_axes.pt
is used.  Pass --selection_path to choose another result explicitly.
"""

import argparse
import os
import subprocess
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATION_DIR = REPO_ROOT / "Generation"

DEFAULT_SOURCE_FEATURES = (
    REPO_ROOT
    / "features"
    / "ViT-H-14_features_train.pt"
)


def find_latest_selection(subject):
    root = (
        GENERATION_DIR
        / "outputs"
        / "svd_greedy_axis_pruning"
        / subject
    )

    matches = list(
        root.glob(
            "*/selected_svd_axes.pt"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Greedy selection file was not found. "
            "Run analyze_svd_greedy_axis_pruning.py first, "
            "or pass --selection_path explicitly. "
            f"Searched: {root}"
        )

    return max(
        matches,
        key=lambda path: path.stat().st_mtime,
    )


def load_feature_cache(path):
    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(saved, dict):
        raise TypeError(
            "Feature cache must be a dict: "
            f"{path}"
        )

    if "img_features" not in saved:
        raise KeyError(
            "'img_features' was not found in "
            f"{path}"
        )

    img_features = saved[
        "img_features"
    ]

    if (
        not torch.is_tensor(img_features)
        or img_features.ndim != 2
    ):
        raise ValueError(
            "img_features must be a 2-D tensor, "
            f"got {type(img_features).__name__} "
            f"{getattr(img_features, 'shape', None)}"
        )

    return (
        saved,
        img_features.detach().cpu(),
    )


def load_selection(path):
    selected = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = [
        "subject",
        "keep_mask",
        "global_mean",
        "singular_values",
        "vh",
        "removed_axes_zero_based",
    ]

    missing = [
        key
        for key in required
        if key not in selected
    ]

    if missing:
        raise KeyError(
            "Selection file is missing keys: "
            + ", ".join(missing)
        )

    return selected


def reconstruct_with_selected_mask(
    img_features,
    selected,
    chunk_size,
):
    original_dtype = img_features.dtype
    x = img_features.float()

    mean = (
        selected["global_mean"]
        .detach()
        .cpu()
        .float()
    )

    vh = (
        selected["vh"]
        .detach()
        .cpu()
        .float()
    )

    keep_mask = (
        selected["keep_mask"]
        .detach()
        .cpu()
        .bool()
    )

    feature_dim = x.shape[1]

    if tuple(mean.shape) != (
        1,
        feature_dim,
    ):
        raise ValueError(
            "global_mean shape mismatch: "
            f"expected={(1, feature_dim)}, "
            f"actual={tuple(mean.shape)}"
        )

    if tuple(vh.shape) != (
        feature_dim,
        feature_dim,
    ):
        raise ValueError(
            "vh shape mismatch: "
            f"expected={(feature_dim, feature_dim)}, "
            f"actual={tuple(vh.shape)}"
        )

    if tuple(keep_mask.shape) != (
        feature_dim,
    ):
        raise ValueError(
            "keep_mask shape mismatch: "
            f"expected={(feature_dim,)}, "
            f"actual={tuple(keep_mask.shape)}"
        )

    components = vh[
        keep_mask
    ]

    reconstructed = torch.empty_like(
        x
    )

    for start in range(
        0,
        x.shape[0],
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            x.shape[0],
        )

        centered = (
            x[start:end]
            - mean
        )

        coefficients = (
            centered
            @ components.T
        )

        reconstructed[
            start:end
        ] = (
            coefficients
            @ components
            + mean
        )

    return reconstructed.to(
        dtype=original_dtype
    )


def save_pruned_cache(
    source_cache,
    reconstructed,
    selected,
    selection_path,
    source_features_path,
    output_path,
):
    output = dict(
        source_cache
    )

    output[
        "img_features"
    ] = reconstructed.cpu()

    removed_axes = (
        selected[
            "removed_axes_zero_based"
        ]
        .detach()
        .cpu()
        .long()
    )

    singular_values = (
        selected[
            "singular_values"
        ]
        .detach()
        .cpu()
        .float()
    )

    keep_mask = (
        selected[
            "keep_mask"
        ]
        .detach()
        .cpu()
        .bool()
    )

    variance = (
        singular_values
        .pow(2)
    )

    kept_variance_ratio = (
        variance[
            keep_mask
        ].sum()
        / variance.sum()
    ).item()

    previous_metadata = (
        output.get(
            "feature_metadata"
        )
    )

    metadata = {
        "transform": (
            "greedy_validation_selected_"
            "svd_axis_removal"
        ),
        "subject_used_for_selection": (
            selected["subject"]
        ),
        "selection_path": str(
            Path(
                selection_path
            ).resolve()
        ),
        "source_features_path": str(
            Path(
                source_features_path
            ).resolve()
        ),
        "removed_axes_zero_based": (
            removed_axes.tolist()
        ),
        "removed_pcs_one_based": [
            int(axis) + 1
            for axis in removed_axes.tolist()
        ],
        "removed_component_count": int(
            removed_axes.numel()
        ),
        "kept_component_count": int(
            keep_mask.sum().item()
        ),
        "kept_variance_ratio": (
            kept_variance_ratio
        ),
        "svd_fit_scope": (
            "train_9_of_10_conditions_per_class"
        ),
        "selection_scope": (
            "held_out_validation_conditions"
        ),
        "seed": selected.get(
            "seed"
        ),
        "val_ratio": selected.get(
            "val_ratio"
        ),
        "selection_metric": (
            selected.get(
                "selection_metric"
            )
        ),
        "selection_baseline_top1": (
            selected.get(
                "baseline_top1"
            )
        ),
        "selection_final_top1": (
            selected.get(
                "final_top1"
            )
        ),
        "selection_baseline_top5": (
            selected.get(
                "baseline_top5"
            )
        ),
        "selection_final_top5": (
            selected.get(
                "final_top5"
            )
        ),
    }

    if (
        previous_metadata
        is not None
    ):
        metadata[
            "source_feature_metadata"
        ] = previous_metadata

    output[
        "feature_metadata"
    ] = metadata

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output,
        output_path,
    )

    return kept_variance_ratio


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Greedy-pruned CLIP teacher cache "
            "and retrain the encoder."
        )
    )

    parser.add_argument(
        "--subject",
        default="sub-01",
    )
    parser.add_argument(
        "--selection_path",
        default=None,
        help=(
            "selected_svd_axes.pt produced by "
            "analyze_svd_greedy_axis_pruning.py. "
            "If omitted, the newest file for --subject is used."
        ),
    )
    parser.add_argument(
        "--source_features_path",
        default=str(
            DEFAULT_SOURCE_FEATURES
        ),
    )
    parser.add_argument(
        "--cache_path",
        default=None,
        help=(
            "Optional exact path for the generated pruned "
            "training feature cache."
        ),
    )
    parser.add_argument(
        "--encoder_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--checkpoint_criterion",
        default="val_rdm_mse",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--gpu",
        default="cuda:0",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--force_rebuild_cache",
        action="store_true",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    args = parser.parse_args()

    selection_path = (
        Path(
            args.selection_path
        )
        if args.selection_path
        is not None
        else find_latest_selection(
            args.subject
        )
    )

    source_features_path = Path(
        args.source_features_path
    )

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection file not found: {selection_path}"
        )

    if not source_features_path.exists():
        raise FileNotFoundError(
            "Source CLIP feature cache not found: "
            f"{source_features_path}"
        )

    selected = load_selection(
        selection_path
    )

    selection_subject = (
        selected["subject"]
    )

    if (
        selection_subject
        != args.subject
    ):
        raise ValueError(
            "Selection subject does not match --subject: "
            f"selection={selection_subject}, "
            f"requested={args.subject}"
        )

    removed_axes = (
        selected[
            "removed_axes_zero_based"
        ]
        .detach()
        .cpu()
        .long()
    )

    n_removed = int(
        removed_axes.numel()
    )

    if n_removed == 0:
        raise ValueError(
            "The selection contains no removed axes."
        )

    selection_stamp = (
        selection_path
        .parent
        .name
    )

    variant_name = (
        f"svd_greedy_remove{n_removed}_"
        f"{args.subject}_"
        f"{selection_stamp}"
    )

    cache_path = (
        Path(
            args.cache_path
        )
        if args.cache_path
        is not None
        else (
            REPO_ROOT
            / "features"
            / (
                "ViT-H-14_features_train_"
                f"{variant_name}.pt"
            )
        )
    )

    experiment_path = (
        Path(
            "baseline"
        )
        / "clip"
        / variant_name
        / "mse_contrastive"
        / args.checkpoint_criterion
    )

    output_dir = (
        GENERATION_DIR
        / "outputs"
        / experiment_path
    ).resolve()

    model_save_dir = (
        GENERATION_DIR
        / "models"
        / experiment_path
    ).resolve()

    print("=" * 72)
    print("Greedy-pruned SVD retraining")
    print("=" * 72)
    print(
        "Subject:",
        args.subject,
    )
    print(
        "Selection:",
        selection_path,
    )
    print(
        "Removed PCs:",
        selected.get(
            "removed_pcs_one_based"
        ),
    )
    print(
        "Dimensions:",
        f"1024 -> {1024 - n_removed}",
    )
    print(
        "Teacher cache:",
        cache_path,
    )
    print(
        "Checkpoint criterion:",
        args.checkpoint_criterion,
    )
    print(
        "Output dir:",
        output_dir,
    )
    print(
        "Model dir:",
        model_save_dir,
    )
    print("=" * 72)

    if args.dry_run:
        print(
            "Dry run only. "
            "No cache was created and training was not started."
        )
        return

    if (
        args.force_rebuild_cache
        or not cache_path.exists()
    ):
        print("")
        print(
            "Creating Greedy-pruned teacher cache..."
        )

        (
            source_cache,
            img_features,
        ) = load_feature_cache(
            source_features_path
        )

        reconstructed = (
            reconstruct_with_selected_mask(
                img_features=img_features,
                selected=selected,
                chunk_size=args.chunk_size,
            )
        )

        kept_variance_ratio = (
            save_pruned_cache(
                source_cache=source_cache,
                reconstructed=reconstructed,
                selected=selected,
                selection_path=selection_path,
                source_features_path=source_features_path,
                output_path=cache_path,
            )
        )

        print(
            "Saved teacher cache:",
            cache_path,
        )
        print(
            "img_features:",
            tuple(
                reconstructed.shape
            ),
        )
        print(
            "Kept training variance:",
            f"{kept_variance_ratio * 100.0:.6f}%",
        )
    else:
        print("")
        print(
            "Using existing teacher cache:",
            cache_path,
        )

    benchmark_script = (
        GENERATION_DIR
        / "benchmark_encoder_only_within_subject.sh"
    )

    env = os.environ.copy()

    env.update(
        {
            "SUBJECTS": args.subject,
            "IMAGE_ENCODER": "clip",
            "FEATURE_SPACE": "clip",
            "TRAIN_FEATURES_PATH": str(
                cache_path.resolve()
            ),
            "ENCODER_EPOCHS": str(
                args.encoder_epochs
            ),
            "BATCH_SIZE": str(
                args.batch_size
            ),
            "LR_ENCODER": str(
                args.lr_encoder
            ),
            "RSA_WEIGHT": "0.0",
            "RSA_LOSS_TYPE": "pearson",
            "CHECKPOINT_CRITERION": (
                args.checkpoint_criterion
            ),
            "VAL_RATIO": str(
                selected.get(
                    "val_ratio",
                    0.1,
                )
            ),
            "PATIENCE": str(
                args.patience
            ),
            "AVG_SIGNAL_TRAINING": "true",
            "SEED": str(
                args.seed
            ),
            "GPU": args.gpu,
            "METHOD": "baseline",
            "EXPERIMENT_TYPE": (
                "greedy_svd_pruning"
            ),
            "OUTPUT_DIR": str(
                output_dir
            ),
            "MODEL_SAVE_DIR": str(
                model_save_dir
            ),
        }
    )

    print("")
    print(
        "Starting encoder retraining + "
        "retrieval/generation evaluation..."
    )

    subprocess.run(
        [
            "bash",
            str(
                benchmark_script
            ),
        ],
        cwd=str(
            REPO_ROOT
        ),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
