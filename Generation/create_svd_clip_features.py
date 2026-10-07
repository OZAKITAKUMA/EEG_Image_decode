#!/usr/bin/env python3
"""
Create SVD/PCA-transformed image-feature caches for EEG experiments.

The SVD basis is always fitted only on the training portion of the existing
9:1 condition split (9 conditions per class). The held-out validation
condition is never used to estimate the centering statistics or the SVD basis.

Centering modes
---------------
global:
    Subtract one mean vector computed from all training conditions.

class:
    Compute one mean vector per class from that class's 9 training conditions,
    subtract the corresponding class mean, then pool all class-centered
    residuals before fitting one SVD basis.

Component modes
---------------
keep_top_rank:
    Keep only the first k principal directions. This reproduces the previous
    low-rank reconstruction when centering=global.

remove_top / remove_middle / remove_bottom:
    Keep every principal direction except the requested block.

remove_range:
    Remove an arbitrary contiguous block. --remove_start is 1-based.

All transformed features are reconstructed back to the original feature
dimension.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder_utils import stratified_condition_split


N_CLASSES = 1654
CONDITIONS_PER_CLASS = 10
DEFAULT_RANKS = [1008, 992, 960, 896, 768]

CENTERING_CHOICES = [
    "global",
    "class",
]

COMPONENT_MODE_CHOICES = [
    "keep_top_rank",
    "remove_top",
    "remove_middle",
    "remove_bottom",
    "remove_range",
]


def load_feature_cache(path):
    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(saved, dict):
        raise TypeError(
            f"Expected a dict feature cache, got {type(saved).__name__}: {path}"
        )

    if "img_features" not in saved:
        raise KeyError(
            f"'img_features' was not found in feature cache: {path}"
        )

    img_features = saved["img_features"]

    if not torch.is_tensor(img_features):
        raise TypeError(
            f"'img_features' must be a torch.Tensor: {path}"
        )

    if img_features.ndim != 2:
        raise ValueError(
            "Expected img_features to have shape (N, D), "
            f"but got {tuple(img_features.shape)} in {path}"
        )

    return saved, img_features.detach().cpu()


def compute_train_fit_indices(seed, val_ratio):
    train_indices, val_indices = stratified_condition_split(
        n_classes=N_CLASSES,
        conditions_per_class=CONDITIONS_PER_CLASS,
        trials_per_condition=1,
        val_ratio=val_ratio,
        seed=seed,
    )

    expected_total = N_CLASSES * CONDITIONS_PER_CLASS
    expected_val = N_CLASSES
    expected_train = expected_total - expected_val

    if len(train_indices) != expected_train:
        raise RuntimeError(
            "Unexpected number of SVD-fit training conditions: "
            f"expected={expected_train}, actual={len(train_indices)}"
        )

    if len(val_indices) != expected_val:
        raise RuntimeError(
            "Unexpected number of held-out validation conditions: "
            f"expected={expected_val}, actual={len(val_indices)}"
        )

    return train_indices, val_indices


def compute_class_means(img_features, train_indices):
    x = img_features.float()

    expected_samples = N_CLASSES * CONDITIONS_PER_CLASS

    if x.shape[0] != expected_samples:
        raise ValueError(
            "Class centering assumes the standard training layout "
            f"({expected_samples} samples), got {x.shape[0]}."
        )

    x_by_class = x.reshape(
        N_CLASSES,
        CONDITIONS_PER_CLASS,
        x.shape[1],
    )

    train_mask = torch.zeros(
        expected_samples,
        dtype=torch.bool,
    )
    train_mask[
        torch.as_tensor(
            train_indices,
            dtype=torch.long,
        )
    ] = True

    train_mask = train_mask.reshape(
        N_CLASSES,
        CONDITIONS_PER_CLASS,
    )

    counts = train_mask.sum(dim=1)

    if not torch.all(counts == counts[0]):
        raise RuntimeError(
            "Every class must contribute the same number of training conditions."
        )

    class_means = (
        x_by_class
        * train_mask.unsqueeze(-1)
    ).sum(dim=1) / counts.unsqueeze(-1)

    return class_means


def fit_centered_svd(
    img_features,
    train_indices,
    centering,
    device,
):
    x = img_features.float()

    index_tensor = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    x_fit = x[index_tensor]

    if centering == "global":
        centers = x_fit.mean(
            dim=0,
            keepdim=True,
        )

        x_fit_centered = (
            x_fit
            - centers
        )

    elif centering == "class":
        centers = compute_class_means(
            img_features,
            train_indices,
        )

        fit_class_ids = (
            index_tensor
            // CONDITIONS_PER_CLASS
        )

        x_fit_centered = (
            x_fit
            - centers[fit_class_ids]
        )

    else:
        raise ValueError(
            f"Unknown centering mode: {centering}"
        )

    x_fit_centered = x_fit_centered.to(device)

    print(
        "Fitting centered SVD:",
        f"centering={centering}",
        f"shape={tuple(x_fit_centered.shape)}",
        f"device={device}",
    )

    _, singular_values, vh = torch.linalg.svd(
        x_fit_centered,
        full_matrices=False,
    )

    return (
        centers.cpu(),
        singular_values.cpu(),
        vh.cpu(),
    )


def component_keep_mask(
    feature_dim,
    component_mode,
    rank=None,
    remove_count=None,
    remove_start=None,
):
    keep_mask = torch.ones(
        feature_dim,
        dtype=torch.bool,
    )

    if component_mode == "keep_top_rank":
        if rank is None or not 1 <= rank <= feature_dim:
            raise ValueError(
                f"rank must be in [1, {feature_dim}], got {rank}"
            )

        keep_mask[rank:] = False
        removed_start = rank
        removed_end = feature_dim

    else:
        if remove_count is None or not 1 <= remove_count < feature_dim:
            raise ValueError(
                "remove_count must be in "
                f"[1, {feature_dim - 1}], got {remove_count}"
            )

        if component_mode == "remove_top":
            start = 0

        elif component_mode == "remove_bottom":
            start = feature_dim - remove_count

        elif component_mode == "remove_middle":
            start = (
                feature_dim
                - remove_count
            ) // 2

        elif component_mode == "remove_range":
            if remove_start is None:
                raise ValueError(
                    "remove_range requires --remove_start."
                )

            start = remove_start - 1

            if (
                start < 0
                or start + remove_count > feature_dim
            ):
                raise ValueError(
                    "Requested component range is outside "
                    f"PC1-PC{feature_dim}."
                )

        else:
            raise ValueError(
                f"Unknown component mode: {component_mode}"
            )

        end = start + remove_count
        keep_mask[start:end] = False
        removed_start = start
        removed_end = end

    return keep_mask, removed_start, removed_end


def center_for_rows(
    centers,
    centering,
    start,
    end,
):
    if centering == "global":
        return centers

    row_ids = torch.arange(
        start,
        end,
        dtype=torch.long,
    )

    class_ids = (
        row_ids
        // CONDITIONS_PER_CLASS
    )

    return centers[class_ids]


def reconstruct_features(
    img_features,
    centers,
    centering,
    vh,
    keep_mask,
    chunk_size,
):
    original_dtype = img_features.dtype
    x = img_features.float()

    components = vh[
        keep_mask
    ].float()

    reconstructed = torch.empty_like(
        x,
        dtype=torch.float32,
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

        row_centers = center_for_rows(
            centers=centers,
            centering=centering,
            start=start,
            end=end,
        ).float()

        centered = (
            x[start:end]
            - row_centers
        )

        coefficients = (
            centered
            @ components.T
        )

        reconstructed[start:end] = (
            coefficients
            @ components
            + row_centers
        )

    return reconstructed.to(
        dtype=original_dtype
    )


def kept_variance_ratio(
    singular_values,
    keep_mask,
):
    variance = (
        singular_values.float()
        ** 2
    )

    total = variance.sum()

    if total <= 0:
        return 0.0

    return (
        variance[keep_mask].sum()
        / total
    ).item()


def save_reconstructed_cache(
    source_cache,
    reconstructed_features,
    output_path,
    metadata,
):
    output = dict(source_cache)
    output["img_features"] = reconstructed_features.cpu()

    previous_metadata = output.get(
        "feature_metadata"
    )

    feature_metadata = {
        "transform": "flexible_centered_svd_reconstruction",
        **metadata,
    }

    if previous_metadata is not None:
        feature_metadata["source_feature_metadata"] = previous_metadata

    output["feature_metadata"] = feature_metadata

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output,
        output_path,
    )


def build_default_output_path(
    input_path,
    output_dir,
    centering,
    component_mode,
    rank,
    remove_count,
    remove_start,
):
    source = Path(input_path)

    if component_mode == "keep_top_rank":
        if centering == "global":
            suffix = f"_svd{rank}"
        else:
            suffix = f"_svd_{centering}_keep{rank}"

    elif component_mode == "remove_range":
        end = remove_start + remove_count - 1
        suffix = (
            f"_svd_{centering}_"
            f"remove_pc{remove_start}-{end}"
        )

    else:
        position = component_mode.removeprefix(
            "remove_"
        )
        suffix = (
            f"_svd_{centering}_"
            f"remove_{position}{remove_count}"
        )

    directory = (
        Path(output_dir)
        if output_dir is not None
        else source.parent
    )

    return (
        directory
        / f"{source.stem}{suffix}{source.suffix}"
    )


def run_one_transform(
    args,
    source_cache,
    train_features,
    centers,
    singular_values,
    vh,
    rank=None,
):
    feature_dim = train_features.shape[1]

    keep_mask, removed_start, removed_end = component_keep_mask(
        feature_dim=feature_dim,
        component_mode=args.component_mode,
        rank=rank,
        remove_count=args.remove_count,
        remove_start=args.remove_start,
    )

    ratio = kept_variance_ratio(
        singular_values,
        keep_mask,
    )

    print("")
    print("=" * 70)
    print(
        "Transform:",
        f"centering={args.centering}",
        f"mode={args.component_mode}",
    )
    print(
        "Kept components:",
        int(keep_mask.sum().item()),
        "/",
        feature_dim,
    )
    print(
        "Kept training variance:",
        f"{ratio * 100.0:.6f}%",
    )

    reconstructed = reconstruct_features(
        train_features,
        centers=centers,
        centering=args.centering,
        vh=vh,
        keep_mask=keep_mask,
        chunk_size=args.chunk_size,
    )

    if args.output_path is not None:
        output_path = Path(
            args.output_path
        )
    else:
        output_path = build_default_output_path(
            input_path=args.train_features_path,
            output_dir=args.output_dir,
            centering=args.centering,
            component_mode=args.component_mode,
            rank=rank,
            remove_count=args.remove_count,
            remove_start=args.remove_start,
        )

    metadata = {
        "centering": args.centering,
        "component_mode": args.component_mode,
        "rank": rank,
        "remove_count": args.remove_count,
        "remove_start": args.remove_start,
        "removed_component_start_zero_based": removed_start,
        "removed_component_end_exclusive_zero_based": removed_end,
        "original_dim": feature_dim,
        "kept_component_count": int(
            keep_mask.sum().item()
        ),
        "kept_variance_ratio": ratio,
        "svd_fit_scope": "train_9_of_10_conditions_per_class",
        "svd_fit_seed": args.seed,
        "svd_fit_val_ratio": args.val_ratio,
        "svd_l2_normalized_before_fit": False,
        "source_features_path": os.path.abspath(
            args.train_features_path
        ),
    }

    save_reconstructed_cache(
        source_cache=source_cache,
        reconstructed_features=reconstructed,
        output_path=output_path,
        metadata=metadata,
    )

    print(
        "Saved train cache:",
        output_path,
    )
    print(
        "  img_features:",
        tuple(reconstructed.shape),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create flexible SVD/PCA-transformed image-feature caches "
            "using only the training 9/10 conditions to fit the transform."
        )
    )

    parser.add_argument(
        "--train_features_path",
        default="../features/ViT-H-14_features_train.pt",
    )

    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Exact output cache path. Intended for experiment_launcher.py. "
            "Cannot be used with multiple --ranks."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Legacy/manual output directory when --output_path is omitted."
        ),
    )

    parser.add_argument(
        "--centering",
        choices=CENTERING_CHOICES,
        default="global",
    )

    parser.add_argument(
        "--component_mode",
        choices=COMPONENT_MODE_CHOICES,
        default="keep_top_rank",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank used by keep_top_rank.",
    )

    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Legacy multi-rank interface. Only valid with keep_top_rank. "
            "If neither --rank nor --ranks is given, the old default ranks "
            "1008 992 960 896 768 are generated."
        ),
    )

    parser.add_argument(
        "--remove_count",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--remove_start",
        type=int,
        default=None,
        help="1-based start PC used only by remove_range.",
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
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--save_basis",
        action="store_true",
    )

    args = parser.parse_args()

    source_cache, train_features = load_feature_cache(
        args.train_features_path
    )

    expected_train_samples = (
        N_CLASSES
        * CONDITIONS_PER_CLASS
    )

    if train_features.shape[0] != expected_train_samples:
        raise ValueError(
            "Unexpected number of training image features: "
            f"expected={expected_train_samples}, "
            f"actual={train_features.shape[0]}"
        )

    if (
        args.component_mode != "keep_top_rank"
        and args.ranks is not None
    ):
        raise ValueError(
            "--ranks can only be used with keep_top_rank."
        )

    if (
        args.output_path is not None
        and args.ranks is not None
        and len(args.ranks) > 1
    ):
        raise ValueError(
            "--output_path cannot be used with multiple --ranks."
        )

    train_indices, val_indices = compute_train_fit_indices(
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    print(
        "Training feature shape:",
        tuple(train_features.shape),
    )
    print(
        "SVD-fit conditions:",
        len(train_indices),
    )
    print(
        "Held-out validation conditions:",
        len(val_indices),
    )

    centers, singular_values, vh = fit_centered_svd(
        train_features,
        train_indices=train_indices,
        centering=args.centering,
        device=torch.device(args.device),
    )

    if args.save_basis:
        source = Path(
            args.train_features_path
        )
        basis_path = (
            Path(args.output_dir)
            if args.output_dir is not None
            else source.parent
        ) / (
            f"{source.stem}_svd_{args.centering}_basis"
            f"{source.suffix}"
        )

        torch.save(
            {
                "centering": args.centering,
                "centers": centers,
                "singular_values": singular_values,
                "vh": vh,
                "train_indices": torch.as_tensor(
                    train_indices,
                    dtype=torch.long,
                ),
                "val_indices": torch.as_tensor(
                    val_indices,
                    dtype=torch.long,
                ),
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "n_classes": N_CLASSES,
                "conditions_per_class": CONDITIONS_PER_CLASS,
            },
            basis_path,
        )

        print(
            "Saved SVD basis:",
            basis_path,
        )

    if args.component_mode == "keep_top_rank":
        if args.ranks is not None:
            ranks = sorted(
                set(args.ranks),
                reverse=True,
            )
        elif args.rank is not None:
            ranks = [args.rank]
        else:
            ranks = DEFAULT_RANKS

        for rank in ranks:
            run_one_transform(
                args=args,
                source_cache=source_cache,
                train_features=train_features,
                centers=centers,
                singular_values=singular_values,
                vh=vh,
                rank=rank,
            )

    else:
        run_one_transform(
            args=args,
            source_cache=source_cache,
            train_features=train_features,
            centers=centers,
            singular_values=singular_values,
            vh=vh,
            rank=None,
        )

    print("")
    print(
        "Finished creating SVD-transformed image-feature cache."
    )


if __name__ == "__main__":
    main()
