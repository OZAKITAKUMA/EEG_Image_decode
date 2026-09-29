#!/usr/bin/env python3
"""
Create low-rank reconstructed CLIP image-feature caches with centered SVD.

The SVD basis is fitted only on the training portion of the 9:1
within-subject condition split (9 conditions per class). The held-out
validation condition is therefore not used to estimate the mean or SVD basis.

For each requested rank k, the original D-dimensional CLIP image feature x is
reconstructed in the original coordinate system as

    x_recon = (x - mean) @ V_k.T @ V_k + mean

where V_k contains the first k right-singular vectors fitted on the training
features.

The saved img_features keep the original feature dimensionality (1024 for
ViT-H-14 projected CLIP), while lower-variance directions beyond rank k are
removed.

By default the script creates ranks:
    1008, 992, 960, 896, 768

Example
-------
From Generation/:

    python create_svd_clip_features.py

From the repository root:

    python Generation/create_svd_clip_features.py \
        --train_features_path ./features/ViT-H-14_features_train.pt
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


def add_suffix(path, suffix, output_dir):
    source = Path(path)
    filename = f"{source.stem}{suffix}{source.suffix}"
    return Path(output_dir) / filename


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


def fit_centered_svd(img_features, train_indices, device):
    x = img_features.float()

    index_tensor = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    x_fit = x[index_tensor].to(device)

    mean = x_fit.mean(
        dim=0,
        keepdim=True,
    )

    x_fit_centered = x_fit - mean

    print(
        "Fitting centered SVD:",
        f"shape={tuple(x_fit_centered.shape)}",
        f"device={device}",
    )

    _, singular_values, vh = torch.linalg.svd(
        x_fit_centered,
        full_matrices=False,
    )

    return (
        mean.cpu(),
        singular_values.cpu(),
        vh.cpu(),
    )


def reconstruct_features(
    img_features,
    mean,
    vh,
    rank,
    chunk_size,
):
    original_dtype = img_features.dtype
    x = img_features.float()

    feature_dim = x.shape[1]

    if not 1 <= rank <= feature_dim:
        raise ValueError(
            f"rank must be in [1, {feature_dim}], got {rank}"
        )

    components = vh[:rank].float()
    mean = mean.float()

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

        centered = (
            x[start:end]
            - mean
        )

        low_dim = (
            centered
            @ components.T
        )

        reconstructed[start:end] = (
            low_dim
            @ components
            + mean
        )

    return reconstructed.to(
        dtype=original_dtype
    )


def retained_variance_ratio(
    singular_values,
    rank,
):
    variance = (
        singular_values.float()
        ** 2
    )

    total = variance.sum()

    if total <= 0:
        return 0.0

    return (
        variance[:rank].sum()
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
        "transform": "centered_svd_reconstruction",
        **metadata,
    }

    if previous_metadata is not None:
        feature_metadata["source_feature_metadata"] = previous_metadata

    output["feature_metadata"] = feature_metadata

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    torch.save(
        output,
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create low-rank reconstructed CLIP feature caches "
            "using centered SVD fitted only on the training 9/10 conditions."
        )
    )

    parser.add_argument(
        "--train_features_path",
        default="../features/ViT-H-14_features_train.pt",
        help=(
            "Original training CLIP feature cache. "
            "Default assumes execution from Generation/."
        ),
    )

    parser.add_argument(
        "--test_features_path",
        default=None,
        help=(
            "Optional test CLIP feature cache. If supplied, the same "
            "training-fitted SVD basis is applied to the test features."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for reconstructed feature caches. "
            "Default: same directory as --train_features_path."
        ),
    )

    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=DEFAULT_RANKS,
        help=(
            "SVD ranks to keep. Default: "
            + " ".join(str(rank) for rank in DEFAULT_RANKS)
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to reproduce the 9:1 condition split.",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio used by the training split.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device used only while fitting the SVD, e.g. cpu or cuda:0. "
            "Reconstruction and saving are performed on CPU."
        ),
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="Number of feature rows reconstructed at once.",
    )

    parser.add_argument(
        "--save_basis",
        action="store_true",
        help=(
            "Also save the fitted mean, singular values, right-singular "
            "vectors, and fit/validation indices."
        ),
    )

    args = parser.parse_args()

    train_cache, train_features = load_feature_cache(
        args.train_features_path
    )

    expected_train_samples = (
        N_CLASSES
        * CONDITIONS_PER_CLASS
    )

    if train_features.shape[0] != expected_train_samples:
        raise ValueError(
            "Unexpected number of training CLIP features: "
            f"expected={expected_train_samples}, "
            f"actual={train_features.shape[0]}"
        )

    feature_dim = train_features.shape[1]

    ranks = sorted(
        set(args.ranks),
        reverse=True,
    )

    for rank in ranks:
        if not 1 <= rank <= feature_dim:
            raise ValueError(
                f"All ranks must be in [1, {feature_dim}], got {rank}"
            )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.dirname(
            os.path.abspath(
                args.train_features_path
            )
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
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
    print(
        "Requested ranks:",
        ranks,
    )

    mean, singular_values, vh = fit_centered_svd(
        train_features,
        train_indices=train_indices,
        device=torch.device(args.device),
    )

    if args.save_basis:
        basis_path = add_suffix(
            args.train_features_path,
            "_svd_basis",
            output_dir,
        )

        torch.save(
            {
                "mean": mean,
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

    test_cache = None
    test_features = None

    if args.test_features_path is not None:
        test_cache, test_features = load_feature_cache(
            args.test_features_path
        )

        if test_features.shape[1] != feature_dim:
            raise ValueError(
                "Train/test feature dimensions do not match: "
                f"train={feature_dim}, "
                f"test={test_features.shape[1]}"
            )

        print(
            "Test feature shape:",
            tuple(test_features.shape),
        )

    for rank in ranks:
        retained_ratio = retained_variance_ratio(
            singular_values,
            rank,
        )

        print("")
        print(
            "=" * 70
        )
        print(
            f"rank={rank} / {feature_dim}"
        )
        print(
            "Retained training variance:",
            f"{retained_ratio * 100.0:.6f}%",
        )

        reconstructed_train = reconstruct_features(
            train_features,
            mean=mean,
            vh=vh,
            rank=rank,
            chunk_size=args.chunk_size,
        )

        train_output_path = add_suffix(
            args.train_features_path,
            f"_svd{rank}",
            output_dir,
        )

        metadata = {
            "rank": rank,
            "original_dim": feature_dim,
            "retained_variance_ratio": retained_ratio,
            "svd_fit_scope": "train_9_of_10_conditions_per_class",
            "svd_fit_seed": args.seed,
            "svd_fit_val_ratio": args.val_ratio,
            "svd_centered": True,
            "svd_l2_normalized_before_fit": False,
            "source_features_path": os.path.abspath(
                args.train_features_path
            ),
        }

        save_reconstructed_cache(
            source_cache=train_cache,
            reconstructed_features=reconstructed_train,
            output_path=str(train_output_path),
            metadata=metadata,
        )

        print(
            "Saved train cache:",
            train_output_path,
        )
        print(
            "  img_features:",
            tuple(reconstructed_train.shape),
        )

        if test_cache is not None:
            reconstructed_test = reconstruct_features(
                test_features,
                mean=mean,
                vh=vh,
                rank=rank,
                chunk_size=args.chunk_size,
            )

            test_output_path = add_suffix(
                args.test_features_path,
                f"_svd{rank}",
                output_dir,
            )

            test_metadata = dict(metadata)
            test_metadata[
                "source_features_path"
            ] = os.path.abspath(
                args.test_features_path
            )
            test_metadata[
                "svd_basis_source"
            ] = os.path.abspath(
                args.train_features_path
            )

            save_reconstructed_cache(
                source_cache=test_cache,
                reconstructed_features=reconstructed_test,
                output_path=str(test_output_path),
                metadata=test_metadata,
            )

            print(
                "Saved test cache:",
                test_output_path,
            )
            print(
                "  img_features:",
                tuple(reconstructed_test.shape),
            )

    print("")
    print(
        "Finished creating SVD-reconstructed CLIP feature caches."
    )


if __name__ == "__main__":
    main()
