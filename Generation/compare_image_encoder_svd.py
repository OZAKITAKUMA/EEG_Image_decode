#!/usr/bin/env python3
"""Compare PCA covariance eigenvalue spectra of cached image features.

Usage (from repository root):
    python Generation/compare_image_encoder_svd.py

Requires: torch, numpy, matplotlib. This script is read-only for input caches.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_CACHES = {
    "CLIP ViT-H/14": "ViT-H-14_features_train.pt",
    "SigLIP2 Large": "SigLIP2-Large-Patch16-256_features_train.pt",
    "DINOv3 ViT-L/16": "DINOv3-ViTL16_features_train.pt",
}


def load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    # Use only local/trusted feature caches: torch.load deserializes pickle data.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "img_features" not in obj:
        raise ValueError(f"Expected dictionary with 'img_features' key: {path}")
    features = obj["img_features"]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"'img_features' must be a 2D torch.Tensor: {path}")
    arr = features.detach().cpu().numpy().astype(np.float64, copy=False)
    if arr.shape[0] < 2 or arr.shape[1] < 1 or not np.isfinite(arr).all():
        raise ValueError(f"Invalid shape or nonfinite values: {path}: {arr.shape}")
    return arr


def covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    """Eigenvalues of unbiased sample covariance, descending.

    Equivalent to squared singular values of centered X divided by (N-1).
    """
    centered = x - x.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / (len(x) - 1)
    eig = np.linalg.eigvalsh(covariance)[::-1]
    return np.clip(eig, 0.0, None)


def metrics(eig: np.ndarray) -> tuple[float, int, int]:
    total = eig.sum()
    if total <= 0:
        raise ValueError("Feature covariance has zero total variance")
    p = eig / total
    positive = p[p > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    cumulative = np.cumsum(p)
    k90 = int(np.searchsorted(cumulative, 0.9) + 1)
    k99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return effective_rank, k90, k99


def draw_curves(values: dict[str, np.ndarray], out: Path, title: str,
                ylabel: str, *, logy: bool = False, percentage: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, y in values.items():
        xx = np.arange(1, len(y) + 1)
        ax.plot(xx, y * (100 if percentage else 1), label=name, linewidth=1.7)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Principal component index (descending variance)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("features"),
                        help="Feature cache folder, relative to current working directory")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Generation/outputs/image_encoder_svd_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectra: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    common_n = None
    for name, filename in DEFAULT_CACHES.items():
        path = args.features_dir / filename
        print(f"[LOAD] {name}: {path}", flush=True)
        x = load_matrix(path)
        print(f"       shape={x.shape}, dtype={x.dtype}", flush=True)
        if common_n is None:
            common_n = x.shape[0]
        elif x.shape[0] != common_n:
            raise ValueError(f"Sample count differs: {name}: {x.shape[0]} vs {common_n}")
        eig = covariance_eigenvalues(x)
        spectra[name] = eig
        erank, k90, k99 = metrics(eig)
        rows.append({"model": name, "samples": x.shape[0], "dimensions": x.shape[1],
                     "total_variance": float(eig.sum()), "effective_rank": erank,
                     "pc90": k90, "pc99": k99})
        print(f"       effective_rank={erank:.2f}, PC90={k90}, PC99={k99}", flush=True)

    # Absolute scale can differ across encoders.
    draw_curves(spectra, args.output_dir / "01_eigenvalues_log.png",
                "Covariance eigenvalue spectra (absolute scale)",
                "Eigenvalue (variance; log scale)", logy=True)
    normalized = {name: eig / eig.sum() for name, eig in spectra.items()}
    draw_curves(normalized, args.output_dir / "02_normalized_eigenvalues_log.png",
                "Normalized covariance eigenvalue spectra",
                "Eigenvalue / total variance (log scale)", logy=True)
    cum = {name: np.cumsum(eig / eig.sum()) for name, eig in spectra.items()}
    draw_curves(cum, args.output_dir / "03_cumulative_explained_variance.png",
                "Cumulative explained variance", "Cumulative explained variance (%)",
                percentage=True)

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "eigenvalues.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_index"] + list(spectra))
        for k in range(max(map(len, spectra.values()))):
            writer.writerow([k + 1] + [float(v[k]) if k < len(v) else "" for v in spectra.values()])
    print(f"[DONE] Saved 3 plots and 2 CSV files to {args.output_dir}")
    print("Note: PCA directions are independently fitted for each encoder; PC indices are ranks, not shared directions.")


if __name__ == "__main__":
    main()
