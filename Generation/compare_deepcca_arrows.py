#!/usr/bin/env python3
"""Compare paired CLIP→EEG error arrows for baseline and Deep-CCA models.

Run from Generation/: python compare_deepcca_arrows.py --deepcca-path <best.pth>
No training, SDXL, diffusion prior or adapter is used here.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from eegdatasets import EEGDataset
from encoder_utils import stratified_condition_split
from models.atms import ATMS, extract_id_from_string

DEFAULT_BASELINE = (
    HERE / "models/baseline/clip/full/mse_contrastive/val_rdm_mse/"
    "encoder/sub-01/10-10_07-51/best.pth"
)


def infer(checkpoint, loader, device, subject_id):
    model = ATMS(outputs_dim=1024).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    predictions, references = [], []
    with torch.inference_mode():
        for eeg, _, _, _, _, img_features in loader:
            eeg = eeg.to(device)
            ids = torch.full(
                (eeg.size(0),), subject_id, device=device, dtype=torch.long
            )
            predictions.append(model(eeg, ids).float().cpu().numpy())
            references.append(img_features.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(references)


def stats_for_pc(true_pc, pred_pc):
    ratios, corrs = [], []
    for i in range(2):
        a, b = true_pc[:, i], pred_pc[:, i]
        std_a, std_b = np.std(a), np.std(b)
        ratios.append(float(std_b / std_a) if std_a > 0 else np.nan)
        corrs.append(
            float(np.corrcoef(a, b)[0, 1]) if std_a > 0 and std_b > 0
            else np.nan
        )
    return ratios, corrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepcca-path", type=Path, required=True)
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument(
        "--data-path", type=Path,
        default=Path("/home/moepy/ozakitakuma/data_eeg"),
    )
    parser.add_argument(
        "--img-dir-training", type=Path,
        default=Path("/home/moepy/ozakitakuma/data_image/training_images"),
    )
    parser.add_argument("--features-dir", type=Path, default=HERE.parent / "features")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "deepcca_arrow_comparison")
    parser.add_argument("--pc-x", type=int, default=600)
    parser.add_argument("--pc-y", type=int, default=601)
    parser.add_argument("--n-arrows", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not 1 <= args.pc_x <= 1024 or not 1 <= args.pc_y <= 1024:
        parser.error("PC numbers must be between 1 and 1024")
    if args.pc_x == args.pc_y:
        parser.error("Select two different PCs")
    if args.n_arrows < 0:
        parser.error("--n-arrows must be non-negative")
    for checkpoint in (args.deepcca_path, args.baseline_path):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading sub-01 averaged training EEG and CLIP features ...", flush=True)
    dataset = EEGDataset(
        str(args.data_path),
        img_dir_training=str(args.img_dir_training),
        subjects=[args.subject],
        train=True,
        avg_trials=True,
        feature_type="clip",
        feature_space="clip",
        features_dir=str(args.features_dir),
    )
    if len(dataset) != 16540:
        raise RuntimeError(f"Expected 16540 EEG samples, received {len(dataset)}")

    train_idx, val_idx = stratified_condition_split(
        n_classes=1654, conditions_per_class=10, trials_per_condition=1,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    # Fit PCA on model TRAIN images only, excluding the held-out validation
    # images, and use the SAME basis for both models.
    train_true = torch.as_tensor(dataset.img_features)[train_idx].cpu().double().numpy()
    mean = train_true.mean(axis=0)
    x = train_true - mean
    cov = x.T @ x / (len(x) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ordering = np.argsort(eigvals)[::-1]
    directions = eigvecs[:, ordering[:max(args.pc_x, args.pc_y)]]
    axes = directions[:, [args.pc_x - 1, args.pc_y - 1]]
    del x, cov, eigvals, eigvecs, train_true

    loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    device = torch.device(args.device)
    sid = extract_id_from_string(args.subject)
    print("Baseline inference ...", flush=True)
    baseline_pred, target = infer(args.baseline_path, loader, device, sid)
    print("Deep CCA inference ...", flush=True)
    cca_pred, cca_target = infer(args.deepcca_path, loader, device, sid)
    if not np.allclose(target, cca_target, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Different image features/order between model runs")

    true_pc = (target - mean) @ axes
    baseline_pc = (baseline_pred - mean) @ axes
    cca_pc = (cca_pred - mean) @ axes
    data = [("Baseline", baseline_pc), ("Baseline + Deep CCA", cca_pc)]

    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(len(val_idx), min(args.n_arrows, len(val_idx)), replace=False)

    fig, subplots = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    low = float(min(true_pc.min(), baseline_pc.min(), cca_pc.min()))
    high = float(max(true_pc.max(), baseline_pc.max(), cca_pc.max()))
    pad = max((high - low) * 0.08, 0.02)
    min_v, max_v = low - pad, high + pad

    summaries = []
    for ax, (name, pred_pc) in zip(subplots, data):
        ax.scatter(true_pc[:, 0], true_pc[:, 1], s=9, alpha=0.15,
                   color="tab:blue", label="True CLIP")
        ax.scatter(pred_pc[:, 0], pred_pc[:, 1], s=9, alpha=0.15,
                   color="tab:orange", label="EEG prediction")
        # Arrows run from the TRUE location to the corresponding PREDICTION.
        for j in chosen:
            ax.annotate(
                "", xy=pred_pc[j], xytext=true_pc[j],
                arrowprops=dict(arrowstyle="->", color="black",
                                alpha=0.6, lw=0.9, mutation_scale=9),
            )
        ax.scatter(true_pc[chosen, 0], true_pc[chosen, 1], s=17,
                   color="tab:blue", zorder=3)
        ax.scatter(pred_pc[chosen, 0], pred_pc[chosen, 1], s=17,
                   color="tab:orange", zorder=3)
        std_ratios, corrs = stats_for_pc(true_pc, pred_pc)
        errors = np.linalg.norm(pred_pc - true_pc, axis=1)
        closer = np.linalg.norm(pred_pc, axis=1) < np.linalg.norm(true_pc, axis=1)
        ax.set_title(
            f"{name} (validation N={len(val_idx)})\n"
            f"std ratio: {std_ratios[0]:.3f}, {std_ratios[1]:.3f}  "
            f"corr: {corrs[0]:.3f}, {corrs[1]:.3f}\n"
            f"mean 2D error={errors.mean():.3f}, inward={closer.mean():.1%}",
            fontsize=11,
        )
        ax.set_xlabel(f"PC{args.pc_x} coordinate")
        ax.set_ylabel(f"PC{args.pc_y} coordinate")
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, c="gray", lw=0.7)
        ax.axvline(0, c="gray", lw=0.7)
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8)
        summaries.append(
            dict(model=name, pc_x_std_ratio=std_ratios[0],
                 pc_y_std_ratio=std_ratios[1],
                 pc_x_corr=corrs[0], pc_y_corr=corrs[1],
                 mean_2d_error=float(errors.mean()),
                 inward_fraction=float(closer.mean())),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"pc{args.pc_x:04d}_pc{args.pc_y:04d}_baseline_vs_deepcca.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    with (args.output_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    paired_path = args.output_dir / "validation_paired_pc_coordinates.csv"
    with paired_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset_index", "true_x", "true_y",
                         "baseline_x", "baseline_y", "deepcca_x", "deepcca_y"])
        for j, original_index in enumerate(val_idx):
            writer.writerow([
                original_index, *true_pc[j], *baseline_pc[j], *cca_pc[j],
            ])

    print("Plot:", plot_path)
    print("Metrics:", args.output_dir / "summary.csv")
    print("Coordinates:", paired_path)


if __name__ == "__main__":
    main()
