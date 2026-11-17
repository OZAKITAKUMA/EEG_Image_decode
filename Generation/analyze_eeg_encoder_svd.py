#!/usr/bin/env python3
"""SVD analysis of ATMS EEG-predicted image features vs. target CLIP features.

Run this from Generation/ in OZAKITAKUMA/EEG_Image_decode.
Does NOT train the encoder and does NOT modify GitHub files.
Analyzes 1024-D ATMS outputs, not raw EEG or internal 1440-D embeddings.

Plots:
  1) singular-value spectra, each representation's OWN SVD;
  2) prediction/reference spread ratio in the SAME train-CLIP PCA basis.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from encoder_utils import stratified_condition_split
from models.atms import ATMS, extract_id_from_string


def load_eeg(eeg_path):
    raw = np.load(eeg_path, allow_pickle=True)
    try:
        eeg = raw['preprocessed_eeg_data']
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError('EEG file must contain preprocessed_eeg_data') from exc
    # avg_trials=true: (16540, 4, 63, 250) -> (16540, 63, 250)
    if eeg.ndim == 4:
        eeg = eeg.mean(axis=1)
    if eeg.ndim != 3 or eeg.shape[1:] != (63, 250):
        raise ValueError(f'Expected averaged EEG [N,63,250], got {eeg.shape}')
    return torch.from_numpy(np.asarray(eeg, dtype=np.float32))


def load_clip(cache_path):
    saved = torch.load(cache_path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or 'img_features' not in saved:
        raise ValueError('CLIP cache must be a dict with key img_features')
    features = torch.as_tensor(saved['img_features']).float().cpu()
    if features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError(f'Expected CLIP [N,1024], got {tuple(features.shape)}')
    return features


def infer(model, eeg, batch_size, device, subject_id):
    loader = DataLoader(TensorDataset(eeg), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    output = []
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(device)
            ids = torch.full((len(batch),), subject_id,
                             dtype=torch.long, device=device)
            output.append(model(batch, ids).float().cpu())
    return torch.cat(output)


def eigenspectrum(features):
    centered = features.double() - features.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(features) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)
    return eigenvalues.numpy()


def effective_rank(evals):
    total = float(evals.sum())
    if total <= 0:
        return 0.0
    p = evals / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='ATMS best.pth (same baseline as the previous analysis)')
    p.add_argument('--eeg-file', type=Path, required=True,
                   help='sub-01/preprocessed_eeg_training.npy')
    p.add_argument('--clip-cache', type=Path,
                   default=ROOT / 'features/ViT-H-14_features_train.pt')
    p.add_argument('--subject', default='sub-01')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--output-dir', type=Path,
                   default=Path('outputs/eeg_encoder_svd_sub01'))
    p.add_argument('--no-subject-id', action='store_true',
                   help='Use the shared subject token for an ID-free checkpoint')
    args = p.parse_args()

    for path in (args.checkpoint, args.eeg_file, args.clip_cache):
        if not path.is_file():
            p.error(f'File not found: {path}')
    if args.batch_size < 1:
        p.error('batch-size must be positive')

    torch.manual_seed(args.seed)
    eeg = load_eeg(args.eeg_file)
    clip = load_clip(args.clip_cache)
    if len(eeg) != 16540 or len(clip) != 16540:
        raise ValueError(f'Expected 16540 paired samples, got EEG={len(eeg)}, CLIP={len(clip)}')

    print(f'EEG: {tuple(eeg.shape)}, CLIP: {tuple(clip.shape)}', flush=True)
    device = torch.device(args.device)
    model = ATMS(outputs_dim=1024).to(device)
    weights = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(weights, strict=True)
    model.eval()
    subject_id = 0 if args.no_subject_id else extract_id_from_string(args.subject)
    print('Computing EEG Encoder outputs...', flush=True)
    predictions = infer(model, eeg, args.batch_size, device, subject_id)
    del model, eeg
    if not torch.isfinite(predictions).all() or not torch.isfinite(clip).all():
        raise ValueError('Non-finite value found in features')

    train_idx, val_idx = stratified_condition_split(
        n_classes=1654, conditions_per_class=10,
        trials_per_condition=1, val_ratio=args.val_ratio, seed=args.seed)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    assert len(train_idx) + len(val_idx) == len(clip)

    # Fit the CLIP coordinate system on TRAIN only; never fit on validation.
    clip_train = clip[train_idx].double()
    mu = clip_train.mean(dim=0, keepdim=True)
    centered_train = clip_train - mu
    covariance = centered_train.T @ centered_train / (len(clip_train) - 1)
    _, eigenvectors_asc = torch.linalg.eigh(covariance)
    basis = eigenvectors_asc.flip(dims=[1])  # [1024,1024], largest variance first

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'pc_comparison.csv'
    rows = []
    spectra = {}

    for split_name, indices in [('train', train_idx), ('validation', val_idx)]:
        true = clip[indices].double()
        pred = predictions[indices].double()
        # PCA coordinates in fixed TRAIN CLIP reference basis.
        true_pc = (true - mu) @ basis
        pred_pc = (pred - mu) @ basis
        std_true = true_pc.std(dim=0, unbiased=True).numpy()
        std_pred = pred_pc.std(dim=0, unbiased=True).numpy()
        ratio = np.divide(std_pred, std_true,
                          out=np.full_like(std_pred, np.nan), where=std_true > 1e-12)
        pred_mean = pred_pc.mean(dim=0).numpy()
        true_mean = true_pc.mean(dim=0).numpy()
        for k in range(1024):
            rows.append(dict(split=split_name, pc=k+1,
                             clip_std=std_true[k], eeg_pred_std=std_pred[k],
                             std_ratio=ratio[k],
                             mean_difference=pred_mean[k]-true_mean[k]))
        eig_true = eigenspectrum(true)
        eig_pred = eigenspectrum(pred)
        spectra[split_name] = (eig_true, eig_pred)
        for title, eig in [('CLIP', eig_true), ('EEG prediction', eig_pred)]:
            print(f'{split_name} {title}: effective rank={effective_rank(eig):.2f}, '
                  f'90%-variance components={np.searchsorted(np.cumsum(eig)/eig.sum(), 0.9)+1}',
                  flush=True)
        for pc in [1, 2, 100, 256, 500, 600, 601, 800, 1024]:
            print(f'  {split_name} PC{pc}: std ratio={ratio[pc-1]:.3f}')

    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # The two spectra have independent SVD bases, so this visualizes intrinsic
    # dimensional collapse, NOT same-direction per-PC alignment.
    for split_name, (eig_true, eig_pred) in spectra.items():
        plt.figure(figsize=(9, 5.4))
        ks = np.arange(1, 1025)
        plt.plot(ks, np.sqrt(eig_true), label='True CLIP (own SVD)')
        plt.plot(ks, np.sqrt(eig_pred), label='EEG prediction (own SVD)')
        plt.yscale('log')
        plt.xlabel('Singular direction rank (descending)')
        plt.ylabel('Standard deviation (log scale)')
        plt.title(f'Singular-value spectrum (scaled by sample count): {split_name}')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / f'{split_name}_svd_spectrum.png', dpi=180)
        plt.close()

    plt.figure(figsize=(10, 5.5))
    for split_name in ['train', 'validation']:
        values = [r['std_ratio'] for r in rows if r['split'] == split_name]
        plt.plot(np.arange(1, 1025), values, label=split_name, alpha=0.85)
    plt.axhline(1, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('CLIP PCA component (fit on training CLIP)')
    plt.ylabel('EEG prediction std / CLIP std')
    plt.title('EEG prediction contraction in fixed CLIP PCA directions')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_dir / 'clip_pc_std_ratio.png', dpi=180)
    plt.close()

    print('\nSaved to:', args.output_dir.resolve())
    print(' - train_svd_spectrum.png / validation_svd_spectrum.png')
    print(' - clip_pc_std_ratio.png / pc_comparison.csv')


if __name__ == '__main__':
    main()
