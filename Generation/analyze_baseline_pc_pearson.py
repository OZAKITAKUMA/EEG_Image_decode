#!/usr/bin/env python3
"""PC-wise target variance, raw MSE, NMSE and Pearson r from specified ATMS checkpoint.
Run from Generation/.  No generation, prior, adapter, or Git modifications.
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
from torch.utils.data import DataLoader

# Generation/ script must be able to import parent-level eegdatasets.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eegdatasets import EEGDataset
from models.atms import ATMS, extract_id_from_string


DEFAULT_CHECKPOINT = (
    '/home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/'
    'models/baseline/clip/full/mse_contrastive/val_rdm_mse/'
    'encoder/sub-01/10-10_07-51/best.pth'
)


def load_clip(path, expected_rows):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'CLIP feature file not found: {path}')
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict) or 'img_features' not in obj:
        raise ValueError(f'{path} must contain the img_features key')
    x = torch.as_tensor(obj['img_features']).detach().float().cpu().numpy().astype(np.float64)
    if x.shape != (expected_rows, 1024) or not np.isfinite(x).all():
        raise ValueError(f'Expected ({expected_rows},1024) finite CLIP features, got {x.shape}')
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data-path', default='/home/moepy/ozakitakuma/data_eeg')
    parser.add_argument('--img-dir-test', default='/home/moepy/ozakitakuma/data_image/test_images')
    parser.add_argument('--train-features', default='../features/ViT-H-14_features_train.pt')
    parser.add_argument('--test-features', default='../features/ViT-H-14_features_test.pt')
    parser.add_argument('--subject', default='sub-01')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='outputs/pc_mse_variance_baseline_verified')
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    print(f'CHECKPOINT: {ckpt}', flush=True)
    print(f'SUBJECT: {args.subject}', flush=True)
    print(f'DEVICE: {device}', flush=True)
    print('Loading original (non-L2-normalized) CLIP features ...', flush=True)
    train = load_clip(args.train_features, 16540)
    target = load_clip(args.test_features, 200)

    # Load test EEG with identical preprocessing/ordering to the project's evaluate.py.
    # For test data, EEGDataset averages the 80 EEG trials per test image.
    test_set = EEGDataset(
        args.data_path,
        img_dir_test=args.img_dir_test,
        features_dir=str(Path(args.train_features).resolve().parent),
        subjects=[args.subject],
        train=False,
        feature_space='clip',
    )
    if len(test_set) != 200:
        raise ValueError(f'Expected 200 test EEG samples, got {len(test_set)}')
    dataset_target = torch.as_tensor(test_set.img_features).float().cpu().numpy()
    if dataset_target.shape != (200, 1024):
        raise ValueError(f'Dataset test CLIP features have shape {dataset_target.shape}')
    if not np.allclose(dataset_target, target, atol=1e-5, rtol=1e-5):
        raise ValueError('Test CLIP cache in EEGDataset disagrees with --test-features. Check paths/order.')

    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ATMS(outputs_dim=1024)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    subject_id = extract_id_from_string(args.subject)
    predictions, paired_targets = [], []
    print('Inferring from the specified checkpoint (no saved predictions used) ...', flush=True)
    with torch.inference_mode():
        for eeg, _, _, _, _, img_feat in loader:
            eeg = eeg.to(device)
            ids = torch.full((eeg.size(0),), subject_id, dtype=torch.long, device=device)
            predictions.append(model(eeg, ids).float().cpu())
            paired_targets.append(img_feat.float().cpu())
    pred = torch.cat(predictions).numpy().astype(np.float64)
    paired = torch.cat(paired_targets).numpy().astype(np.float64)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError(f'Prediction shape/finite mismatch: {pred.shape}')
    if not np.allclose(paired, target, atol=1e-5, rtol=1e-5):
        raise ValueError('DataLoader order does not align with test CLIP cache!')

    # PCA axes from TRAIN CLIP only, use the same train mean for both test vectors.
    print('Computing 1024 train-CLIP principal directions ...', flush=True)
    train_mean = train.mean(axis=0)
    centered = train - train_mean
    gram = (centered.T @ centered) / (train.shape[0] - 1)
    eigvals, components = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    components = components[:, order]
    test_pc = (target - train_mean) @ components
    pred_pc = (pred - train_mean) @ components

    # Use ddof=0 consistently with MSE's averaging over 200 test samples.
    test_variance = np.var(test_pc, axis=0, ddof=0)
    raw_mse = np.mean((pred_pc - test_pc) ** 2, axis=0)
    # Avoid manufacturing huge values when variance is exactly zero.
    nmse = np.divide(raw_mse, test_variance, out=np.full(1024, np.nan), where=test_variance > 0)

    # Pearson correlation per PC across 200 paired test images.
    # Center each series independently; unlike MSE, Pearson ignores mean/scale offsets.
    t_centered = test_pc - test_pc.mean(axis=0, keepdims=True)
    p_centered = pred_pc - pred_pc.mean(axis=0, keepdims=True)
    numerator = np.sum(t_centered * p_centered, axis=0)
    denominator = np.sqrt(
        np.sum(t_centered ** 2, axis=0)
        * np.sum(p_centered ** 2, axis=0)
    )
    pearson = np.divide(
        numerator, denominator,
        out=np.full(1024, np.nan, dtype=np.float64),
        where=denominator > 1e-15,
    )
    pearson = np.clip(pearson, -1.0, 1.0)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'pc_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pc_rank', 'test_target_variance', 'raw_mse', 'normalized_mse', 'pearson_r'])
        for j in range(1024):
            writer.writerow([j + 1, test_variance[j], raw_mse[j], nmse[j], pearson[j]])

    fig, ax = plt.subplots(figsize=(15, 6.2))
    xs = np.arange(1, 1025)
    ax.plot(xs, test_variance, label='Test target variance', linewidth=1.2)
    ax.plot(xs, raw_mse, label='Raw MSE', linewidth=1.2)
    ax.plot(xs, nmse, label='Normalized MSE (MSE / variance)', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('Principal component rank (train CLIP PCA)')
    ax.set_ylabel('Value (log scale)')
    ax.set_title(f'EEG prediction error and target variance by CLIP PC ({args.subject}, baseline best.pth)')
    ax.grid(alpha=0.15)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'pc_mse_variance.png', dpi=200)
    plt.close(fig)

    # Dedicated Pearson plot for reading correlation on its native [-1, 1] scale.
    fig, ax = plt.subplots(figsize=(15, 5.0))
    ax.plot(xs, pearson, color='tab:purple', linewidth=1.1, label='Pearson r (EEG vs CLIP)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlim(1, 1024)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('Principal component rank (train CLIP PCA)')
    ax.set_ylabel('Pearson correlation r')
    ax.set_title(f'EEG vs CLIP Pearson correlation by PC ({args.subject}, baseline best.pth)')
    ax.grid(alpha=0.15)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'pc_pearson.png', dpi=200)
    plt.close(fig)

    # Two aligned panels for immediate comparison with variance, MSE and NMSE.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2]})
    ax1.plot(xs, test_variance, label='Test target variance', linewidth=1.1)
    ax1.plot(xs, raw_mse, label='Raw MSE', linewidth=1.1)
    ax1.plot(xs, nmse, label='NMSE (MSE / variance)', linewidth=1.1)
    ax1.set_yscale('log')
    ax1.set_ylabel('Variance / MSE / NMSE (log scale)')
    ax1.grid(alpha=0.15)
    ax1.legend(loc='upper left')
    ax2.plot(xs, pearson, color='tab:purple', linewidth=1.1,
             label='Pearson r (EEG vs CLIP)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel('Pearson correlation r')
    ax2.set_xlabel('Principal component rank (train CLIP PCA)')
    ax2.grid(alpha=0.15)
    ax2.legend(loc='upper right')
    fig.suptitle(f'CLIP PC prediction error and correlation ({args.subject}, baseline best.pth)')
    fig.tight_layout()
    fig.savefig(out / 'pc_mse_variance_with_pearson.png', dpi=200)
    plt.close(fig)

    with (out / 'provenance.txt').open('w') as f:
        f.write(f'checkpoint={ckpt}\nsubject={args.subject}\n')
        f.write(f'train_features={Path(args.train_features).resolve()}\n')
        f.write(f'test_features={Path(args.test_features).resolve()}\n')
        f.write(f'data_path={Path(args.data_path).resolve()}\n')
        f.write('train PCA=raw CLIP, centered; test EEG=80 trials averaged by EEGDataset\n')
        f.write('test variance ddof=0; NMSE=raw MSE/test variance\n')
        f.write('Pearson r=correlation across 200 paired test images for each train PCA direction\n')
    print('Prediction shape:', pred.shape)
    print('Saved:', out / 'pc_mse_variance.png')
    print('Saved:', out / 'pc_pearson.png')
    print('Saved:', out / 'pc_mse_variance_with_pearson.png')
    print('Saved:', out / 'pc_metrics.csv')
    print('Saved:', out / 'provenance.txt')


if __name__ == '__main__':
    main()
