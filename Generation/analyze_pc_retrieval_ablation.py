#!/usr/bin/env python3
"""Evaluate which CLIP PCA directions matter for EEG-to-image retrieval.

Loads the specified ATMS checkpoint, infers 200 sub-01 test EEG signals,
computes PCA on 16,540 TRAIN CLIP features, then runs retrieval ablations.
Does not retrain, use SDXL/prior/adapter, or modify GitHub.

Note: all sweeps using test data are exploratory, not unbiased model-selection.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eegdatasets import EEGDataset
from models.atms import ATMS, extract_id_from_string

DEFAULT_CHECKPOINT = (
    '/home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/'
    'models/baseline/clip/full/mse_contrastive/val_rdm_mse/'
    'encoder/sub-01/10-10_07-51/best.pth'
)


def load_clip(path, rows):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    obj = torch.load(p, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict) or 'img_features' not in obj:
        raise ValueError(f'Missing img_features in {p}')
    x = torch.as_tensor(obj['img_features']).detach().cpu().numpy().astype(np.float64)
    if x.shape != (rows, 1024) or not np.isfinite(x).all():
        raise ValueError(f'Expected ({rows},1024) finite features, got {x.shape}')
    return x


def retrieval(pred, target):
    """Return paired 200-way Top-1/Top-5 from original cosine retrieval."""
    pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    targ_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    similarities = pred_norm @ targ_norm.T
    ranks = np.argsort(-similarities, axis=1)
    true = np.arange(pred.shape[0])
    return float(np.mean(ranks[:, 0] == true)), float(np.mean(np.any(ranks[:, :5] == true[:, None], axis=1)))


def plot_pair(path, x, top1, top5, xlabel, title, labels=None, baseline=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, np.asarray(top1)*100, 'o-', label='Top-1 (%)')
    ax.plot(x, np.asarray(top5)*100, 's-', label='Top-5 (%)')
    if baseline is not None:
        ax.axhline(baseline[0]*100, linestyle='--', color='C0', alpha=.45, label='Baseline Top-1')
        ax.axhline(baseline[1]*100, linestyle='--', color='C1', alpha=.45, label='Baseline Top-5')
    ax.set_xlabel(xlabel)
    if labels is not None:
        ax.set_xticks(x, labels=labels, rotation=25, ha='right')
    ax.set_ylabel('200-way retrieval accuracy (%)')
    ax.set_title(title)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--data-path', default='/home/moepy/ozakitakuma/data_eeg')
    p.add_argument('--img-dir-test', default='/home/moepy/ozakitakuma/data_image/test_images')
    p.add_argument('--train-features', default='../features/ViT-H-14_features_train.pt')
    p.add_argument('--test-features', default='../features/ViT-H-14_features_test.pt')
    p.add_argument('--subject', default='sub-01')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='outputs/pc_retrieval_ablation')
    p.add_argument('--cutoff', type=int, default=630, help='PC rank where attenuation begins after this rank')
    args = p.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {ckpt}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if not 1 <= args.cutoff < 1024:
        raise ValueError('--cutoff must be in [1, 1023]')
    print('Checkpoint:', ckpt, flush=True)
    print('Subject:', args.subject, flush=True)
    train = load_clip(args.train_features, 16540)
    target = load_clip(args.test_features, 200)

    dataset = EEGDataset(
        args.data_path, img_dir_test=args.img_dir_test,
        features_dir=str(Path(args.test_features).resolve().parent),
        subjects=[args.subject], train=False, feature_space='clip',
    )
    if len(dataset) != 200 or not np.allclose(dataset.img_features.float().cpu().numpy(), target, atol=1e-5, rtol=1e-5):
        raise ValueError('EEG and CLIP test image feature order/cache mismatch')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ATMS(outputs_dim=1024)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False), strict=True)
    model = model.to(device).eval()
    subj_id = extract_id_from_string(args.subject)
    predicted, paired = [], []
    print('Inferring EEG with specified baseline checkpoint ...', flush=True)
    with torch.inference_mode():
        for eeg, _, _, _, _, image_features in loader:
            eeg = eeg.to(device)
            subject_ids = torch.full((eeg.shape[0],), subj_id, device=device, dtype=torch.long)
            predicted.append(model(eeg, subject_ids).float().cpu().numpy().astype(np.float64))
            paired.append(image_features.float().cpu().numpy().astype(np.float64))
    pred = np.concatenate(predicted, axis=0)
    if pred.shape != (200, 1024) or not np.allclose(np.concatenate(paired), target, atol=1e-5, rtol=1e-5):
        raise ValueError('Inference/image pairing mismatch')
    baseline = retrieval(pred, target)
    print('Baseline Top-1/Top-5:', baseline, flush=True)

    # Same raw CLIP feature space as the original baseline; PCA only defines directions.
    mu = train.mean(axis=0)
    centered_train = train - mu
    gram = centered_train.T @ centered_train / (train.shape[0]-1)
    eigvals, eigvecs = np.linalg.eigh(gram)
    indices = np.argsort(eigvals)[::-1]
    V = eigvecs[:, indices]  # [1024,1024]
    pred_pc = (pred-mu) @ V
    target_pc = (target-mu) @ V

    def reconstruct(pred_scores, target_scores):
        return mu + pred_scores @ V.T, mu + target_scores @ V.T

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    def record(name, detail, val):
        results.append((name, detail, *val))
        print(f'{name:24s} {str(detail):>14s}: Top1={val[0]*100:5.1f}%  Top5={val[1]*100:5.1f}%', flush=True)

    record('baseline', 'original', baseline)

    # Experiment 1: keep first k PCs (both EEG and CLIP); reconstruct WITH training mean.
    ks = [16, 32, 64, 128, 256, 384, 512, 630, 768, 896, 1024]
    keep1, keep5 = [], []
    for k in ks:
        sp = np.zeros_like(pred_pc)
        st = np.zeros_like(target_pc)
        sp[:, :k] = pred_pc[:, :k]
        st[:, :k] = target_pc[:, :k]
        accuracy = retrieval(*reconstruct(sp, st))
        keep1.append(accuracy[0]); keep5.append(accuracy[1])
        record('keep_top_k', k, accuracy)
    plot_pair(out/'01_keep_top_k.png', ks, keep1, keep5,
              'Number of retained leading PCs', 'Retrieval when retaining only leading CLIP PCs', baseline=baseline)

    # Experiment 2: remove exactly one PC band from both EEG & CLIP before retrieval.
    bands = [(1,64),(65,128),(129,256),(257,512),(513,630),(631,768),(769,1024)]
    drop1, drop5, only1, only5, names = [], [], [], [], []
    for start, end in bands:
        names.append(f'{start}-{end}')
        sp = pred_pc.copy(); st = target_pc.copy()
        sp[:, start-1:end] = 0
        st[:, start-1:end] = 0
        drop_accuracy = retrieval(*reconstruct(sp, st))
        drop1.append(drop_accuracy[0]); drop5.append(drop_accuracy[1])
        record('remove_band', names[-1], drop_accuracy)
        # Only a band's 2-D-to-256-D PCA coordinates, no common mean included.
        only_accuracy = retrieval(pred_pc[:, start-1:end], target_pc[:, start-1:end])
        only1.append(only_accuracy[0]); only5.append(only_accuracy[1])
        record('only_band', names[-1], only_accuracy)
    x_positions = np.arange(len(bands))
    plot_pair(out/'02_remove_pc_band.png', x_positions, drop1, drop5,
              'Removed principal component band (both EEG & CLIP)',
              'Retrieval after removing each PC band', labels=names, baseline=baseline)
    plot_pair(out/'03_only_pc_band.png', x_positions, only1, only5,
              'Retained principal component band (PC scores only)',
              'Retrieval from each PC band alone', labels=names)

    # Experiment 3: attenuate PC>cutoff in EEG prediction ONLY; targets unchanged.
    # alpha=1 must recover baseline. On TEST data this is exploration, not model selection.
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    att1, att5 = [], []
    for alpha in alphas:
        sp = pred_pc.copy()
        sp[:, args.cutoff:] *= alpha
        new_pred = mu + sp @ V.T
        accuracy = retrieval(new_pred, target)
        att1.append(accuracy[0]); att5.append(accuracy[1])
        record('pred_tail_scale', alpha, accuracy)
    plot_pair(out/'04_pred_tail_attenuation.png', alphas, att1, att5,
              f'EEG prediction scaling of PCs {args.cutoff+1}-1024 (alpha)',
              'Effect of scaling lower-PC EEG prediction only', baseline=baseline)

    if abs(keep1[-1] - baseline[0]) > 1e-10 or abs(keep5[-1] - baseline[1]) > 1e-10:
        raise RuntimeError('PCA full-rank projection changed baseline retrieval; inspect computations')
    if abs(att1[alphas.index(1.0)] - baseline[0]) > 1e-10 or abs(att5[alphas.index(1.0)] - baseline[1]) > 1e-10:
        raise RuntimeError('Alpha=1 did not reproduce baseline')

    with (out/'retrieval_results.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['experiment','parameter','top1','top5'])
        w.writerows(results)
    with (out/'provenance.txt').open('w') as f:
        f.write(f'checkpoint={ckpt}\nsubject={args.subject}\n')
        f.write(f'features_train={Path(args.train_features).resolve()}\n')
        f.write(f'features_test={Path(args.test_features).resolve()}\n')
        f.write(f'cutoff={args.cutoff}\n')
        f.write('All feature projections use PCA axes from training CLIP only.\n')
        f.write('PC removal and keep_top_k reconstruct with training mean; only_band uses PC scores.\n')
        f.write('Exploratory TEST-set sweep. Do not claim a selected configuration as an unbiased test result.\n')
    print('SAVED:', out.resolve(), flush=True)
    for file in sorted(out.glob('*.png')):
        print('  ', file.name, flush=True)


if __name__ == '__main__':
    main()
