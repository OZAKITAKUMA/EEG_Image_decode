#!/usr/bin/env python3
"""Fit UMAP on all 16,540 CLIP training features; visualize all and 12 neighbors."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

N_CLASSES = 1654
N_PER_CLASS = 10


def main():
    parser = argparse.ArgumentParser(description='Full CLIP feature UMAP and class highlights')
    parser.add_argument('--features-path', default='../features/ViT-H-14_features_train.pt')
    parser.add_argument('--output-dir', default='./outputs/clip_umap_all')
    parser.add_argument('--anchor-class', type=int, default=0)
    parser.add_argument('--n-classes', type=int, default=12)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recompute', action='store_true', help='Recalculate UMAP instead of using saved coordinates')
    args = parser.parse_args()

    if not (0 <= args.anchor_class < N_CLASSES):
        parser.error(f'--anchor-class must be in [0, {N_CLASSES-1}]')
    if not (2 <= args.n_classes <= N_CLASSES):
        parser.error(f'--n-classes must be in [2, {N_CLASSES}]')

    cache = torch.load(args.features_path, map_location='cpu', weights_only=False)
    if not isinstance(cache, dict) or 'img_features' not in cache:
        raise ValueError('Expected a dict with img_features')
    x = cache['img_features'].detach().float().cpu().numpy()
    if x.shape != (N_CLASSES * N_PER_CLASS, 1024):
        raise ValueError(f'Expected shape (16540, 1024), got {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError('Features contain NaN/Inf')

    # Choose 12 classes by cosine similarity between their original 1024-D means.
    class_means = x.reshape(N_CLASSES, N_PER_CLASS, 1024).mean(axis=1)
    mean_norms = np.maximum(np.linalg.norm(class_means, axis=1, keepdims=True), 1e-12)
    mean_unit = class_means / mean_norms
    similarity = mean_unit @ mean_unit[args.anchor_class]
    chosen = np.argsort(-similarity)[:args.n_classes]

    # Reuse existing UMAP coordinates whenever possible.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    coord_path = out / 'umap_all_coordinates.npy'
    if coord_path.exists() and not args.recompute:
        embedding = np.load(coord_path)
        if embedding.shape != (N_CLASSES * N_PER_CLASS, 2):
            raise ValueError(f'Unexpected saved coordinates shape: {embedding.shape}')
        print(f'Using saved UMAP coordinates: {coord_path}', flush=True)
    else:
        try:
            from umap import UMAP
        except ImportError as exc:
            raise SystemExit('umap-learn is required: pip install umap-learn') from exc
        x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        print(f'Fitting UMAP on all {x_norm.shape[0]} images ...', flush=True)
        embedding = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                         n_components=2, metric='cosine', random_state=args.seed).fit_transform(x_norm)
        np.save(coord_path, embedding)
    if not np.isfinite(embedding).all():
        raise RuntimeError('UMAP produced non-finite coordinates')
    coords = embedding.reshape(N_CLASSES, N_PER_CLASS, 2)

    def draw_class(ax, class_id, color, label, size=40):
        points = coords[class_id]
        ax.scatter(points[:, 0], points[:, 1], s=size, alpha=0.95,
                   color=color, label=label, zorder=4)
        center = points.mean(axis=0)
        cov = np.cov(points.T)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        vectors = vectors[:, order]
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        ellipse = Ellipse(center, width=4*np.sqrt(values[0]),
                          height=4*np.sqrt(values[1]), angle=angle,
                          facecolor='none', edgecolor=color,
                          linewidth=2.2, linestyle='--', zorder=5)
        ax.add_patch(ellipse)
        ax.scatter(*center, color=color, marker='x', s=85, linewidths=2, zorder=6)
        return points

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.30,
               color='slategray', linewidths=0, rasterized=True)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
           title='All 1,654 CLIP classes / 16,540 images')
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(out / 'umap_all_classes.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.13,
               color='gray', label='All 16,540 images', linewidths=0, rasterized=True)
    colors = plt.get_cmap('tab20')
    for i, class_id in enumerate(chosen):
        color = colors(i % 20)
        label = f'class {class_id:04d}' + (' (anchor)' if class_id == args.anchor_class else '')
        points = draw_class(ax, int(class_id), color, label, size=30)
        p = points.mean(axis=0)
        ax.text(p[0], p[1], str(class_id), fontsize=8, weight='bold')
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
           title=f'Full-data UMAP: {args.n_classes} classes nearest to class {args.anchor_class} (1024-D cosine)')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, markerscale=1.3)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(out / 'umap_highlight_12_classes.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Zoom into the anchor and its nearest class in the original 1024-D space.
    neighbor = int(chosen[1])
    selected = np.concatenate([coords[args.anchor_class], coords[neighbor]], axis=0)
    center = selected.mean(axis=0)
    span = np.ptp(selected, axis=0)
    half = np.maximum(span * 0.8, 0.08)
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_class(ax, args.anchor_class, 'tab:blue', f'class {args.anchor_class:04d} (anchor)', size=65)
    draw_class(ax, neighbor, 'tab:orange', f'class {neighbor:04d} (nearest)', size=65)
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
           title=f'Zoom: class {args.anchor_class} vs class {neighbor} (10 images each)')
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / 'umap_zoom_2_classes.png', dpi=220)
    
    # ============================================================
    # Compare original UMAP and UMAP using only PCA components 1, 2
    # ============================================================
    from sklearn.decomposition import PCA
    from umap import UMAP

    # Normalize the original 1024-D CLIP features.
    x_norm = x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True), 1e-12
    )

    # PCA: 1024 dimensions -> 2 dimensions (PC1 and PC2).
    pca = PCA(n_components=2, svd_solver='randomized', random_state=args.seed)
    x_pca2 = pca.fit_transform(x_norm)

    print(
        'PCA explained variance ratios:',
        pca.explained_variance_ratio_.tolist(),
        flush=True
    )

    # UMAP on PCA-reduced 2-D features.
    pca_umap_path = out / 'umap_pca2_coordinates.npy'

    if pca_umap_path.exists() and not args.recompute:
        embedding_pca2 = np.load(pca_umap_path)
        if embedding_pca2.shape != (N_CLASSES * N_PER_CLASS, 2):
            raise ValueError(
                f'Unexpected PCA-UMAP coordinates: {embedding_pca2.shape}'
            )
        print('Using saved PCA2-UMAP coordinates', flush=True)
    else:
        print('Fitting UMAP on PCA1 and PCA2...', flush=True)

        embedding_pca2 = UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            metric='euclidean',
            random_state=args.seed
        ).fit_transform(x_pca2)

        np.save(pca_umap_path, embedding_pca2)

    # Side-by-side visualization.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    titles = [
        'UMAP: Original 1024-D CLIP',
        'UMAP: PCA1 + PCA2 only'
    ]

    embeddings = [embedding, embedding_pca2]

    for ax, emb, title in zip(axes, embeddings, titles):

        # All 16,540 images.
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            s=3,
            alpha=0.15,
            color='gray',
            linewidths=0,
            rasterized=True
        )

        # Highlight the same 12 classes.
        for i, class_id in enumerate(chosen):
            indices = np.arange(
                int(class_id) * N_PER_CLASS,
                (int(class_id) + 1) * N_PER_CLASS
            )

            points = emb[indices]

            color = colors(i % 20)

            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=35,
                color=color,
                alpha=0.95,
                label=f'class {class_id:04d}',
                zorder=5
            )

            center = points.mean(axis=0)

            ax.text(
                center[0],
                center[1],
                str(class_id),
                fontsize=8,
                weight='bold'
            )

        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(alpha=0.12)

    axes[1].legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8
    )

    fig.suptitle(
        f'Full CLIP vs PCA(2) UMAP: anchor class {args.anchor_class}',
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        out / 'umap_original_vs_pca2.png',
        dpi=200,
        bbox_inches='tight'
    )

    plt.close(fig)

    print('Saved:', out / 'umap_original_vs_pca2.png')

    plt.close(fig)

    print('Highlighted class IDs:', chosen.tolist())
    print('Cosine similarities to anchor:', np.round(similarity[chosen], 4).tolist())
    print('Saved:', out / 'umap_all_classes.png')
    print('Saved:', out / 'umap_highlight_12_classes.png')
    print('Saved:', out / 'umap_zoom_2_classes.png')
    print('Saved:', out / 'umap_all_coordinates.npy')
    print('Both figures use the exact same full-dataset UMAP coordinates.')


if __name__ == '__main__':
    main()
