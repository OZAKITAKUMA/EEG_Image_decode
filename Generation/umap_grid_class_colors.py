#!/usr/bin/env python3
"""Plot spatially tiled UMAP pages with globally consistent per-class colors.

CLIP features: one training image is one row, and 10 consecutive images belong
 to one THINGS class. A *single* UMAP fit uses all 16,540 images (1,654 classes).
The 2-D embedding is then split spatially into 10 columns x 5 rows.

No labels are passed to UMAP, and every page uses the SAME global coordinates.
The palette assignment considers all pages simultaneously: classes that occur
on the same page are assigned perceptually separated colors as far as possible.

Run from Generation/:
    python umap_grid_class_colors.py

Reuse coordinates from an earlier UMAP run (N, 2, .npy or .npz):
    python umap_grid_class_colors.py --embedding-path PATH_TO_EMBEDDING.npy

Dependency for fitting: pip install umap-learn
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FEATURES = HERE.parent / "features" / "ViT-H-14_features_train.pt"
DEFAULT_OUTPUT = HERE / "outputs" / "umap_spatial_tiles"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES,
                        help="CLIP cache (.pt, dict['img_features'], N x 1024).")
    parser.add_argument("--embedding-path", type=Path, default=None,
                        help="Reuse existing (N,2) UMAP coordinates, .npy or .npz. "
                             "If set, CLIP features and umap-learn are not required.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Optional train image root; sorted class folder names "
                             "are included in CSV files.")
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--point-size", type=float, default=14.0)
    parser.add_argument("--refit", action="store_true",
                        help="Refit UMAP even if a matching embedding cache exists.")
    return parser.parse_args()


def load_existing_embedding(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"UMAP embedding not found: {path}")
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            for key in ("embedding", "embedding_xy", "umap_embedding",
                        "umap_coords", "coords", "xy"):
                if key in saved:
                    array = saved[key]
                    break
            else:
                if len(saved.files) == 1:
                    array = saved[saved.files[0]]
                else:
                    raise ValueError(
                        f"Cannot identify UMAP coordinates in {path}. "
                        f"Available keys: {saved.files}"
                    )
    else:
        raise ValueError("--embedding-path must be a .npy or .npz file")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        raise ValueError(f"Expected UMAP coordinates (N,2), received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("UMAP coordinates contain NaN or infinity")
    return array


def embedding_for_features(args: argparse.Namespace) -> np.ndarray:
    cache_path = args.output_dir / "embedding_xy.npy"
    metadata_path = args.output_dir / "embedding_metadata.json"
    if not args.features_path.is_file():
        raise FileNotFoundError(f"CLIP feature file not found: {args.features_path}")

    stat = args.features_path.stat()
    requested = dict(
        features_path=str(args.features_path.resolve()),
        features_file_bytes=stat.st_size,
        features_file_mtime_ns=stat.st_mtime_ns,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    if not args.refit and cache_path.is_file() and metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if all(previous.get(key) == value for key, value in requested.items()):
            print(f"[UMAP] Reusing existing full-data coordinates: {cache_path}", flush=True)
            return load_existing_embedding(cache_path)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load the feature cache") from exc
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn: pip install umap-learn") from exc

    print(f"[CLIP] Loading {args.features_path}", flush=True)
    cache = torch.load(args.features_path, map_location="cpu", weights_only=False)
    if isinstance(cache, dict):
        if "img_features" not in cache:
            raise KeyError("Feature cache dict must contain 'img_features'")
        features = cache["img_features"]
    else:
        features = cache
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError(f"Expected CLIP features (N,1024), received {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinity")

    print(f"[UMAP] Fitting ONCE on all {len(features):,} images, "
          f"metric={args.metric}, seed={args.seed} ...", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        low_memory=True,
    )
    coords = np.asarray(reducer.fit_transform(features), dtype=np.float32)
    del features, reducer, cache
    np.save(cache_path, coords)
    requested["num_images"] = int(len(coords))
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(requested, handle, indent=2, ensure_ascii=False)
    print(f"[UMAP] Saved reusable coordinates: {cache_path}", flush=True)
    return coords


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB -> CIELAB (D65); used for perceptual color spacing."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3 * delta**2) + 4 / 29,
    )
    return np.stack([
        116 * transformed[:, 1] - 16,
        500 * (transformed[:, 0] - transformed[:, 1]),
        200 * (transformed[:, 1] - transformed[:, 2]),
    ], axis=1)


def generate_palette() -> tuple[np.ndarray, np.ndarray]:
    """Many reusable saturated colors, ordered by Lab farthest-point spacing."""
    hue_count = 60
    saturation_value = (
        (0.90, 0.88),
        (0.80, 0.66),
        (0.63, 0.97),
        (0.99, 0.63),
        (0.65, 0.81),
        (0.94, 0.98),
        (0.80, 0.75),
        (0.70, 0.92),
    )
    pool = np.array([
        colorsys.hsv_to_rgb(h / hue_count, saturation, value)
        for saturation, value in saturation_value
        for h in range(hue_count)
    ])
    lab_pool = rgb_to_lab(pool)
    chosen = []
    available = np.ones(len(pool), dtype=bool)
    min_distance_sq = np.full(len(pool), np.inf)
    # Start with a legible saturated color, then spread choices through Lab.
    start = int(np.argmax(np.linalg.norm(lab_pool[:, 1:], axis=1)))
    for _ in range(len(pool)):
        choice = start if not chosen else int(np.argmax(np.where(
            available, min_distance_sq, -1.0,
        )))
        chosen.append(choice)
        available[choice] = False
        distances_sq = np.sum((lab_pool - lab_pool[choice])**2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distances_sq)
    return pool[chosen], lab_pool[chosen]


def spatial_tiles(coords: np.ndarray, rows: int, cols: int):
    eps_x = max(float(np.ptp(coords[:, 0])) * 1e-5, 1e-5)
    eps_y = max(float(np.ptp(coords[:, 1])) * 1e-5, 1e-5)
    x_edges = np.linspace(coords[:, 0].min() - eps_x,
                          coords[:, 0].max() + eps_x, cols + 1)
    y_edges = np.linspace(coords[:, 1].min() - eps_y,
                          coords[:, 1].max() + eps_y, rows + 1)
    # Row 0 is top row, col 0 is leftmost column: page 01 is top-left.
    col_idx = np.clip(np.searchsorted(x_edges, coords[:, 0], side="right") - 1,
                      0, cols - 1)
    ascending_row = np.clip(np.searchsorted(
        y_edges, coords[:, 1], side="right") - 1, 0, rows - 1)
    row_idx = rows - 1 - ascending_row
    tile_ids = row_idx * cols + col_idx
    return tile_ids.astype(np.int32), x_edges, y_edges


def choose_global_colors(labels: np.ndarray, tile_ids: np.ndarray,
                         n_classes: int, n_tiles: int):
    """Maximize minimum CIELAB distance to already colored co-page classes.

    One fixed RGB color is assigned to each class across ALL tiles.
    Co-occurring classes are considered conflicting even if in far-apart parts
    of their tile. When too many classes share a tile, perfect distinction is
    mathematically impossible; optimize minimum perceptual distance instead.
    """
    presence = np.zeros((n_classes, n_tiles), dtype=np.uint8)
    presence[labels, tile_ids] = 1
    cooccur = (presence.astype(np.int16) @ presence.T.astype(np.int16)) > 0
    np.fill_diagonal(cooccur, False)
    degree = cooccur.sum(axis=1)
    rgb_palette, lab_palette = generate_palette()
    assigned = np.full(n_classes, -1, dtype=np.int32)

    # Color the most constrained classes first, with stable class-ID tie break.
    order = np.lexsort((np.arange(n_classes), -degree))
    distances = np.sum(
        (lab_palette[:, None, :] - lab_palette[None, :, :])**2, axis=2
    )
    for k, class_id in enumerate(order):
        used = np.unique(assigned[cooccur[class_id]])
        used = used[used >= 0]
        if len(used):
            # A palette candidate's worst (closest) co-page color.
            min_distance = np.min(distances[:, used], axis=1)
            assigned[class_id] = int(np.argmax(min_distance))
        else:
            # Prefer different starting colors for independent classes.
            assigned[class_id] = int(k % len(rgb_palette))
    return rgb_palette[assigned], presence


def class_names(image_dir: Path | None, expected: int) -> list[str]:
    if image_dir is None:
        return [f"class_{i:04d}" for i in range(expected)]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"--image-dir not found: {image_dir}")
    folders = sorted(p.name for p in image_dir.iterdir() if p.is_dir())
    if len(folders) != expected:
        raise ValueError(
            f"Image folder count ({len(folders)}) != inferred classes ({expected}). "
            "Feature-row order must match sorted class folders."
        )
    return folders


def write_csv(path: Path, columns: list[str], rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def plot_overview(args, coords, tile_ids, x_edges, y_edges, labels):
    fig, ax = plt.subplots(figsize=(15, 11), constrained_layout=True)
    ax.scatter(coords[:, 0], coords[:, 1], s=2.5,
               c="#73869c", alpha=0.35, linewidths=0)
    for x in x_edges:
        ax.axvline(x, color="#444444", linewidth=0.6, alpha=0.65)
    for y in y_edges:
        ax.axhline(y, color="#444444", linewidth=0.6, alpha=0.65)
    for r in range(args.rows):
        for c in range(args.cols):
            tile = r * args.cols + c
            x = 0.5 * (x_edges[c] + x_edges[c + 1])
            y = 0.5 * (y_edges[args.rows - 1 - r] +
                       y_edges[args.rows - r])
            ids = tile_ids == tile
            ax.text(x, y, f"{tile + 1:02d}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#273442",
                    bbox=dict(facecolor="white", alpha=0.70,
                              boxstyle="round,pad=0.20", edgecolor="none"))
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Full-data UMAP / spatial {args.rows}x{args.cols} tiles "
                 f"({len(coords):,} images, {len(np.unique(labels)):,} classes)")
    ax.set_aspect("equal", adjustable="box")
    path = args.output_dir / "00_all_tiles_index.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"[Saved] {path}", flush=True)


def plot_page(args, page_number, coords, labels, tile_ids, x_edges, y_edges,
              class_rgb, names, tile_dir):
    tile_index = page_number - 1
    row = tile_index // args.cols
    col = tile_index % args.cols
    ids = np.flatnonzero(tile_ids == tile_index)
    unique, counts = np.unique(labels[ids], return_counts=True)
    y_low_idx = args.rows - 1 - row

    fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)
    if len(ids):
        ax.scatter(coords[ids, 0], coords[ids, 1],
                   c=class_rgb[labels[ids]], s=args.point_size,
                   alpha=0.88, edgecolors="none", rasterized=True)
    else:
        ax.text(0.5, 0.5, "No images in this tile", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#777777")
    ax.set_xlim(x_edges[col], x_edges[col + 1])
    ax.set_ylim(y_edges[y_low_idx], y_edges[y_low_idx + 1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.set_axisbelow(True)
    ax.set_xlabel("UMAP 1 (same full-data embedding)")
    ax.set_ylabel("UMAP 2 (same full-data embedding)")
    ax.set_title(f"Tile {page_number:02d} / {args.rows * args.cols:02d}  "
                 f"(top row {row + 1}, left col {col + 1})\n"
                 f"{len(ids)} images / {len(unique)} classes", fontsize=13)
    output = tile_dir / f"tile_{page_number:02d}.png"
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)

    # Counts by class in this tile; class IDs and colors are global.
    counts_csv = tile_dir / f"tile_{page_number:02d}_classes.csv"
    count_rows = [
        (int(cl), names[cl], int(count), to_hex(class_rgb[cl]),
         ";".join(str(t + 1) for t in np.unique(tile_ids[labels == cl])
                  if t != tile_index))
        for cl, count in zip(unique, counts)
    ]
    count_rows.sort(key=lambda row_: (-row_[2], row_[0]))
    write_csv(counts_csv,
              ["class_index_0based", "class_name", "images_in_this_tile",
               "hex_color", "other_tile_pages"], count_rows)
    print(f"[Tile {page_number:02d}/{args.rows * args.cols:02d}] "
          f"{len(ids):5d} images / {len(unique):4d} classes", flush=True)
    return len(ids), len(unique)


def main():
    args = arguments()
    if args.cols <= 0 or args.rows <= 0:
        raise ValueError("--rows and --cols must both be positive")
    if args.images_per_class <= 0:
        raise ValueError("--images-per-class must be positive")
    if args.point_size <= 0 or args.dpi <= 0:
        raise ValueError("--point-size and --dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.embedding_path is not None:
        coords = load_existing_embedding(args.embedding_path)
        print(f"[UMAP] Loaded existing coordinates: {args.embedding_path}", flush=True)
        np.save(args.output_dir / "embedding_xy.npy", coords)
        # The old feature-fitted cache metadata must not validate external XY.
        (args.output_dir / "embedding_metadata.json").unlink(missing_ok=True)
    else:
        coords = embedding_for_features(args)

    if len(coords) % args.images_per_class:
        raise ValueError(
            f"Number of images ({len(coords)}) is not divisible by "
            f"images per class ({args.images_per_class})"
        )
    n_classes = len(coords) // args.images_per_class
    labels = np.arange(len(coords), dtype=np.int32) // args.images_per_class
    names = class_names(args.image_dir, n_classes)
    tile_ids, x_edges, y_edges = spatial_tiles(coords, args.rows, args.cols)
    rgb_by_class, _ = choose_global_colors(
        labels, tile_ids, n_classes, args.rows * args.cols,
    )

    write_csv(
        args.output_dir / "class_colors.csv",
        ["class_index_0based", "class_name", "hex_color"],
        ((cl, name, to_hex(rgb_by_class[cl])) for cl, name in enumerate(names)),
    )
    # Point-to-page mapping also records the 2-D coordinates for reuse.
    write_csv(
        args.output_dir / "image_coordinates_and_tiles.csv",
        ["image_index_0based", "class_index_0based", "umap_x", "umap_y",
         "page_1based", "hex_color"],
        ((i, int(labels[i]), float(coords[i, 0]), float(coords[i, 1]),
          int(tile_ids[i] + 1), to_hex(rgb_by_class[labels[i]]))
         for i in range(len(coords))),
    )

    plot_overview(args, coords, tile_ids, x_edges, y_edges, labels)
    tiles_dir = args.output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    counts = [
        plot_page(args, p, coords, labels, tile_ids, x_edges, y_edges,
                  rgb_by_class, names, tiles_dir)
        for p in range(1, args.rows * args.cols + 1)
    ]
    write_csv(
        args.output_dir / "tile_summary.csv",
        ["page_1based", "images", "classes"],
        ((i + 1, image_count, class_count)
         for i, (image_count, class_count) in enumerate(counts)),
    )
    print(f"\n[Finished] Overview + {args.rows * args.cols} color-coded tiles")
    print(f"  Folder: {args.output_dir.resolve()}")
    print("  Note: With hundreds of classes on a page, no palette can make "
          "all colors unmistakably different.")


if __name__ == "__main__":
    main()
