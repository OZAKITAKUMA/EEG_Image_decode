#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def first_image(folder: Path):
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"画像が見つかりません: {folder}")
    return files[0]

def class_label_from_gt_folder(folder_name: str):
    # evaluate.py の load_test_texts() と同じ考え方:
    # "00001_banana" -> "banana"
    if "_" in folder_name:
        return folder_name.split("_", 1)[1]
    return folder_name

def generated_image(gen_root: Path, label: str):
    class_dir = gen_root / label
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"生成画像のクラスフォルダが見つかりません:\n"
            f"  label={label}\n"
            f"  expected={class_dir}"
        )

    # 今の実験は num_gen_per_class=1 なので、まず 0.png を優先
    zero_png = class_dir / "0.png"
    if zero_png.is_file():
        return zero_png

    return first_image(class_dir)

def load_cell(path: Path, cell_size: int):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (cell_size, cell_size), "white")
        x = (cell_size - im.width) // 2
        y = (cell_size - im.height) // 2
        canvas.paste(im, (x, y))
        return canvas

def main():
    parser = argparse.ArgumentParser(
        description="GT / BASE / PROPOSED を3段で並べ、50クラスずつ画像化する"
    )
    parser.add_argument("--base_dir", required=True,
                        help="BASE生成画像のクラスフォルダ群が入ったディレクトリ")
    parser.add_argument("--proposed_dir", required=True,
                        help="PROPOSED生成画像のクラスフォルダ群が入ったディレクトリ")
    parser.add_argument(
        "--gt_dir",
        default="/home/moepy/ozakitakuma/data_image/test_images",
        help="GT test_images ディレクトリ"
    )
    parser.add_argument("--output_dir", default="./comparison_grids")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--cell_size", type=int, default=128)
    parser.add_argument("--label_width", type=int, default=110)
    args = parser.parse_args()

    gt_root = Path(args.gt_dir)
    base_root = Path(args.base_dir)
    proposed_root = Path(args.proposed_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for p, name in [
        (gt_root, "GT"),
        (base_root, "BASE"),
        (proposed_root, "PROPOSED"),
    ]:
        if not p.is_dir():
            raise NotADirectoryError(f"{name} directory が存在しません: {p}")

    # GT側は evaluate.py と同じくフォルダ名でsortして順序を決める
    gt_class_dirs = sorted(
        p for p in gt_root.iterdir()
        if p.is_dir()
    )

    if len(gt_class_dirs) == 0:
        raise RuntimeError(f"GTクラスフォルダがありません: {gt_root}")

    print(f"GT classes: {len(gt_class_dirs)}")

    items = []
    for idx, gt_class_dir in enumerate(gt_class_dirs):
        label = class_label_from_gt_folder(gt_class_dir.name)

        gt_img = first_image(gt_class_dir)
        base_img = generated_image(base_root, label)
        proposed_img = generated_image(proposed_root, label)

        items.append((idx, label, gt_img, base_img, proposed_img))

    font = ImageFont.load_default()

    for start in range(0, len(items), args.chunk_size):
        chunk = items[start:start + args.chunk_size]
        n = len(chunk)

        width = args.label_width + n * args.cell_size
        height = 3 * args.cell_size

        grid = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(grid)

        row_names = ["GT", "BASE", "PROPOSED"]
        for row, name in enumerate(row_names):
            y0 = row * args.cell_size
            draw.text(
                (10, y0 + args.cell_size // 2 - 6),
                name,
                fill="black",
                font=font,
            )

        for col, (_, label, gt_img, base_img, proposed_img) in enumerate(chunk):
            x0 = args.label_width + col * args.cell_size

            paths = [gt_img, base_img, proposed_img]
            for row, path in enumerate(paths):
                cell = load_cell(path, args.cell_size)
                y0 = row * args.cell_size
                grid.paste(cell, (x0, y0))

            # 列境界を薄く表示して、上下3枚の対応を追いやすくする
            draw.line(
                [(x0, 0), (x0, height)],
                fill=(220, 220, 220),
                width=1,
            )

        end = start + n
        out_path = out_root / f"compare_{start+1:03d}-{end:03d}.png"
        grid.save(out_path)
        print(f"saved: {out_path}")

    print("\n完了")
    print(f"output: {out_root.resolve()}")

if __name__ == "__main__":
    main()


# #使い方
# python compare_generated_grid.py \ 
# --base_dir "/home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/outputs/baseline/clip/full/mse_contrastive/val_rdm_mse/sub-01/10-10_07-51/generated_imgs_encoder_only/sub-01" \ 
# --proposed_dir "/home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/outputs/baseline/clip/full/mse_contrastive_svd_bottom256_w4/val_rdm_mse/sub-01/10-10_06-04/generated_imgs_encoder_only/sub-01"