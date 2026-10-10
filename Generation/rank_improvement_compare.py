#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

from eegdatasets import EEGDataset, get_image_encoder_feature_dim
from models.atms import ATMS, extract_id_from_string


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_paths_info(result_dir: Path):
    path = result_dir / "paths_info.txt"
    if not path.is_file():
        raise FileNotFoundError(f"paths_info.txt が見つかりません: {path}")

    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    if "encoder_path" not in info:
        raise KeyError(f"encoder_path が paths_info.txt にありません: {path}")

    return info


def extract_features(subject, encoder_path, test_loader, feature_dim, device):
    model = ATMS(outputs_dim=feature_dim)
    state = torch.load(encoder_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    subject_id = extract_id_from_string(subject)
    feats = []

    with torch.no_grad():
        for eeg_data, labels, text, text_features, img, img_features in test_loader:
            eeg_data = eeg_data.to(device)
            batch_size = eeg_data.size(0)
            subject_ids = torch.full(
                (batch_size,),
                subject_id,
                dtype=torch.long,
                device=device,
            )
            out = model(eeg_data, subject_ids)
            feats.append(out.detach().cpu())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(feats, dim=0).float()


def compute_ranks(eeg_features, img_features):
    eeg_norm = F.normalize(eeg_features.float(), dim=1)
    img_norm = F.normalize(img_features.float(), dim=1)

    sim = eeg_norm @ img_norm.T
    order = torch.argsort(sim, dim=1, descending=True)

    n = sim.size(0)
    targets = torch.arange(n)

    # rank: 1始まり
    match = order.eq(targets[:, None])
    ranks = match.float().argmax(dim=1) + 1

    top1 = order[:, 0]
    gt_sim = sim[targets, targets]

    return {
        "similarity": sim,
        "order": order,
        "ranks": ranks,
        "top1": top1,
        "gt_similarity": gt_sim,
    }


def class_name(folder_name: str):
    if "_" in folder_name:
        return folder_name.split("_", 1)[1]
    return folder_name


def first_image(folder: Path):
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"画像が見つかりません: {folder}")
    return files[0]


def load_thumb(path: Path, size: int):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - im.width) // 2
        y = (size - im.height) // 2
        canvas.paste(im, (x, y))
        return canvas


def make_montage(rows, class_dirs, out_path, chunk_start, cell=150, label_h=52):
    # 1ファイル10例: 各列が1サンプル、上から GT / Base top1 / Proposed top1
    n = len(rows)
    left = 95
    width = left + n * cell
    height = label_h + 3 * cell

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    row_labels = ["GT", "BASE top1", "PROPOSED top1"]
    for r, name in enumerate(row_labels):
        draw.text((8, label_h + r * cell + cell // 2 - 6), name, fill="black", font=font)

    for c, row in enumerate(rows):
        x0 = left + c * cell

        gt_idx = int(row["test_index"])
        base_idx = int(row["base_top1_index"])
        prop_idx = int(row["proposed_top1_index"])

        imgs = [
            first_image(class_dirs[gt_idx]),
            first_image(class_dirs[base_idx]),
            first_image(class_dirs[prop_idx]),
        ]

        for r, p in enumerate(imgs):
            thumb = load_thumb(p, cell)
            canvas.paste(thumb, (x0, label_h + r * cell))

        title = (
            f"{row['class_name']}\n"
            f"{row['base_rank']}->{row['proposed_rank']} "
            f"(+{row['rank_improvement']})"
        )
        draw.multiline_text((x0 + 2, 2), title, fill="black", font=font, spacing=2)

        draw.line(
            [(x0, label_h), (x0, height)],
            fill=(220, 220, 220),
            width=1,
        )

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="BASE と PROPOSED の200-way検索順位を比較し、改善ランキングを出す"
    )
    parser.add_argument("--base_result_dir", required=True,
                        help="BASEの timestamp 結果ディレクトリ（paths_info.txt がある場所）")
    parser.add_argument("--proposed_result_dir", required=True,
                        help="PROPOSEDの timestamp 結果ディレクトリ（paths_info.txt がある場所）")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--data_path",
                        default="/home/moepy/ozakitakuma/data_eeg")
    parser.add_argument("--img_dir_training",
                        default="/home/moepy/ozakitakuma/data_image/training_images")
    parser.add_argument("--img_dir_test",
                        default="/home/moepy/ozakitakuma/data_image/test_images")
    parser.add_argument("--output_dir", default="./rank_improvement")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--top_n", type=int, default=30)
    args = parser.parse_args()

    base_result_dir = Path(args.base_result_dir)
    proposed_result_dir = Path(args.proposed_result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_info = read_paths_info(base_result_dir)
    proposed_info = read_paths_info(proposed_result_dir)

    base_encoder = Path(base_info["encoder_path"])
    proposed_encoder = Path(proposed_info["encoder_path"])

    print("BASE encoder    :", base_encoder)
    print("PROPOSED encoder:", proposed_encoder)

    if not base_encoder.is_file():
        raise FileNotFoundError(f"BASE encoder が見つかりません: {base_encoder}")
    if not proposed_encoder.is_file():
        raise FileNotFoundError(f"PROPOSED encoder が見つかりません: {proposed_encoder}")

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    feature_dim = get_image_encoder_feature_dim("clip")

    print("\nLoading test dataset...")
    test_dataset = EEGDataset(
        args.data_path,
        img_dir_training=args.img_dir_training,
        img_dir_test=args.img_dir_test,
        feature_type="clip",
        subjects=[args.subject],
        train=False,
        feature_space="clip",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    img_features = test_dataset.img_features.float()

    class_dirs = sorted(
        p for p in Path(args.img_dir_test).iterdir()
        if p.is_dir()
    )
    names = [class_name(p.name) for p in class_dirs]

    n = len(img_features)
    if len(class_dirs) != n:
        raise RuntimeError(
            f"test image folders ({len(class_dirs)}) と "
            f"test features ({n}) の数が一致しません"
        )

    print("\nExtracting BASE EEG features...")
    base_eeg = extract_features(
        args.subject, base_encoder, test_loader, feature_dim, device
    )

    print("Extracting PROPOSED EEG features...")
    proposed_eeg = extract_features(
        args.subject, proposed_encoder, test_loader, feature_dim, device
    )

    if base_eeg.shape != proposed_eeg.shape or base_eeg.shape[0] != n:
        raise RuntimeError(
            f"shape mismatch: base={tuple(base_eeg.shape)}, "
            f"proposed={tuple(proposed_eeg.shape)}, targets={n}"
        )

    base = compute_ranks(base_eeg, img_features)
    proposed = compute_ranks(proposed_eeg, img_features)

    rows = []
    for i in range(n):
        br = int(base["ranks"][i].item())
        pr = int(proposed["ranks"][i].item())
        improvement = br - pr  # 正なら改善

        base_top1_idx = int(base["top1"][i].item())
        prop_top1_idx = int(proposed["top1"][i].item())

        rows.append({
            "test_index": i,
            "class_name": names[i],
            "base_rank": br,
            "proposed_rank": pr,
            "rank_improvement": improvement,
            "base_top5": "yes" if br <= 5 else "no",
            "proposed_top5": "yes" if pr <= 5 else "no",
            "top5_transition": (
                "OUT->IN" if br > 5 and pr <= 5
                else "IN->OUT" if br <= 5 and pr > 5
                else "IN->IN" if br <= 5 and pr <= 5
                else "OUT->OUT"
            ),
            "base_top1_index": base_top1_idx,
            "base_top1_class": names[base_top1_idx],
            "proposed_top1_index": prop_top1_idx,
            "proposed_top1_class": names[prop_top1_idx],
            "base_gt_similarity": float(base["gt_similarity"][i].item()),
            "proposed_gt_similarity": float(proposed["gt_similarity"][i].item()),
            "gt_similarity_change": float(
                proposed["gt_similarity"][i].item()
                - base["gt_similarity"][i].item()
            ),
        })

    # 改善量の大きい順
    rows.sort(
        key=lambda r: (
            r["rank_improvement"],
            r["gt_similarity_change"],
        ),
        reverse=True,
    )

    csv_path = output_dir / "rank_improvement.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    base_top1 = (base["ranks"] == 1).float().mean().item()
    prop_top1 = (proposed["ranks"] == 1).float().mean().item()
    base_top5 = (base["ranks"] <= 5).float().mean().item()
    prop_top5 = (proposed["ranks"] <= 5).float().mean().item()

    print("\n==============================")
    print("Retrieval summary")
    print("==============================")
    print(f"BASE     Top-1={base_top1:.4f}  Top-5={base_top5:.4f}")
    print(f"PROPOSED Top-1={prop_top1:.4f}  Top-5={prop_top5:.4f}")
    print("==============================")

    print(f"\nTop {min(args.top_n, len(rows))} rank improvements")
    print("-" * 90)
    print(f"{'#':>3}  {'class':<24} {'base':>5} {'prop':>5} {'improve':>8} {'Top5':>9}")
    print("-" * 90)

    for rank_no, row in enumerate(rows[:args.top_n], start=1):
        print(
            f"{rank_no:>3}  "
            f"{row['class_name'][:24]:<24} "
            f"{row['base_rank']:>5} "
            f"{row['proposed_rank']:>5} "
            f"{row['rank_improvement']:>+8} "
            f"{row['top5_transition']:>9}"
        )

    # 上位改善例を10件ずつモンタージュ化
    top_rows = rows[:args.top_n]
    for start in range(0, len(top_rows), 10):
        chunk = top_rows[start:start + 10]
        out_img = output_dir / f"top_improved_{start+1:02d}-{start+len(chunk):02d}.png"
        make_montage(chunk, class_dirs, out_img, start)
        print("saved:", out_img)

    print("\nCSV:", csv_path)
    print("Done.")


if __name__ == "__main__":
    main()

# # python rank_improvement_compare.py \
#   --base_result_dir "/home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/outputs/baseline/clip/full/mse_contrastive/val_rdm_mse/sub-01/10-10_07-51" \
#   --proposed_result_dir "/home/moepy/ozakitakuma/EEG_Image_decode_develop/Generation/outputs/baseline/clip/full/mse_contrastive_svd_bottom256_w4/val_rdm_mse/sub-01/10-10_06-04" \
#   --subject sub-01