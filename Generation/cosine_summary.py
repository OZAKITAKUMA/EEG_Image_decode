import argparse
import csv
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rank_improvement_compare import (
    EEGDataset,
    get_image_encoder_feature_dim,
    read_paths_info,
    extract_features,
    compute_ranks,
)


EXPERIMENTS = {
    "Base": "mse_contrastive",
    "Bottom256_x2": "mse_contrastive_svd_bottom256_w2",
    "Bottom256_x4": "mse_contrastive_svd_bottom256_w4",
    "Bottom256_x8": "mse_contrastive_svd_bottom256_w8",
}


def find_run(generation_dir, experiment, subject, latest):
    parent = (
        generation_dir / "outputs/baseline/clip/full"
        / experiment / "val_rdm_mse" / subject
    )
    runs = sorted(
        p for p in parent.glob("*")
        if p.is_dir() and (p / "paths_info.txt").is_file()
    )
    if not runs:
        raise FileNotFoundError(f"結果フォルダがありません: {parent}")
    if len(runs) > 1 and not latest:
        names = ", ".join(p.name for p in runs)
        raise RuntimeError(
            f"{experiment}/{subject} に複数の実験があります: {names}\n"
            "最新を使う場合は --latest を指定してください。"
        )
    return runs[-1]


def find_encoder(generation_dir, experiment, subject, run):
    info = read_paths_info(run)
    recorded = Path(info["encoder_path"])
    if not recorded.is_absolute():
        recorded = generation_dir / recorded

    # 結果フォルダに対応するモデルパス。
    # paths_info.txt に bottom64 等の古いパスが残る場合にも対応。
    expected = (
        generation_dir / "models/baseline/clip/full"
        / experiment / "val_rdm_mse/encoder"
        / subject / run.name / "best.pth"
    )

    if expected.is_file():
        if recorded.resolve() != expected.resolve():
            print(
                f"  encoderパス不一致: {experiment}/{subject}\n"
                f"    記録: {recorded}\n"
                f"    使用: {expected}"
            )
        return expected

    if recorded.is_file():
        return recorded

    raise FileNotFoundError(
        f"encoderがありません:\n{expected}\n{recorded}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir", default=".")
    parser.add_argument(
        "--data_path", default="/home/moepy/ozakitakuma/data_eeg"
    )
    parser.add_argument(
        "--img_dir_training",
        default="/home/moepy/ozakitakuma/data_image/training_images",
    )
    parser.add_argument(
        "--img_dir_test",
        default="/home/moepy/ozakitakuma/data_image/test_images",
    )
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--output", default="cosine_summary.csv")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    generation_dir = Path(args.generation_dir).resolve()
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    labels = list(EXPERIMENTS)

    # 推論前に40個の結果フォルダ・モデルを確認。
    selected = {}
    print("使用する実験フォルダ:")
    for subject in subjects:
        for label, experiment in EXPERIMENTS.items():
            run = find_run(
                generation_dir, experiment, subject, args.latest
            )
            encoder = find_encoder(
                generation_dir, experiment, subject, run
            )
            selected[subject, label] = encoder
            print(f"  {subject}  {label:<14}  {run.name}")

    device = torch.device(
        args.gpu if torch.cuda.is_available() else "cpu"
    )
    feature_dim = get_image_encoder_feature_dim("clip")
    rows = []

    for subject in subjects:
        # データの読み込みは被験者ごとに一度だけ。
        # 順位分析コードと同じデータセット設定を使用。
        dataset = EEGDataset(
            args.data_path,
            img_dir_training=args.img_dir_training,
            img_dir_test=args.img_dir_test,
            feature_type="clip",
            subjects=[subject],
            train=False,
            feature_space="clip",
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        image_features = dataset.img_features.float()
        row = {"Subject": subject}

        for label in labels:
            eeg_features = extract_features(
                subject,
                selected[subject, label],
                loader,
                feature_dim,
                device,
            )
            if eeg_features.shape != image_features.shape:
                raise RuntimeError(
                    f"{subject}/{label}: 特徴形状が一致しません。"
                    f" EEG={tuple(eeg_features.shape)},"
                    f" GT={tuple(image_features.shape)}"
                )

            scores = compute_ranks(
                eeg_features, image_features
            )["gt_similarity"]

            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"{subject}/{label}: 類似度にNaN/Infがあります。"
                )
            row[label] = scores.mean().item()

        rows.append(row)
        print(
            subject + "\t"
            + "\t".join(f"{row[label]:.6f}" for label in labels)
        )
        del dataset, loader, image_features

    rows.append({
        "Subject": "Mean",
        **{
            label: statistics.mean(r[label] for r in rows)
            for label in labels
        },
    })
    # 最初の10行だけを使い、被験者間の標本標準偏差を計算。
    rows.append({
        "Subject": "Std",
        **{
            label: statistics.stdev(r[label] for r in rows[:10])
            for label in labels
        },
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Subject"] + labels
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nExcel貼り付け用:")
    print("\t".join(["Subject"] + labels))
    for row in rows:
        print(
            row["Subject"] + "\t"
            + "\t".join(f"{row[label]:.6f}" for label in labels)
        )
    print(f"\n保存先: {output.resolve()}")


if __name__ == "__main__":
    main()