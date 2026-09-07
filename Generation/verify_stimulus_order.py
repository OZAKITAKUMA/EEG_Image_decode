import argparse
import os
import sys

import torch
import torch.nn.functional as F


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(ROOT_DIR)

from eegdatasets import (
    EEGDataset,
    _ensure_clip_loaded,
    _clip_state,
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify test-stimulus ordering."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/moepy/ozakitakuma/data_eeg",
    )

    parser.add_argument(
        "--img_dir_test",
        type=str,
        default="/home/moepy/ozakitakuma/data_image/test_images",
    )

    parser.add_argument(
        "--clip_feature_path",
        type=str,
        default="../features/ViT-H-14_features_test.pt",
    )

    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. EEG側のテスト刺激順を確認
    # ---------------------------------------------------------
    dataset = EEGDataset(
        data_path=args.data_path,
        img_dir_test=args.img_dir_test,
        features_path=args.clip_feature_path,
        subjects=[args.subject],
        train=False,
        feature_space="clip",
    )

    n_samples = len(dataset)

    expected_labels = torch.arange(n_samples)

    eeg_order_ok = torch.equal(
        dataset.labels.cpu(),
        expected_labels,
    )

    print("\n===== EEG stimulus order =====")
    print(f"Samples: {n_samples}")
    print(
        "Labels are 0..199 in order:",
        eeg_order_ok,
    )

    if not eeg_order_ok:
        print("Actual labels:")
        print(dataset.labels)

    # ---------------------------------------------------------
    # 2. 保存済みCLIP特徴を読み込む
    # ---------------------------------------------------------
    cached_data = torch.load(
        args.clip_feature_path,
        map_location="cpu",
        weights_only=False,
    )

    cached_features = (
        cached_data["img_features"]
        .float()
        .cpu()
    )

    print("\n===== Cached CLIP features =====")
    print(
        "shape:",
        tuple(cached_features.shape),
    )

    # ---------------------------------------------------------
    # 3. test画像をsorted順でもう一度CLIPへ入力
    # ---------------------------------------------------------
    _ensure_clip_loaded()

    fresh_features = dataset._encode_images(
        dataset.img
    ).float().cpu()

    print("\n===== Fresh CLIP features =====")
    print(
        "shape:",
        tuple(fresh_features.shape),
    )

    # ---------------------------------------------------------
    # 4. fresh × cached の200×200 cosine similarity
    # ---------------------------------------------------------
    fresh_norm = F.normalize(
        fresh_features,
        dim=1,
    )

    cached_norm = F.normalize(
        cached_features,
        dim=1,
    )

    similarity = (
        fresh_norm
        @ cached_norm.T
    )

    predicted_indices = similarity.argmax(
        dim=1
    )

    expected_indices = torch.arange(
        len(fresh_features)
    )

    order_accuracy = (
        predicted_indices
        == expected_indices
    ).float().mean()

    diagonal_cosine = similarity.diag()

    print("\n===== Stimulus-order check =====")
    print(
        f"Order Top-1 accuracy: "
        f"{order_accuracy.item():.4f}"
    )
    print(
        f"Diagonal cosine mean: "
        f"{diagonal_cosine.mean().item():.6f}"
    )
    print(
        f"Diagonal cosine min:  "
        f"{diagonal_cosine.min().item():.6f}"
    )

    mismatches = torch.where(
        predicted_indices != expected_indices
    )[0]

    print(
        f"Mismatched rows: "
        f"{len(mismatches)}"
    )

    for index in mismatches.tolist():
        print(
            f"row {index}: "
            f"expected={index}, "
            f"matched={predicted_indices[index].item()}, "
            f"folder={os.path.basename(os.path.dirname(dataset.img[index]))}"
        )

    # ---------------------------------------------------------
    # 5. 最終判定
    # ---------------------------------------------------------
    if (
        eeg_order_ok
        and order_accuracy.item() == 1.0
    ):
        print(
            "\n[OK] EEG, test images, and cached CLIP "
            "features use the same stimulus order."
        )
    else:
        print(
            "\n[WARNING] Stimulus ordering needs investigation."
        )


if __name__ == "__main__":
    main()