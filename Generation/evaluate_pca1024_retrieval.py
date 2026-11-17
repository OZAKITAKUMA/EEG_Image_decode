import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eegdatasets import EEGDataset
from models.atms import ATMS, extract_id_from_string


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--img_directory_test", required=True)
    parser.add_argument("--img_dir_training", default=None)

    parser.add_argument(
        "--encoder_path",
        default=(
            "./models/pca1024_20261116_135400/"
            "encoder/sub-01/11-16_13-54/best.pth"
        ),
    )

    parser.add_argument(
        "--features_path",
        default="../features/ViT-H-14_pca1024_features_test.pt",
    )

    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # 1. PCA1024のTest画像特徴を読み込む
    saved = torch.load(
        args.features_path,
        map_location="cpu",
        weights_only=False,
    )

    img_features = saved["img_features"].float()

    assert img_features.shape == (200, 1024)

    # 2. Test EEGデータセット
    dataset = EEGDataset(
        args.data_path,
        img_dir_training=args.img_dir_training,
        img_dir_test=args.img_directory_test,
        feature_type="clip",
        subjects=[args.subject],
        train=False,
        feature_space="clip",
        preloaded_features=saved,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. 学習済みEncoder
    model = ATMS(outputs_dim=1024)

    model.load_state_dict(
        torch.load(
            args.encoder_path,
            map_location="cpu",
            weights_only=False,
        )
    )

    model = model.to(device)
    model.eval()

    # 4. EEG特徴を予測
    eeg_features = []

    subject_id = extract_id_from_string(args.subject)

    with torch.no_grad():
        for batch in loader:
            eeg = batch[0].to(device)

            subject_ids = torch.full(
                (eeg.shape[0],),
                subject_id,
                dtype=torch.long,
                device=device,
            )

            output = model(eeg, subject_ids)

            eeg_features.append(output.float().cpu())

    eeg_features = torch.cat(eeg_features, dim=0)

    print("EEG features:", eeg_features.shape)
    print("Image features:", img_features.shape)

    assert eeg_features.shape == img_features.shape

    # 5. コサイン類似度
    eeg_norm = F.normalize(eeg_features, dim=1)
    img_norm = F.normalize(img_features, dim=1)

    similarity = eeg_norm @ img_norm.T

    print("Similarity matrix:", similarity.shape)

    # 6. 200-way検索精度
    targets = torch.arange(200)

    top1 = (
        similarity.argmax(dim=1) == targets
    ).float().mean()

    top5_indices = similarity.topk(5, dim=1).indices

    top5 = (
        top5_indices == targets.unsqueeze(1)
    ).any(dim=1).float().mean()

    print("\n===== PCA1024 TEST RETRIEVAL =====")
    print(f"Top-1: {top1.item():.4f}")
    print(f"Top-5: {top5.item():.4f}")

        # 7. 正解画像とのコサイン類似度
    cosine_per_image = similarity.diag()

    mean_cosine = cosine_per_image.mean()
    std_cosine = cosine_per_image.std()

    print("\n===== PCA1024 TEST COSINE =====")
    print(f"Mean cosine: {mean_cosine.item():.4f}")
    print(f"Std cosine:  {std_cosine.item():.4f}")

        # 不正解画像とのコサイン類似度
    mask = ~torch.eye(200, dtype=torch.bool)

    incorrect_cosines = similarity[mask]

    mean_incorrect = incorrect_cosines.mean()
    std_incorrect = incorrect_cosines.std()

    print("\n===== INCORRECT IMAGE COSINE =====")
    print(f"Mean incorrect cosine: {mean_incorrect.item():.4f}")
    print(f"Std incorrect cosine:  {std_incorrect.item():.4f}")

    print(
        f"Correct - Incorrect: "
        f"{(mean_cosine - mean_incorrect).item():.4f}"
    )


if __name__ == "__main__":
    main()