import argparse
import os
import sys
import torch


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(ROOT_DIR)

from eegdatasets import EEGDataset
from encoder_utils import stratified_condition_split

import argparse
import os
import sys

import torch


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(ROOT_DIR)

from eegdatasets import EEGDataset
from encoder_utils import stratified_condition_split
from models.atms import ATMS, extract_id_from_string

from representation_analysis import (
    compute_cosine_rdm,
    compute_rsa_similarity,
)

def sample_train_stimuli(
    train_indices,
    n_classes_total=1654,
    n_classes_sample=200,
    conditions_per_class=10,
    seed=42,
):
    """
    実際に学習に使われたtrain条件だけから、
    200クラスを選び、各クラス1条件ずつ抽出する。

    Returns
    -------
    selected_classes:
        list[int], length=200

    selected_indices:
        list[int], length=200
        full_train_dataset上のindex
    """

    import random

    rng = random.Random(seed)

    # 1654クラスから200クラスを重複なしで抽出
    selected_classes = rng.sample(
        range(n_classes_total),
        n_classes_sample,
    )

    train_index_set = set(train_indices)

    selected_indices = []

    for class_index in selected_classes:
        class_start = (
            class_index
            * conditions_per_class
        )

        # このクラスに属する10条件のうち、
        # 実際のtrain splitに含まれる9条件だけを集める
        candidate_indices = [
            class_start + condition_index
            for condition_index in range(
                conditions_per_class
            )
            if (
                class_start + condition_index
                in train_index_set
            )
        ]

        if len(candidate_indices) != 9:
            raise RuntimeError(
                f"class {class_index}: "
                f"expected 9 train conditions, "
                f"got {len(candidate_indices)}"
            )

        # 9条件の中から1条件だけ選択
        selected_index = rng.choice(
            candidate_indices
        )

        selected_indices.append(
            selected_index
        )

    return selected_classes, selected_indices


def main():
    parser = argparse.ArgumentParser(
        description="Compare train/test EEG-to-CLIP representation structure."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/moepy/ozakitakuma/data_eeg",
    )

    parser.add_argument(
        "--img_dir_training",
        type=str,
        default="/home/moepy/ozakitakuma/data_image/training_images",
    )

    parser.add_argument(
        "--img_dir_test",
        type=str,
        default="/home/moepy/ozakitakuma/data_image/test_images",
    )

    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
    )

    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to the trained ATMS encoder checkpoint.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 学習済みwithin-subject encoderを読み込む
    # ---------------------------------------------------------
    if not os.path.isfile(args.encoder_path):
        raise FileNotFoundError(
            f"Encoder checkpoint not found: "
            f"{args.encoder_path}"
        )

    eeg_model = ATMS(
        outputs_dim=1024,
    )

    state_dict = torch.load(
        args.encoder_path,
        map_location="cpu",
    )

    eeg_model.load_state_dict(
        state_dict
    )

    eeg_model.eval()

    print(
        "Loaded encoder:",
        args.encoder_path,
    )
    
    # ---------------------------------------------------------
    # 1. 学習時と同じfull train datasetを作る
    # ---------------------------------------------------------
    full_train_dataset = EEGDataset(
        data_path=args.data_path,
        img_dir_training=args.img_dir_training,
        subjects=[args.subject],
        train=True,
        avg_trials=True,
        feature_space="clip",
    )

    # ---------------------------------------------------------
    # 2. 学習時と同じseedでtrain / validationを再現
    #
    # avg_trials=Trueなので、
    # 4 trialsを平均した後は1 condition = 1 sample
    # ---------------------------------------------------------
    train_indices, val_indices = stratified_condition_split(
        n_classes=1654,
        conditions_per_class=10,
        trials_per_condition=1,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    selected_classes, selected_indices = sample_train_stimuli(
    train_indices=train_indices,
    n_classes_total=1654,
    n_classes_sample=200,
    conditions_per_class=10,
    seed=args.seed, # args.seedに絶対もどす
    )

    train_index_set = set(train_indices)
    val_index_set = set(val_indices)

    # 200個すべてtrain splitに含まれていることを確認
    if not all(
        index in train_index_set
        for index in selected_indices
    ):
        raise RuntimeError(
            "Selected samples contain non-training indices."
        )

    # validationとの重複が0個であることを確認
    overlap_with_val = (
        set(selected_indices)
        & val_index_set
    )

    if overlap_with_val:
        raise RuntimeError(
            "Selected training samples overlap "
            f"with validation: {sorted(overlap_with_val)}"
        )

    # 選んだindexのEEGラベルが、
    # 選んだclass番号と本当に一致しているか確認
    selected_labels = (
        full_train_dataset.labels[selected_indices]
        .tolist()
    )

    if selected_labels != selected_classes:
        raise RuntimeError(
            "Selected EEG labels do not match "
            "selected class IDs."
        )

        # ---------------------------------------------------------
    # 3. RSAに使用する200刺激を実際に取り出す
    # ---------------------------------------------------------
    train_eeg = full_train_dataset.data[
        selected_indices
    ].float()

    train_clip = full_train_dataset.img_features[
        selected_indices
    ].float()

    if tuple(train_eeg.shape) != (200, 63, 250):
        raise RuntimeError(
            "Unexpected sampled EEG shape: "
            f"{tuple(train_eeg.shape)}"
        )

    if tuple(train_clip.shape) != (200, 1024):
        raise RuntimeError(
            "Unexpected sampled CLIP shape: "
            f"{tuple(train_clip.shape)}"
        )

    if not train_eeg.isfinite().all():
        raise RuntimeError(
            "Sampled EEG contains NaN or Inf."
        )

    if not train_clip.isfinite().all():
        raise RuntimeError(
            "Sampled CLIP features contain NaN or Inf."
        )

        # ---------------------------------------------------------
    # 4. 学習済みencoderでtrain EEGをCLIP空間へ写像
    # ---------------------------------------------------------
    subject_id = extract_id_from_string(
        args.subject
    )

    subject_ids = torch.full(
        (train_eeg.shape[0],),
        subject_id,
        dtype=torch.long,
    )

    with torch.no_grad():
        train_pred_clip = eeg_model(
            train_eeg,
            subject_ids,
        ).float()

    if tuple(train_pred_clip.shape) != (200, 1024):
        raise RuntimeError(
            "Unexpected predicted CLIP shape: "
            f"{tuple(train_pred_clip.shape)}"
        )

    if not train_pred_clip.isfinite().all():
        raise RuntimeError(
            "Predicted CLIP features contain NaN or Inf."
        )

        # ---------------------------------------------------------
    # 5. train prediction と正解CLIPのcosine RSAを計算
    # ---------------------------------------------------------
    train_pred_rdm = compute_cosine_rdm(
        train_pred_clip
    )

    train_clip_rdm = compute_cosine_rdm(
        train_clip
    )

    train_cosine_rsa = compute_rsa_similarity(
        train_pred_rdm,
        train_clip_rdm,
    )

    print("\n===== Train RSA =====")
    print(
        f"Train cosine RSA: "
        f"{train_cosine_rsa.item():.4f}"
    )

    # ---------------------------------------------------------
    # 6. testデータでも同じRSAを再計算
    # ---------------------------------------------------------
    test_dataset = EEGDataset(
        data_path=args.data_path,
        img_dir_training=args.img_dir_training,
        img_dir_test=args.img_dir_test,
        subjects=[args.subject],
        train=False,
        feature_space="clip",
    )

    test_eeg = test_dataset.data.float()
    test_clip = test_dataset.img_features.float()

    if tuple(test_eeg.shape) != (200, 63, 250):
        raise RuntimeError(
            "Unexpected test EEG shape: "
            f"{tuple(test_eeg.shape)}"
        )

    if tuple(test_clip.shape) != (200, 1024):
        raise RuntimeError(
            "Unexpected test CLIP shape: "
            f"{tuple(test_clip.shape)}"
        )

    test_subject_ids = torch.full(
        (test_eeg.shape[0],),
        subject_id,
        dtype=torch.long,
    )

    with torch.no_grad():
        test_pred_clip = eeg_model(
            test_eeg,
            test_subject_ids,
        ).float()

    if tuple(test_pred_clip.shape) != (200, 1024):
        raise RuntimeError(
            "Unexpected test prediction shape: "
            f"{tuple(test_pred_clip.shape)}"
        )

    test_pred_rdm = compute_cosine_rdm(
        test_pred_clip
    )

    test_clip_rdm = compute_cosine_rdm(
        test_clip
    )

    test_cosine_rsa = compute_rsa_similarity(
        test_pred_rdm,
        test_clip_rdm,
    )

    print("\n===== Train vs Test RSA =====")
    print(
        f"Train cosine RSA: "
        f"{train_cosine_rsa.item():.4f}"
    )
    print(
        f"Test cosine RSA:  "
        f"{test_cosine_rsa.item():.4f}"
    )
    
    print("\n===== Sampled train stimuli check =====")
    print(
        "Selected classes:",
        len(selected_classes),
    )
    print(
        "Selected samples:",
        len(selected_indices),
    )
    print(
        "Unique classes:",
        len(set(selected_classes)),
    )
    print(
        "Overlap with validation:",
        len(overlap_with_val),
    )
    print(
        "First 10 classes:",
        selected_classes[:10],
    )
    print(
        "First 10 dataset indices:",
        selected_indices[:10],
    )
    print(
        "First 10 EEG labels:",
        selected_labels[:10],
    )

    print(
        "Selected EEG shape:",
        tuple(train_eeg.shape),
    )
    print(
        "Selected CLIP shape:",
        tuple(train_clip.shape),
    )

    print(
        "Predicted train CLIP shape:",
        tuple(train_pred_clip.shape),
    )

    print(
        "[OK] 200 training stimuli were selected correctly."
    )
    
    print("\n===== Train dataset check =====")
    print(
        "Full train dataset:",
        len(full_train_dataset),
    )
    print(
        "EEG shape:",
        tuple(full_train_dataset.data.shape),
    )
    print(
        "Image CLIP shape:",
        tuple(full_train_dataset.img_features.shape),
    )
    print(
        "Train samples:",
        len(train_indices),
    )
    print(
        "Validation samples:",
        len(val_indices),
    )

    # 1654 classes × 10 conditions
    expected_full = 1654 * 10

    # 各クラス9条件がtrain、1条件がvalidation
    expected_train = 1654 * 9
    expected_val = 1654 * 1

    if len(full_train_dataset) != expected_full:
        raise RuntimeError(
            f"Unexpected full dataset size: "
            f"{len(full_train_dataset)} != {expected_full}"
        )

    if len(train_indices) != expected_train:
        raise RuntimeError(
            f"Unexpected train size: "
            f"{len(train_indices)} != {expected_train}"
        )

    if len(val_indices) != expected_val:
        raise RuntimeError(
            f"Unexpected validation size: "
            f"{len(val_indices)} != {expected_val}"
        )

    print("[OK] Training split was reproduced correctly.")


if __name__ == "__main__":
    main()