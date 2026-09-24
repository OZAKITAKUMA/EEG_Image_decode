import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from eegdatasets import EEGDataset
from encoder_utils import stratified_condition_split
from models.atms import ATMS, extract_id_from_string


# ============================================================
# Utility
# ============================================================

def class_name_from_folder(folder_name):
    """
    例:
        00001_aardvark -> aardvark
    """
    if "_" in folder_name:
        return folder_name.split("_", 1)[1]

    return folder_name


def get_class_folders(img_dir_training):
    folders = sorted(
        folder
        for folder in os.listdir(img_dir_training)
        if os.path.isdir(
            os.path.join(img_dir_training, folder)
        )
    )

    if len(folders) != 1654:
        raise RuntimeError(
            f"Expected 1654 class folders, "
            f"but found {len(folders)}"
        )

    return folders


# ============================================================
# THINGSplus category mapping
# ============================================================

def load_category_mapping(
    category_tsv,
    img_dir_training,
):
    """
    class index
        -> set of THINGSplus categories

    THINGSplus の uniqueID を使って対応付ける。

    multi-labelなので、例えば

        aardvark
            -> {"animal", "mammal"}

    のようになる。
    """

    df = pd.read_csv(
        category_tsv,
        sep="\t",
    )

    required_columns = {
        "category",
        "uniqueID",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "category TSV must contain "
            f"{required_columns}, "
            f"but got {list(df.columns)}"
        )

    concept_to_categories = {}

    for _, row in df.iterrows():
        if pd.isna(row["uniqueID"]):
            continue

        if pd.isna(row["category"]):
            continue

        concept = str(
            row["uniqueID"]
        ).strip().lower()

        category = str(
            row["category"]
        ).strip()

        if not concept or not category:
            continue

        concept_to_categories.setdefault(
            concept,
            set(),
        ).add(category)

    folders = get_class_folders(
        img_dir_training
    )

    class_to_categories = {}

    for class_idx, folder in enumerate(folders):
        concept = class_name_from_folder(
            folder
        ).strip().lower()

        class_to_categories[class_idx] = set(
            concept_to_categories.get(
                concept,
                set(),
            )
        )

    mapped = sum(
        1
        for categories
        in class_to_categories.values()
        if len(categories) > 0
    )

    all_categories = sorted({
        category
        for categories
        in class_to_categories.values()
        for category
        in categories
    })

    print(
        "[THINGSplus mapping]"
    )

    print(
        f"  mapped classes: "
        f"{mapped}/1654"
    )

    print(
        f"  categories: "
        f"{len(all_categories)}"
    )

    return (
        class_to_categories,
        all_categories,
        folders,
    )

# ============================================================
# Subject ID
# ============================================================

def make_subject_ids(
    model,
    subject,
    batch_size,
    device,
    use_subject_id,
):
    """
    encoder_utils.py と同じ考え方。

    use_subject_id=False の場合は、
    Embedding範囲外のIDを渡して
    shared tokenを使わせる。
    """

    if use_subject_id:
        subject_id = extract_id_from_string(
            subject
        )

    else:
        embedding_layer = (
            model
            .encoder
            .enc_embedding
            .subject_embedding
            .subject_embedding
        )

        subject_id = (
            embedding_layer.num_embeddings
        )

    return torch.full(
        (batch_size,),
        subject_id,
        dtype=torch.long,
        device=device,
    )


# ============================================================
# Checkpoint
# ============================================================

def load_model_checkpoint(
    checkpoint_path,
    device,
):
    """
    現在のtrain.pyではbest.pthに
    state_dictそのものを保存している。

    一応、
        state_dict
        model_state_dict
    形式にも対応させておく。
    """

    model = ATMS(
        outputs_dim=1024,
    ).to(device)

    saved = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(saved, dict)
        and "state_dict" in saved
    ):
        state_dict = saved["state_dict"]

    elif (
        isinstance(saved, dict)
        and "model_state_dict" in saved
    ):
        state_dict = saved[
            "model_state_dict"
        ]

    else:
        state_dict = saved

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# Validation feature extraction
# ============================================================

@torch.no_grad()
def extract_validation_features(
    args,
    device,
):
    """
    validation EEGを学習済みEncoderへ1回だけ通す。

    保存するもの:
        EEG prediction : (1654, 1024)
        Teacher CLIP   : (1654, 1024)
        labels         : (1654,)
    """

    print(
        "\n"
        "========================================"
    )

    print(
        "Extract validation EEG features"
    )

    print(
        "========================================"
    )

    dataset = EEGDataset(
        args.data_path,
        img_dir_training=(
            args.img_dir_training
        ),
        img_dir_test=(
            args.img_dir_test
        ),
        features_dir=(
            args.features_dir
        ),
        subjects=[
            args.subject
        ],
        train=True,
        avg_trials=True,
        feature_space="clip",
    )

    # training時と同じvalidation split
    #
    # avg_trials=True なので
    # trials_per_condition = 1
    _, val_indices = (
        stratified_condition_split(
            n_classes=1654,
            conditions_per_class=10,
            trials_per_condition=1,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    )

    val_subset = Subset(
        dataset,
        val_indices,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model_checkpoint(
        args.checkpoint,
        device,
    )

    eeg_feature_list = []
    teacher_feature_list = []
    label_list = []

    for batch in val_loader:
        (
            eeg_data,
            labels,
            text,
            text_features,
            img,
            img_features,
        ) = batch

        eeg_data = eeg_data.to(
            device
        )

        batch_size = eeg_data.size(0)

        subject_ids = make_subject_ids(
            model=model,
            subject=args.subject,
            batch_size=batch_size,
            device=device,
            use_subject_id=(
                not args.no_subject_id
            ),
        )

        eeg_features = model(
            eeg_data,
            subject_ids,
        ).float()

        eeg_feature_list.append(
            eeg_features.detach().cpu()
        )

        teacher_feature_list.append(
            img_features
            .detach()
            .float()
            .cpu()
        )

        label_list.append(
            labels.detach().cpu()
        )

    eeg_features = torch.cat(
        eeg_feature_list,
        dim=0,
    )

    teacher_features = torch.cat(
        teacher_feature_list,
        dim=0,
    )

    labels = torch.cat(
        label_list,
        dim=0,
    ).long()

    print(
        "EEG prediction:",
        eeg_features.shape,
    )

    print(
        "Teacher CLIP:  ",
        teacher_features.shape,
    )

    print(
        "Labels:        ",
        labels.shape,
    )

    if eeg_features.shape != (
        1654,
        1024,
    ):
        raise RuntimeError(
            "Unexpected EEG prediction shape: "
            f"{tuple(eeg_features.shape)}"
        )

    if teacher_features.shape != (
        1654,
        1024,
    ):
        raise RuntimeError(
            "Unexpected teacher CLIP shape: "
            f"{tuple(teacher_features.shape)}"
        )

    unique_labels = torch.unique(
        labels
    )

    if unique_labels.numel() != 1654:
        raise RuntimeError(
            "Validation split should contain "
            "exactly one sample per class, "
            f"but got {unique_labels.numel()} "
            "unique classes."
        )

    cache = {
        "eeg_features":
            eeg_features,

        "teacher_features":
            teacher_features,

        "labels":
            labels,

        "val_indices":
            torch.tensor(
                val_indices,
                dtype=torch.long,
            ),

        "subject":
            args.subject,

        "checkpoint":
            os.path.abspath(
                args.checkpoint
            ),

        "seed":
            args.seed,

        "val_ratio":
            args.val_ratio,
    }

    torch.save(
        cache,
        args.feature_cache,
    )

    print(
        "\nSaved feature cache:"
    )

    print(
        args.feature_cache
    )

    return cache


# ============================================================
# Teacher CLIP kNN
# ============================================================

def compute_teacher_knn(
    teacher_features,
    k,
):
    """
    Teacher CLIP空間そのもののkNN。

    自分自身は除外する。
    """

    teacher_n = F.normalize(
        teacher_features.float(),
        dim=-1,
    )

    similarities = (
        teacher_n
        @ teacher_n.T
    )

    similarities.fill_diagonal_(
        -float("inf")
    )

    knn_indices = torch.topk(
        similarities,
        k=k,
        dim=1,
    ).indices

    return (
        teacher_n,
        knn_indices.cpu(),
    )


# ============================================================
# Category center distance
# ============================================================

def compute_center_distances(
    teacher_n,
    labels,
    class_to_categories,
):
    """
    各validation画像について、

        Teacher CLIP特徴
            vs
        その画像が属するカテゴリ中心

    のcosine distanceを計算。

    multi-labelの場合は、
    各カテゴリ中心までの距離を平均する。

    重要:
        自分自身はカテゴリ中心の計算から除外する
        leave-one-out方式。
    """

    labels_list = labels.tolist()

    row_categories = [
        class_to_categories.get(
            int(class_idx),
            set(),
        )
        for class_idx
        in labels_list
    ]

    # category -> validation row indices
    category_to_rows = {}

    for row_idx, categories in enumerate(
        row_categories
    ):
        for category in categories:
            category_to_rows.setdefault(
                category,
                [],
            ).append(row_idx)

    center_distances = []
    n_centers_used = []

    for row_idx, categories in enumerate(
        row_categories
    ):
        distances = []

        for category in categories:
            members = (
                category_to_rows[
                    category
                ]
            )

            # 自分自身を除外
            other_members = [
                index
                for index in members
                if index != row_idx
            ]

            if len(other_members) == 0:
                continue

            category_vectors = (
                teacher_n[
                    other_members
                ]
            )

            center = (
                category_vectors.mean(
                    dim=0
                )
            )

            center = F.normalize(
                center.unsqueeze(0),
                dim=-1,
            ).squeeze(0)

            similarity = torch.dot(
                teacher_n[row_idx],
                center,
            ).item()

            cosine_distance = (
                1.0
                - similarity
            )

            distances.append(
                cosine_distance
            )

        if len(distances) == 0:
            center_distances.append(
                np.nan
            )

            n_centers_used.append(
                0
            )

        else:
            center_distances.append(
                float(
                    np.mean(
                        distances
                    )
                )
            )

            n_centers_used.append(
                len(distances)
            )

    return (
        np.asarray(
            center_distances,
            dtype=float,
        ),
        np.asarray(
            n_centers_used,
            dtype=int,
        ),
        row_categories,
    )


# ============================================================
# Boundary score
# ============================================================

def compute_boundary_scores(
    knn_indices,
    row_categories,
):
    """
    Teacher CLIPのTop-k近傍について、

    「自分とカテゴリを1つも共有しない近傍」

    の割合を測る。

    例:

        source:
            {animal, mammal}

        neighbor A:
            {animal, bird}
            -> animal共有
            -> boundaryではない

        neighbor B:
            {tool, hardware}
            -> 共有カテゴリ0
            -> boundary側

    THINGSplusに未対応の近傍は
    「別カテゴリ」と断定できないので
    denominatorから除外する。
    """

    boundary_scores = []
    known_neighbor_counts = []
    disjoint_neighbor_counts = []
    shared_neighbor_counts = []

    for row_idx in range(
        len(row_categories)
    ):
        source_categories = (
            row_categories[row_idx]
        )

        # source自体がTHINGSplus未対応
        if len(source_categories) == 0:
            boundary_scores.append(
                np.nan
            )

            known_neighbor_counts.append(
                0
            )

            disjoint_neighbor_counts.append(
                0
            )

            shared_neighbor_counts.append(
                0
            )

            continue

        known = 0
        disjoint = 0
        shared = 0

        for neighbor_idx in (
            knn_indices[
                row_idx
            ].tolist()
        ):
            neighbor_categories = (
                row_categories[
                    neighbor_idx
                ]
            )

            # THINGSplusでカテゴリ不明
            if len(neighbor_categories) == 0:
                continue

            known += 1

            overlap = (
                source_categories
                & neighbor_categories
            )

            if len(overlap) == 0:
                disjoint += 1
            else:
                shared += 1

        if known == 0:
            boundary_score = np.nan

        else:
            boundary_score = (
                disjoint
                / known
            )

        boundary_scores.append(
            boundary_score
        )

        known_neighbor_counts.append(
            known
        )

        disjoint_neighbor_counts.append(
            disjoint
        )

        shared_neighbor_counts.append(
            shared
        )

    return (
        np.asarray(
            boundary_scores,
            dtype=float,
        ),
        np.asarray(
            known_neighbor_counts,
            dtype=int,
        ),
        np.asarray(
            disjoint_neighbor_counts,
            dtype=int,
        ),
        np.asarray(
            shared_neighbor_counts,
            dtype=int,
        ),
    )


# ============================================================
# EEG prediction quality
# ============================================================

def compute_eeg_prediction_quality(
    eeg_features,
    teacher_features,
):
    """
    各validationサンプルについて、

        gt_cosine:
            EEG予測と
            対応する正解Teacher CLIPとのcosine

        gt_rank:
            1654個のvalidation Teacher CLIPの中で
            正解Teacher CLIPが何位か

    を計算する。

    注意:
    train.pyの従来Top-k解析の
    「各classの先頭画像をbankにする」
    方法ではなく、

    今回は「そのvalidation画像自身」を
    GTとして使う。

    サンプル単位の難しさを見るため。
    """

    eeg_n = F.normalize(
        eeg_features.float(),
        dim=-1,
    )

    teacher_n = F.normalize(
        teacher_features.float(),
        dim=-1,
    )

    similarities = (
        eeg_n
        @ teacher_n.T
    )

    n_samples = similarities.size(0)

    row_indices = torch.arange(
        n_samples
    )

    gt_cosine = similarities[
        row_indices,
        row_indices,
    ]

    gt_ranks = []

    for i in range(n_samples):
        gt_similarity = (
            similarities[
                i,
                i,
            ]
        )

        # GTよりcosineが高い候補数 + 1
        rank = (
            torch.sum(
                similarities[i]
                > gt_similarity
            ).item()
            + 1
        )

        gt_ranks.append(
            rank
        )

    return (
        gt_cosine.cpu().numpy(),
        np.asarray(
            gt_ranks,
            dtype=int,
        ),
    )


# ============================================================
# Spearman correlation
# ============================================================

def spearman_rho(
    x,
    y,
):
    """
    scipyへの依存を避けるため、
    rank変換してからPearson相関を取る。
    """

    x = pd.Series(
        x,
        dtype=float,
    )

    y = pd.Series(
        y,
        dtype=float,
    )

    valid = (
        x.notna()
        & y.notna()
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return (
            np.nan,
            len(x),
        )

    x_rank = x.rank(
        method="average"
    )

    y_rank = y.rank(
        method="average"
    )

    rho = x_rank.corr(
        y_rank,
        method="pearson",
    )

    return (
        float(rho),
        len(x),
    )


# ============================================================
# Quartile summary
# ============================================================

def make_quartile_summary(
    df,
    variable,
):
    """
    center_distanceやboundary_scoreを
    低い順に4グループへ分け、
    EEG性能を見る。
    """

    working = df[
        [
            variable,
            "gt_cosine",
            "gt_rank",
        ]
    ].dropna().copy()

    if len(working) < 4:
        return pd.DataFrame()

    try:
        working["quartile"] = pd.qcut(
            working[variable],
            q=4,
            labels=False,
            duplicates="drop",
        )

    except ValueError:
        return pd.DataFrame()

    summary = (
        working
        .groupby(
            "quartile"
        )
        .agg(
            n=(
                variable,
                "size",
            ),

            variable_mean=(
                variable,
                "mean",
            ),

            gt_cosine_mean=(
                "gt_cosine",
                "mean",
            ),

            gt_cosine_median=(
                "gt_cosine",
                "median",
            ),

            gt_rank_mean=(
                "gt_rank",
                "mean",
            ),

            gt_rank_median=(
                "gt_rank",
                "median",
            ),
        )
        .reset_index()
    )

    # 見やすく 1,2,3,4 にする
    summary["quartile"] = (
        summary["quartile"]
        + 1
    )

    return summary


# ============================================================
# Scatter plot
# ============================================================

def save_scatter(
    df,
    x_column,
    y_column,
    title,
    output_path,
):
    plot_df = df[
        [
            x_column,
            y_column,
        ]
    ].dropna()

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        plot_df[x_column],
        plot_df[y_column],
        alpha=0.4,
        s=18,
    )

    plt.xlabel(
        x_column
    )

    plt.ylabel(
        y_column
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether difficult positions "
            "in teacher CLIP space are harder "
            "for EEG prediction."
        )
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=(
            "/home/moepy/"
            "ozakitakuma/"
            "data_eeg"
        ),
    )

    parser.add_argument(
        "--img_dir_training",
        type=str,
        default=(
            "/home/moepy/"
            "ozakitakuma/"
            "data_image/"
            "training_images"
        ),
    )

    parser.add_argument(
        "--img_dir_test",
        type=str,
        default=(
            "/home/moepy/"
            "ozakitakuma/"
            "data_image/"
            "test_images"
        ),
    )

    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--category_tsv",
        type=str,
        default=(
            "Generation/"
            "category53_long-format.tsv"
        ),
    )

    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to trained "
            "encoder best.pth"
        ),
    )

    parser.add_argument(
        "--no_subject_id",
        action="store_true",
        help=(
            "Use shared subject token."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help=(
            "Teacher CLIP kNN size "
            "for boundary score."
        ),
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

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "./outputs/"
            "eeg_sample_difficulty"
        ),
    )

    parser.add_argument(
        "--feature_cache",
        type=str,
        default=None,
        help=(
            "Path for cached "
            "validation EEG features."
        ),
    )

    parser.add_argument(
        "--force_extract",
        action="store_true",
        help=(
            "Ignore feature cache "
            "and run the encoder again."
        ),
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    if args.feature_cache is None:
        args.feature_cache = os.path.join(
            args.output_dir,
            (
                f"{args.subject}_"
                "val_features.pt"
            ),
        )

    cache_parent = os.path.dirname(
        args.feature_cache
    )

    if cache_parent:
        os.makedirs(
            cache_parent,
            exist_ok=True,
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ========================================================
    # 1. EEG prediction feature
    # ========================================================

    if (
        os.path.exists(
            args.feature_cache
        )
        and not args.force_extract
    ):
        print(
            "\nLoad cached EEG features:"
        )

        print(
            args.feature_cache
        )

        cache = torch.load(
            args.feature_cache,
            map_location="cpu",
        )

        cached_subject = cache.get(
            "subject"
        )

        if (
            cached_subject is not None
            and cached_subject
            != args.subject
        ):
            raise RuntimeError(
                "Feature cache subject mismatch: "
                f"cache={cached_subject}, "
                f"requested={args.subject}. "
                "Use another cache path "
                "or --force_extract."
            )

        cached_checkpoint = cache.get(
            "checkpoint"
        )

        current_checkpoint = os.path.abspath(
            args.checkpoint
        )

        if (
            cached_checkpoint is not None
            and os.path.abspath(
                cached_checkpoint
            )
            != current_checkpoint
        ):
            raise RuntimeError(
                "Feature cache was created "
                "from another checkpoint.\n"
                f"cache checkpoint: "
                f"{cached_checkpoint}\n"
                f"requested checkpoint: "
                f"{current_checkpoint}\n"
                "Use another cache path "
                "or --force_extract."
            )

    else:
        cache = extract_validation_features(
            args,
            device,
        )

    eeg_features = cache[
        "eeg_features"
    ].float()

    teacher_features = cache[
        "teacher_features"
    ].float()

    labels = cache[
        "labels"
    ].long()

    # ========================================================
    # 2. THINGSplus category
    # ========================================================

    (
        class_to_categories,
        categories,
        folders,
    ) = load_category_mapping(
        args.category_tsv,
        args.img_dir_training,
    )

    # ========================================================
    # 3. Teacher CLIP kNN
    # ========================================================

    print(
        "\n"
        "Compute Teacher CLIP kNN..."
    )

    (
        teacher_n,
        teacher_knn,
    ) = compute_teacher_knn(
        teacher_features,
        args.k,
    )

    # ========================================================
    # 4. Category center distance
    # ========================================================

    print(
        "Compute category-center distance..."
    )

    (
        center_distance,
        n_centers_used,
        row_categories,
    ) = compute_center_distances(
        teacher_n,
        labels,
        class_to_categories,
    )

    # ========================================================
    # 5. Boundary score
    # ========================================================

    print(
        "Compute category boundary score..."
    )

    (
        boundary_score,
        boundary_known_n,
        boundary_disjoint_n,
        boundary_shared_n,
    ) = compute_boundary_scores(
        teacher_knn,
        row_categories,
    )

    # ========================================================
    # 6. EEG prediction quality
    # ========================================================

    print(
        "Compute EEG prediction quality..."
    )

    (
        gt_cosine,
        gt_rank,
    ) = compute_eeg_prediction_quality(
        eeg_features,
        teacher_features,
    )

    # ========================================================
    # 7. Per-sample table
    # ========================================================

    rows = []

    for row_idx in range(
        len(labels)
    ):
        class_idx = int(
            labels[row_idx].item()
        )

        class_name = (
            class_name_from_folder(
                folders[class_idx]
            )
        )

        category_text = "|".join(
            sorted(
                row_categories[
                    row_idx
                ]
            )
        )

        rows.append({
            "row_index":
                row_idx,

            "class_index":
                class_idx,

            "class_name":
                class_name,

            "categories":
                category_text,

            "n_categories":
                len(
                    row_categories[
                        row_idx
                    ]
                ),

            "center_distance":
                center_distance[
                    row_idx
                ],

            "n_centers_used":
                n_centers_used[
                    row_idx
                ],

            "boundary_score":
                boundary_score[
                    row_idx
                ],

            "boundary_known_n":
                boundary_known_n[
                    row_idx
                ],

            "boundary_disjoint_n":
                boundary_disjoint_n[
                    row_idx
                ],

            "boundary_shared_n":
                boundary_shared_n[
                    row_idx
                ],

            "gt_cosine":
                gt_cosine[
                    row_idx
                ],

            "gt_rank":
                gt_rank[
                    row_idx
                ],
        })

    result_df = pd.DataFrame(
        rows
    )

    sample_csv_path = os.path.join(
        args.output_dir,
        "eeg_sample_difficulty.csv",
    )

    result_df.to_csv(
        sample_csv_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        sample_csv_path
    )

    # ========================================================
    # 8. Spearman correlations
    # ========================================================

    correlation_rows = []

    correlation_pairs = [
        (
            "center_distance",
            "gt_cosine",
        ),
        (
            "center_distance",
            "gt_rank",
        ),
        (
            "boundary_score",
            "gt_cosine",
        ),
        (
            "boundary_score",
            "gt_rank",
        ),
    ]

    print(
        "\n"
        "========================================"
    )

    print(
        "Spearman correlations"
    )

    print(
        "========================================"
    )

    for x_column, y_column in (
        correlation_pairs
    ):
        rho, n = spearman_rho(
            result_df[
                x_column
            ],
            result_df[
                y_column
            ],
        )

        correlation_rows.append({
            "x":
                x_column,

            "y":
                y_column,

            "spearman_rho":
                rho,

            "n":
                n,
        })

        print(
            f"{x_column:20s} "
            f"vs "
            f"{y_column:10s} "
            f"| rho={rho:.4f} "
            f"| n={n}"
        )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    correlation_csv_path = os.path.join(
        args.output_dir,
        "spearman_correlations.csv",
    )

    correlation_df.to_csv(
        correlation_csv_path,
        index=False,
    )

    # ========================================================
    # 9. Quartile summaries
    # ========================================================

    center_quartiles = (
        make_quartile_summary(
            result_df,
            "center_distance",
        )
    )

    boundary_quartiles = (
        make_quartile_summary(
            result_df,
            "boundary_score",
        )
    )

    center_quartile_path = os.path.join(
        args.output_dir,
        "center_distance_quartiles.csv",
    )

    boundary_quartile_path = os.path.join(
        args.output_dir,
        "boundary_score_quartiles.csv",
    )

    center_quartiles.to_csv(
        center_quartile_path,
        index=False,
    )

    boundary_quartiles.to_csv(
        boundary_quartile_path,
        index=False,
    )

    print(
        "\n=== Center-distance quartiles ==="
    )

    print(
        center_quartiles.to_string(
            index=False
        )
    )

    print(
        "\n=== Boundary-score quartiles ==="
    )

    print(
        boundary_quartiles.to_string(
            index=False
        )
    )

    # ========================================================
    # 10. Scatter plots
    # ========================================================

    save_scatter(
        result_df,
        "center_distance",
        "gt_rank",
        (
            "Teacher CLIP category-center "
            "distance vs EEG GT rank"
        ),
        os.path.join(
            args.output_dir,
            (
                "center_distance_"
                "vs_gt_rank.png"
            ),
        ),
    )

    save_scatter(
        result_df,
        "center_distance",
        "gt_cosine",
        (
            "Teacher CLIP category-center "
            "distance vs EEG GT cosine"
        ),
        os.path.join(
            args.output_dir,
            (
                "center_distance_"
                "vs_gt_cosine.png"
            ),
        ),
    )

    save_scatter(
        result_df,
        "boundary_score",
        "gt_rank",
        (
            "Teacher CLIP boundary score "
            "vs EEG GT rank"
        ),
        os.path.join(
            args.output_dir,
            (
                "boundary_score_"
                "vs_gt_rank.png"
            ),
        ),
    )

    save_scatter(
        result_df,
        "boundary_score",
        "gt_cosine",
        (
            "Teacher CLIP boundary score "
            "vs EEG GT cosine"
        ),
        os.path.join(
            args.output_dir,
            (
                "boundary_score_"
                "vs_gt_cosine.png"
            ),
        ),
    )

    print(
        "\nAnalysis finished."
    )

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()