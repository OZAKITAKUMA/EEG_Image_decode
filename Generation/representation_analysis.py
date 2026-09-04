import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import json

def load_subject_features(feature_dir, subjects):
    """
    被験者ごとの保存済み特徴量を読み込む。

    Args:
        feature_dir:
            sub01.pth, sub02.pth ... が保存されているディレクトリ

        subjects:
            ["sub01", "sub02", ...] のような被験者名のリスト

    Returns:
    辞書型でkey:被験者番号. value:テンソル
        features:
            {
                "sub01": Tensor [200, 1024],
                "sub02": Tensor [200, 1024],
                ...
            }
    """
    features = {}

    for subject in subjects:
        feature_path = os.path.join(
            feature_dir,
            f"{subject}.pth",
        )

        if not os.path.isfile(feature_path):
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        feature = torch.load(
            feature_path,
            map_location="cpu",
        ).float()

        if feature.ndim != 2:
            raise ValueError(
                f"{subject}: expected 2-D tensor, "
                f"but got shape {tuple(feature.shape)}"
            )

        features[subject] = feature

        print(
            f"Loaded {subject}: "
            f"{tuple(feature.shape)}"
        )

    return features

def compute_subject_cosine_matrix(features, subjects):
    """
    同じ刺激に対する特徴量を被験者間で比較する。

    各被験者ペアについて、
    stimulus 0同士、stimulus 1同士、... を比較し、
    cosine similarityを全刺激で平均する。

    Returns:
        similarity_matrix:
            Tensor [num_subjects, num_subjects]
    """
    reference_shape = features[subjects[0]].shape

    for subject in subjects:
        if features[subject].shape != reference_shape:
            raise ValueError(
                f"Feature shape mismatch: "
                f"{subject}={tuple(features[subject].shape)}, "
                f"expected={tuple(reference_shape)}"
            )

    normalized_features = {
        subject: F.normalize(
            features[subject],
            dim=1,
        )
        for subject in subjects
    }

    num_subjects = len(subjects)

    similarity_matrix = torch.empty(
        num_subjects,
        num_subjects,
        dtype=torch.float32,
    )

    for i, subject_i in enumerate(subjects):
        for j, subject_j in enumerate(subjects):

            feature_i = normalized_features[subject_i]
            feature_j = normalized_features[subject_j]

            # [200, 1024] * [200, 1024]
            #        ↓ sum(dim=1)
            #       [200]
            stimulus_cosine = (
                feature_i * feature_j
            ).sum(dim=1)

            # 200刺激の平均
            similarity_matrix[i, j] = (
                stimulus_cosine.mean()
            )

    return similarity_matrix

def plot_subject_cosine_heatmap(
    similarity_matrix,
    subjects,
    output_path=None,
):
    """
    被験者間cosine similarity行列をheatmapとして表示・保存する。
    """
    matrix = similarity_matrix.cpu().numpy()

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(matrix)

    ax.set_xticks(range(len(subjects)))
    ax.set_yticks(range(len(subjects)))

    ax.set_xticklabels(subjects)
    ax.set_yticklabels(subjects)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    ax.set_title(
        "Cross-subject cosine similarity"
    )

    for i in range(len(subjects)):
        for j in range(len(subjects)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Mean cosine similarity",
    )

    fig.tight_layout()

    if output_path is not None:
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved heatmap: {output_path}"
        )

    plt.show()

    return fig


######## RSA(Representational Similarity Analysis) ########

def compute_rdm(features):
    """
    features:
        [N, D]
        N = 刺激数
        D = 特徴次元

    returns:
        rdm: [N, N]
        各要素 rdm[i, j] は
        刺激iと刺激jの二乗ユークリッド距離
    """
    # 各刺激ベクトルのL2ノルムの二乗
    # [N, D] -> [N]
    squared_norm = (features ** 2).sum(dim = 1)

    # 全刺激ペアの二乗ユークリッド距離
    #
    # ||xi - xj||²
    # = ||xi||² + ||xj||² - 2 xi^T xj
    #
    # [N, 1] + [1, N] - [N, N]
    #                  ↓
    #               [N, N]
    rdm = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2 * (features @ features.T)
    )

    return rdm

def compute_cosine_rdm(features):
    """
    features:
        Tensor [N, D]

    returns:
        rdm:
            Tensor [N, N]
            cosine distance = 1 - cosine similarity
    """

    # 各刺激特徴をL2正規化
    # [N, D] -> [N, D]
    normalized_features = F.normalize(
        features,
        dim=1,
    )

    # 全刺激ペアのcosine similarity
    # [N, D] @ [D, N]
    #       ↓
    #    [N, N]
    cosine_similarity = (
        normalized_features
        @ normalized_features.T
    )

    # similarity -> distance
    rdm = 1.0 - cosine_similarity

    return rdm

def compute_rsa_similarity(rdm_a, rdm_b):
    """
    rdm_a:
        被験者AのRDM [N, N]

    rdm_b:
        被験者BのRDM [N, N]

    returns:
        rsa_similarity:
            RDM同士のPearson相関（スカラー）
    """

    # RDMのshapeが同じか確認
    if rdm_a.shape != rdm_b.shape:
        raise ValueError(
            f"RDM shape mismatch: "
            f"{rdm_a.shape} vs {rdm_b.shape}"
        )

    N = rdm_a.shape[0]

    # 対角線を除いた上三角部分のindexを取得
    # N=200なら19900組
    upper_indices = torch.triu_indices(
        N,
        N,
        offset=1,
    )

    # [N, N] -> [N(N-1)/2]
    vec_a = rdm_a[
        upper_indices[0],
        upper_indices[1],
    ]

    vec_b = rdm_b[
        upper_indices[0],
        upper_indices[1],
    ]

    # Pearson相関のため、それぞれ平均を0にする
    vec_a_centered = vec_a - vec_a.mean()
    vec_b_centered = vec_b - vec_b.mean()

    # Pearson correlation
    # 分子：2つのベクトルの内積
    numerator = (
        vec_a_centered
        * vec_b_centered
    ).sum()

    # 分母：それぞれのL2ノルムの積
    denominator = (
        torch.sqrt(
            (vec_a_centered ** 2).sum()
        )
        *
        torch.sqrt(
            (vec_b_centered ** 2).sum()
        )
    )

    rsa_similarity = numerator / denominator

    return rsa_similarity

def compute_subject_rsa_matrix(features, subjects):
    """
    全被験者についてRDMを作成し、
    被験者ペアごとのRSA similarityを計算する。

    Args:
        features:
            {
                "sub01": Tensor [N, D],
                "sub02": Tensor [N, D],
                ...
            }

        subjects:
            ["sub01", "sub02", ...]

    Returns:
        rsa_matrix:
            Tensor [num_subjects, num_subjects]
    """

    # まず各被験者についてRDMを作る
    rdms = {}

    for subject in subjects:
        rdms[subject] = compute_rdm(
            features[subject]
        )

        print(
            f"RDM {subject}: "
            f"{tuple(rdms[subject].shape)}"
        )

    num_subjects = len(subjects)

    rsa_matrix = torch.empty(
        num_subjects,
        num_subjects,
        dtype=torch.float32,
    )

    # 被験者ペアごとにRSA
    # 被験者Aを固定してAからFまでの被験者とRSAの要素を求める
    for i, subject_i in enumerate(subjects):
        for j, subject_j in enumerate(subjects):

            rsa_matrix[i, j] = compute_rsa_similarity(
                rdms[subject_i],
                rdms[subject_j],
            )

    return rsa_matrix

def compute_subject_cosine_rsa_matrix(
    features,
    subjects,
):
    """
    全被験者についてcosine-distance RDMを作成し、
    被験者ペアごとのRSA similarityを計算する。

    Args:
        features:
            {
                "sub01": Tensor [N, D],
                "sub02": Tensor [N, D],
                ...
            }

        subjects:
            ["sub01", "sub02", ...]

    Returns:
        rsa_matrix:
            Tensor [num_subjects, num_subjects]
    """

    rdms = {}

    # 各被験者のcosine-distance RDMを作る
    for subject in subjects:
        rdms[subject] = compute_cosine_rdm(
            features[subject]
        )

        print(
            f"Cosine RDM {subject}: "
            f"{tuple(rdms[subject].shape)}"
        )

    num_subjects = len(subjects)

    rsa_matrix = torch.empty(
        num_subjects,
        num_subjects,
        dtype=torch.float32,
    )

    # 全被験者ペアについてRDM同士をRSA
    for i, subject_i in enumerate(subjects):
        for j, subject_j in enumerate(subjects):

            rsa_matrix[i, j] = compute_rsa_similarity(
                rdms[subject_i],
                rdms[subject_j],
            )

    return rsa_matrix

def plot_subject_rsa_heatmap(
    rsa_matrix,
    subjects,
    output_path=None,
):
    """
    被験者間RSA similarityをヒートマップとして可視化する。

    Args:
        rsa_matrix:
            Tensor [num_subjects, num_subjects]

        subjects:
            ["sub01", "sub02", ...]

        output_path:
            保存先。Noneなら保存しない。
    """

    # matplotlibで扱いやすいようにNumPyへ変換
    matrix = rsa_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        range(len(subjects)),
        labels=subjects,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(subjects)),
        labels=subjects,
    )

    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    ax.set_title("RSA similarity between subjects")

    # 各マスにRSA値を表示
    for i in range(len(subjects)):
        for j in range(len(subjects)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Pearson correlation",
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig

def compute_subject_to_clip_rsa(features, clip_features, subjects,):
    clip_rdm = compute_rdm(clip_features)

    rsa_scores = []

    for subject in subjects:
        feature = features[subject]

        target_rdm = compute_rdm(feature)

        rsa_with_clip = compute_rsa_similarity(
            clip_rdm,
            target_rdm,
        )

        rsa_scores.append(rsa_with_clip)

    # listに入れたscalar Tensorを
    # [10] Tensorにまとめる
    rsa_scores = torch.stack(rsa_scores)

    return rsa_scores, clip_rdm

def compute_subject_to_clip_cosine_rsa(
    features,
    clip_features,
    subjects,
):
    """
    各被験者モデルの特徴と正解CLIP特徴について
    cosine-distance RDMを作り、
    RDM同士のRSA similarityを計算する。

    Returns:
        rsa_scores:
            Tensor [num_subjects]

        clip_rdm:
            正解CLIPのcosine-distance RDM [N, N]
    """

    clip_rdm = compute_cosine_rdm(
        clip_features
    )

    rsa_scores = []

    for subject in subjects:
        feature = features[subject]

        target_rdm = compute_cosine_rdm(
            feature
        )

        rsa_with_clip = compute_rsa_similarity(
            clip_rdm,
            target_rdm,
        )

        rsa_scores.append(
            rsa_with_clip
        )

    rsa_scores = torch.stack(
        rsa_scores
    )

    return rsa_scores, clip_rdm



def plot_clip_rdm_heatmap(
    clip_rdm,
    output_path=None,
):
    """
    正解CLIP特徴から作成したRDMを可視化する。

    Args:
        clip_rdm:
            Tensor [N, N]
            各要素は刺激ペア間の二乗ユークリッド距離

        output_path:
            保存先。Noneなら保存しない。

    Returns:
        fig:
            matplotlibのFigure
    """

    # Tensor -> NumPy
    matrix = clip_rdm.detach().cpu().numpy()

    # FigureとAxesを作成
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    # RDMを画像として表示
    image = ax.imshow(
        matrix
    )

    # 軸ラベル
    ax.set_xlabel("Stimulus index")
    ax.set_ylabel("Stimulus index")

    # タイトル
    ax.set_title(
        "Ground-truth CLIP RDM"
    )

    # カラーバー
    fig.colorbar(
        image,
        ax=ax,
        label="Squared Euclidean distance",
    )

    # レイアウト調整
    fig.tight_layout()

    # 保存先が指定されている場合のみ保存
    if output_path is not None:

        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved CLIP RDM: {output_path}"
        )

    return fig

def plot_clip_cosine_rdm_heatmap(
    clip_rdm,
    output_path=None,
):
    """
    正解CLIP特徴のcosine-distance RDMを可視化する。

    Args:
        clip_rdm:
            Tensor [N, N]
            cosine distance = 1 - cosine similarity

        output_path:
            保存先。Noneなら保存しない。
    """

    matrix = clip_rdm.detach().cpu().numpy()

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=2.0,
    )

    ax.set_xlabel("Stimulus index")
    ax.set_ylabel("Stimulus index")

    ax.set_title(
        "Ground-truth CLIP cosine-distance RDM"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Cosine distance",
    )

    fig.tight_layout()

    if output_path is not None:
        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved CLIP cosine RDM: {output_path}"
        )

    return fig

def load_semantic_groups(group_path):
    with open(group_path, "r") as f:
        semantic_groups = json.load(f)

    return semantic_groups

def make_grouped_indices(semantic_groups):

    grouped_indices = []

    for group_name, folder_ids in semantic_groups.items():

        for folder_id in folder_ids:

            grouped_indices.append(
                folder_id - 1
            )

    return grouped_indices

def plot_grouped_rdm_heatmap(
    rdm,
    semantic_groups,
    title,
    output_path=None,
    distance_label="Squared Euclidean distance",
):
    """
    RDMを意味カテゴリ順に並べ替えて可視化する。

    Args:
        rdm:
            Tensor [N, N]

        semantic_groups:
            {
                "mammals": [2, 12, ...],
                "birds": [58, 69, ...],
                ...
            }

        title:
            図のタイトル

        output_path:
            保存先。Noneなら保存しない。

    Returns:
        fig:
            matplotlib Figure

        reordered_rdm:
            意味カテゴリ順へ並べ替えたRDM [N, N]
    """

    grouped_indices = make_grouped_indices(
        semantic_groups
    )

    # 200刺激すべてが1回ずつ含まれているか確認
    if len(grouped_indices) != rdm.shape[0]:
        raise ValueError(
            f"Number of grouped indices ({len(grouped_indices)}) "
            f"does not match RDM size ({rdm.shape[0]})"
        )

    if len(set(grouped_indices)) != len(grouped_indices):
        raise ValueError(
            "Duplicate stimulus indices found in semantic_groups."
        )

    # list -> Tensor index
    index = torch.tensor(
        grouped_indices,
        dtype=torch.long,
        device=rdm.device,
    )

    # 行と列を同じ意味カテゴリ順へ並べ替える
    # [N, N] -> [N, N]
    reordered_rdm = (
        rdm
        .index_select(0, index)
        .index_select(1, index)
    )

    matrix = reordered_rdm.detach().cpu().numpy()

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        matrix
    )

    # 各カテゴリ名を置く中心位置
    group_centers = []
    group_names = []

    start = 0

    for group_name, folder_ids in semantic_groups.items():
        group_size = len(folder_ids)
        end = start + group_size

        center = (start + end - 1) / 2

        group_centers.append(center)
        group_names.append(group_name)

        # カテゴリ境界線
        if end < len(grouped_indices):
            boundary = end - 0.5

            ax.axhline(
                boundary,
                linewidth=0.8,
            )

            ax.axvline(
                boundary,
                linewidth=0.8,
            )

        start = end

    ax.set_xticks(group_centers)
    ax.set_yticks(group_centers)

    ax.set_xticklabels(
        group_names,
        rotation=90,
    )

    ax.set_yticklabels(
        group_names,
    )

    ax.set_xlabel("Semantic group")
    ax.set_ylabel("Semantic group")
    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label=distance_label,
    )

    fig.tight_layout()

    if output_path is not None:
        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved grouped RDM: {output_path}"
        )

    return fig, reordered_rdm