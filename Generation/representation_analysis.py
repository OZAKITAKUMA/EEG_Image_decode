import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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