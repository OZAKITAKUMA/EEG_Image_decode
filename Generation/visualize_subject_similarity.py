import argparse
import os
import torch

from representation_analysis import (
    load_subject_features,
    compute_subject_cosine_matrix,
    plot_subject_cosine_heatmap,
    compute_subject_rsa_matrix,
    plot_subject_rsa_heatmap,
    compute_subject_to_clip_rsa,
    plot_clip_rdm_heatmap,
    load_semantic_groups,
    plot_grouped_rdm_heatmap,
    compute_subject_to_clip_cosine_rsa,
    compute_subject_cosine_rsa_matrix,
    plot_clip_cosine_rdm_heatmap,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cross-subject EEG feature similarity."
    )

    parser.add_argument(
        "--feature_dir",
        type=str,
        default="./outputs/benchmark_encoder_only/clip/test_feature",
        help="Directory containing sub01.pth ... sub10.pth",
    )

    parser.add_argument(
        "--clip_feature_path",
        type=str,
        default="../features/ViT-H-14_features_test.pt",
    )

    parser.add_argument(
            "--group_path",
            type=str,
            default="./semantic_groups.json",
        )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/benchmark_encoder_only/clip/"
                "representation_analysis/",
        help="Path to save the cosine similarity heatmap",
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[
            "sub01",
            "sub02",
            "sub03",
            "sub04",
            "sub05",
            "sub06",
            "sub07",
            "sub08",
            "sub09",
            "sub10",
        ],
        help="Subjects to include in the analysis",
    )

    args = parser.parse_args()

    semantic_groups = load_semantic_groups(args.group_path)
    
    features = load_subject_features(
        args.feature_dir,
        args.subjects,
    )
    
    clip_feature_data = torch.load(
                args.clip_feature_path,
                map_location="cpu",
            )
    clip_feature = clip_feature_data["img_features"].float()

    similarity_matrix = compute_subject_cosine_matrix(
        features,
        args.subjects,
    )

    rsa_matrix = compute_subject_rsa_matrix(
        features,
        args.subjects,
    )

    cosine_rsa_matrix = compute_subject_cosine_rsa_matrix(
        features,
        args.subjects,
    )

    rsa_scores, clip_rdm = compute_subject_to_clip_rsa(
        features, 
        clip_feature, 
        args.subjects,
    )

    cosine_rsa_scores, cosine_clip_rdm = (
        compute_subject_to_clip_cosine_rsa(
            features,
            clip_feature,
            args.subjects,
        )
    )

    print(
    "\nRSA similarity to ground-truth CLIP RDM "
    "(squared Euclidean):"
    )

    for subject, score in zip(args.subjects, rsa_scores,):
        print(f"{subject}: {score.item():.4f}")

    print(f"Mean: {rsa_scores.mean().item():.4f}")
    print(f"Min:  {rsa_scores.min().item():.4f}")
    print(f"Max:  {rsa_scores.max().item():.4f}")

    print(
        "\nRSA similarity to ground-truth CLIP RDM "
        "(cosine distance):"
    )

    for subject, score in zip(
        args.subjects,
        cosine_rsa_scores,
    ):
        print(
            f"{subject}: {score.item():.4f}"
        )

    print(
        f"Mean: {cosine_rsa_scores.mean().item():.4f}"
    )
    print(
        f"Min:  {cosine_rsa_scores.min().item():.4f}"
    )
    print(
        f"Max:  {cosine_rsa_scores.max().item():.4f}"
    )
    print("\n===== Euclidean vs Cosine RDM =====")

    print(
        f"Euclidean mean RSA: "
        f"{rsa_scores.mean().item():.4f}"
    )

    print(
        f"Cosine mean RSA:    "
        f"{cosine_rsa_scores.mean().item():.4f}"
    )


    plot_subject_cosine_heatmap(
        similarity_matrix,
        args.subjects,
        output_path=os.path.join(args.output_path, "subject_cosine_heatmap.png"),
    )

    plot_subject_rsa_heatmap(
        rsa_matrix,
        args.subjects,
        output_path=os.path.join(args.output_path, "subject_rsa_heatmap.png"),
    )

    plot_subject_rsa_heatmap(
        cosine_rsa_matrix,
        args.subjects,
        output_path=os.path.join(
            args.output_path,
            "subject_cosine_rsa_heatmap.png",
        ),
    )
    
    plot_clip_rdm_heatmap(
        clip_rdm,
        output_path=os.path.join(args.output_path, "clip_rdm.png"),
    )

    plot_clip_cosine_rdm_heatmap(
        cosine_clip_rdm,
        output_path=os.path.join(
            args.output_path,
            "clip_cosine_rdm.png",
        ),
    )

    plot_grouped_rdm_heatmap(
        clip_rdm,
        semantic_groups,
        title="Ground-truth CLIP RDM grouped by semantic category",
        output_path=os.path.join(
            args.output_path,
            "clip_rdm_grouped.png",
        ),
    )

    plot_grouped_rdm_heatmap(
    cosine_clip_rdm,
    semantic_groups,
    title="Ground-truth CLIP cosine-distance RDM grouped by semantic category",
    output_path=os.path.join(
        args.output_path,
        "clip_cosine_rdm_grouped.png",
    ),
    distance_label="Cosine distance",
)
    
if __name__ == "__main__":
    main()

# このファイルを実行するときは　Generationディレクトリに入ってから