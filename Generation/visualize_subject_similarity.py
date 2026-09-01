import argparse
import os

from representation_analysis import (
    load_subject_features,
    compute_subject_cosine_matrix,
    plot_subject_cosine_heatmap,
    compute_subject_rsa_matrix,
    plot_subject_rsa_heatmap,
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

    features = load_subject_features(
        args.feature_dir,
        args.subjects,
    )

    similarity_matrix = compute_subject_cosine_matrix(
        features,
        args.subjects,
    )

    rsa_matrix = compute_subject_rsa_matrix(
        features,
        args.subjects,
    )


    print("\nSubject cosine similarity matrix:")
    print(similarity_matrix)

    print(rsa_matrix)
    print(rsa_matrix.shape)


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

if __name__ == "__main__":
    main()

# このファイルを実行するときは　Generationディレクトリに入ってから