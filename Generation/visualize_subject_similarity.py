import argparse
import os

from representation_analysis import (
    load_subject_features,
    compute_subject_cosine_matrix,
    plot_subject_cosine_heatmap,
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
                "representation_analysis/"
                "subject_cosine_heatmap.png",
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

    print("\nSubject cosine similarity matrix:")
    print(similarity_matrix)

    plot_subject_cosine_heatmap(
        similarity_matrix,
        args.subjects,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
