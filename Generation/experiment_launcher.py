import os
import subprocess
import argparse

from experiment_config import (
    DEFAULT_CONFIG,
    IMAGE_ENCODER_CHOICES,
    FEATURE_TRANSFORM_CHOICES,
    CHECKPOINT_CRITERION_CHOICES,
    RSA_LOSS_TYPE_CHOICES,
    EXPERIMENT_TYPE_CHOICES,
    EXPERIMENT_PROFILES,
    FEATURE_SOURCE_FILES,
    build_experiment_path,
    build_feature_filename,
)


def ask_choice(message, choices, default):
    print(f"\n{message}")

    for i, choice in enumerate(choices, start=1):
        default_mark = " [default]" if choice == default else ""
        print(f"[{i}] {choice}{default_mark}")

    while True:
        value = input("> ").strip()

        if value == "":
            return default

        if value.isdigit():
            index = int(value) - 1

            if 0 <= index < len(choices):
                return choices[index]

        print("入力が正しくありません。もう一度入力してください。")


def ask_int(message, default):
    while True:
        value = input(
            f"\n{message} [default: {default}]\n> "
        ).strip()

        if value == "":
            return default

        try:
            return int(value)
        except ValueError:
            print("整数を入力してください。")

def ask_float(message, default):
    while True:
        value = input(
            f"\n{message} [default: {default}]\n> "
        ).strip()

        if value == "":
            return default

        try:
            return float(value)
        except ValueError:
            print("数値を入力してください。")


def ask_text(message, default):
    value = input(
        f"\n{message} [default: {default}]\n> "
    ).strip()

    if value == "":
        return default

    return value

def validate_config(config):
    """Check experiment settings before execution."""

    errors = []
    warnings = []

    if config["feature_transform"] == "svd":
        if config["svd_rank"] is None:
            errors.append(
                "SVDを使う場合はsvd_rankを指定してください。"
            )

        elif config["svd_rank"] <= 0:
            errors.append(
                "SVD rankは1以上にしてください。"
            )

        if config["image_encoder"] not in FEATURE_SOURCE_FILES:
            errors.append(
                "選択したImage Encoderの元特徴ファイルが"
                "FEATURE_SOURCE_FILESに登録されていません。"
            )

    if config["rsa_weight"] < 0:
        errors.append(
            "RSA weightは0以上にしてください。"
        )

    if (
        config["rsa_weight"] == 0
        and config["rsa_loss_type"] != "pearson"
    ):
        warnings.append(
            "RSA weightが0なので、"
            "rsa_loss_typeは学習Lossには影響しません。"
        )

    if (
        config["rsa_weight"] == 0
        and config["checkpoint_criterion"] == "val_total_loss"
    ):
        warnings.append(
            "RSA weightが0なので、val_total_lossと"
            "val_base_lossは同じ値になります。"
        )

    return errors, warnings

def main():
    parser = argparse.ArgumentParser(
        description="EEG experiment launcher"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the final experiment configuration without running training.",
    )
    launcher_args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    print("========================================")
    print(" EEG Experiment Launcher")
    print("========================================")

    experiment_type = ask_choice(
        "実験タイプを選択してください。",
        EXPERIMENT_TYPE_CHOICES,
        "image_encoder_compare",
    )

    config["experiment_type"] = experiment_type

    questions = EXPERIMENT_PROFILES[experiment_type]

    for question in questions:

        if question == "image_encoder":
            config["image_encoder"] = ask_choice(
                "Image Encoderを選択してください。",
                IMAGE_ENCODER_CHOICES,
                config["image_encoder"],
            )

        elif question == "feature_transform":
            config["feature_transform"] = ask_choice(
                "特徴変換を選択してください。",
                FEATURE_TRANSFORM_CHOICES,
                config["feature_transform"],
            )

            if config["feature_transform"] == "svd":
                config["svd_rank"] = ask_int(
                    "SVD rankを入力してください。",
                    960,
                )
            else:
                config["svd_rank"] = None
        
        elif question == "rsa_loss_type":
            config["rsa_loss_type"] = ask_choice(
                "RSA loss typeを選択してください。",
                RSA_LOSS_TYPE_CHOICES,
                config["rsa_loss_type"],
            )

        elif question == "rsa_weight":
            config["rsa_weight"] = ask_float(
                "RSA weightを入力してください。",
                config["rsa_weight"],
            )

        elif question == "checkpoint_criterion":
            config["checkpoint_criterion"] = ask_choice(
                "Checkpointの保存基準を選択してください。",
                CHECKPOINT_CRITERION_CHOICES,
                config["checkpoint_criterion"],
            )

        elif question == "subjects":
            config["subjects"] = ask_text(
                "対象被験者を入力してください。"
                " 例: sub-01 sub-02",
                config["subjects"],
            )

    errors, warnings = validate_config(config)

    if errors:
        print("\n===== Configuration Error =====")
        for message in errors:
            print(f"[ERROR] {message}")
        return

    if warnings:
        print("\n===== Configuration Warning =====")
        for message in warnings:
            print(f"[WARN] {message}")
        
    experiment_path = build_experiment_path(config)

    config["experiment_path"] = experiment_path
    config["output_dir"] = f"./outputs/{experiment_path}"
    config["model_save_dir"] = f"./models/{experiment_path}"

    if config["feature_transform"] == "svd":
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        feature_filename = build_feature_filename(config)

        config["train_features_path"] = os.path.join(
            repo_root,
            "features",
            feature_filename,
        )
    else:
        config["train_features_path"] = ""
    
    print("\n========================================")
    print(" Experiment Configuration")
    print("========================================")

    for key, value in config.items():
        print(f"{key:24s}: {value}")

    print("========================================")
    
    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    benchmark_script = os.path.join(
        script_dir,
        "benchmark_encoder_only_within_subject.sh",
    )

    if launcher_args.dry_run:
        print("\n========================================")
        print(" DRY RUN")
        print("========================================")
        print(f"Benchmark script : {benchmark_script}")
        print(f"Output directory : {config['output_dir']}")
        print(f"Model directory  : {config['model_save_dir']}")
        print("Training was not started.")
        return
    
    answer = input(
        "\nこの設定で実験を開始しますか？ [y/N]\n> "
    ).strip().lower()

    if answer != "y":
        print("実験をキャンセルしました。")
        return

    if config["feature_transform"] == "svd":
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        source_features_path = os.path.join(
            repo_root,
            "features",
            FEATURE_SOURCE_FILES[config["image_encoder"]],
        )

        if not os.path.exists(source_features_path):
            raise FileNotFoundError(
                f"Source feature cache not found: {source_features_path}"
            )
        
        if not os.path.exists(config["train_features_path"]):
            
            print(
                "\nSVD feature cacheが見つからないため生成します:"
            )
            print(f"  {config['train_features_path']}")

            subprocess.run(
                [
                    "python",
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "create_svd_clip_features.py",
                    ),
                    "--train_features_path",
                    source_features_path,
                    "--output_dir",
                    os.path.dirname(config["train_features_path"]),
                    "--ranks",
                    str(config["svd_rank"]),
                    "--seed",
                    str(config["seed"]),
                    "--val_ratio",
                    str(config["val_ratio"]),
                ],
                check=True,
            )

    print("\n実験を開始します。")

    env = os.environ.copy()

    env.update({
        "SUBJECTS": str(config["subjects"]),
        "IMAGE_ENCODER": str(config["image_encoder"]),
        "FEATURE_SPACE": str(config["feature_space"]),
        "TRAIN_FEATURES_PATH": str(
            config["train_features_path"]
        ),

        "ENCODER_EPOCHS": str(config["encoder_epochs"]),
        "BATCH_SIZE": str(config["batch_size"]),
        "LR_ENCODER": str(config["lr_encoder"]),

        "RSA_WEIGHT": str(config["rsa_weight"]),
        "RSA_LOSS_TYPE": str(config["rsa_loss_type"]),
        "CHECKPOINT_CRITERION": str(
            config["checkpoint_criterion"]
        ),

        "VAL_RATIO": str(config["val_ratio"]),
        "PATIENCE": str(config["patience"]),
        "SAVE_INTERVAL": str(config["save_interval"]),

        "AVG_SIGNAL_TRAINING": str(
            config["avg_trials"]
        ).lower(),

        "SEED": str(config["seed"]),
        "GPU": str(config["gpu"]),

        "METHOD": str(config["method"]),
        "EXPERIMENT_TYPE": str(
            config["experiment_type"]
        ),

        "OUTPUT_DIR": str(config["output_dir"]),
        "MODEL_SAVE_DIR": str(
            config["model_save_dir"]
        ),
    })

    subprocess.run(
        ["bash", benchmark_script],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()