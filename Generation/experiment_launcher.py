import os
import subprocess
import argparse

from experiment_config import (
    DEFAULT_CONFIG,
    IMAGE_ENCODER_CHOICES,
    FEATURE_TRANSFORM_CHOICES,
    SVD_CENTERING_CHOICES,
    SVD_COMPONENT_MODE_CHOICES,
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
        mode = config["svd_component_mode"]

        if mode == "keep_top_rank":
            if config["svd_rank"] is None:
                errors.append(
                    "keep_top_rankではsvd_rankを指定してください。"
                )
            elif not 1 <= config["svd_rank"] <= 1024:
                errors.append(
                    "SVD rankは1以上1024以下にしてください。"
                )
        else:
            remove_count = config["svd_remove_count"]

            if remove_count is None:
                errors.append(
                    "成分削除では削除数を指定してください。"
                )
            elif not 1 <= remove_count < 1024:
                errors.append(
                    "削除数は1以上1023以下にしてください。"
                )

            if mode == "remove_range":
                start = config["svd_remove_start"]

                if start is None:
                    errors.append(
                        "remove_rangeでは削除開始PCを指定してください。"
                    )
                elif not 1 <= start <= 1024:
                    errors.append(
                        "削除開始PCは1以上1024以下にしてください。"
                    )
                elif (
                    remove_count is not None
                    and start + remove_count - 1 > 1024
                ):
                    errors.append(
                        "削除範囲がPC1024を超えています。"
                    )

        if config["image_encoder"] not in FEATURE_SOURCE_FILES:
            errors.append(
                "選択したImage Encoderの元特徴ファイルが"
                "FEATURE_SOURCE_FILESに登録されていません。"
            )

    if config["svd_mse_weighting"]:
        if config["svd_bottom_k"] < 1 or config["svd_bottom_k"] > 1024:
            errors.append(
                "svd_bottom_kは1以上1024以下にしてください。"
            )

        if config["svd_bottom_weight"] <= 0:
            errors.append(
                "svd_bottom_weightは0より大きくしてください。"
            )

        if config["image_encoder"] != "clip":
            errors.append(
                "SVD bottom weighting experimentは"
                "現在CLIPのみを対象とします。"
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
                config["svd_centering"] = ask_choice(
                    "SVDの中心化方法を選択してください。",
                    SVD_CENTERING_CHOICES,
                    config["svd_centering"],
                )

                config["svd_component_mode"] = ask_choice(
                    "主成分の扱いを選択してください。",
                    SVD_COMPONENT_MODE_CHOICES,
                    config["svd_component_mode"],
                )

                if config["svd_component_mode"] == "keep_top_rank":
                    config["svd_rank"] = ask_int(
                        "保持するSVD rankを入力してください。",
                        960,
                    )
                    config["svd_remove_count"] = None
                    config["svd_remove_start"] = None

                else:
                    config["svd_rank"] = None
                    config["svd_remove_count"] = ask_int(
                        "削除する主成分数を入力してください。",
                        10,
                    )

                    if config["svd_component_mode"] == "remove_range":
                        config["svd_remove_start"] = ask_int(
                            "削除を開始するPC番号（1始まり）を入力してください。",
                            1,
                        )
                    else:
                        config["svd_remove_start"] = None

            else:
                config["svd_rank"] = None
                config["svd_remove_count"] = None
                config["svd_remove_start"] = None
        
        elif question == "svd_bottom_k":
            config["svd_mse_weighting"] = True
            config["feature_transform"] = "full"
            config["svd_bottom_k"] = ask_int(
                "強調する下位SVD成分数を入力してください。",
                config["svd_bottom_k"],
            )

        elif question == "svd_bottom_weight":
            config["svd_mse_weighting"] = True
            config["feature_transform"] = "full"
            config["svd_bottom_weight"] = ask_float(
                "下位SVD成分のMSE重みを入力してください。",
                config["svd_bottom_weight"],
            )

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

            svd_command = [
                "python",
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "create_svd_clip_features.py",
                ),
                "--train_features_path",
                source_features_path,
                "--output_path",
                config["train_features_path"],
                "--centering",
                config["svd_centering"],
                "--component_mode",
                config["svd_component_mode"],
                "--seed",
                str(config["seed"]),
                "--val_ratio",
                str(config["val_ratio"]),
            ]

            if config["svd_component_mode"] == "keep_top_rank":
                svd_command.extend([
                    "--rank",
                    str(config["svd_rank"]),
                ])
            else:
                svd_command.extend([
                    "--remove_count",
                    str(config["svd_remove_count"]),
                ])

                if config["svd_component_mode"] == "remove_range":
                    svd_command.extend([
                        "--remove_start",
                        str(config["svd_remove_start"]),
                    ])

            subprocess.run(
                svd_command,
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

        "SVD_MSE_WEIGHTING": str(
            config["svd_mse_weighting"]
        ).lower(),
        "SVD_BOTTOM_K": str(config["svd_bottom_k"]),
        "SVD_BOTTOM_WEIGHT": str(
            config["svd_bottom_weight"]
        ),

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