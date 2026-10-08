# Default settings shared by experiment launchers.

DEFAULT_CONFIG = {
    "subjects": (
        "sub-01 "
        # "sub-02 "
        # "sub-03 "
        # "sub-04 "
        # "sub-05 "
        # "sub-06 "
        # "sub-07 "
        # "sub-08 "
        # "sub-09 "
        # "sub-10"
    ),
    "image_encoder": "clip",
    "feature_space": "clip",
    "method": "baseline",
    "feature_transform": "full",
    "svd_centering": "global",
    "svd_component_mode": "keep_top_rank",
    "svd_rank": None,
    "svd_remove_count": None,
    "svd_remove_start": None,

    "encoder_epochs": 100,
    "batch_size": 64,
    "lr_encoder": 3e-4,

    "rsa_weight": 0.0,
    "rsa_loss_type": "pearson",

    "svd_mse_weighting": False,
    "svd_bottom_k": 64,
    "svd_bottom_weight": 4.0,

    "checkpoint_criterion": "val_rsa_pearson",

    "val_ratio": 0.1,
    "patience": 50,
    "save_interval": 10,

    "avg_trials": True,
    "seed": 42,
    "gpu": "cuda:0",
}


IMAGE_ENCODER_CHOICES = [
    "clip",
    "dinov3",
    "siglip2",
]

FEATURE_TRANSFORM_CHOICES = [
    "full",
    "svd",
]

SVD_CENTERING_CHOICES = [
    "global",
    "class",
]

SVD_COMPONENT_MODE_CHOICES = [
    "keep_top_rank",
    "remove_top",
    "remove_middle",
    "remove_bottom",
    "remove_range",
]

FEATURE_SOURCE_FILES = {
    "clip": "ViT-H-14_features_train.pt",
    "dinov3": "DINOv3-ViTL16_features_train.pt",
    "siglip2": "SigLIP2-Large-Patch16-256_features_train.pt",
}

from pathlib import Path


RSA_LOSS_TYPE_CHOICES = [
    "pearson",
    "rdm_mse",
]

CHECKPOINT_CRITERION_CHOICES = [
    "val_base_loss",
    "val_total_loss",
    "val_rsa_pearson",
    "val_rdm_mse",
    "val_top1_accuracy",
    "val_top5_accuracy",
]

EXPERIMENT_TYPE_CHOICES = [
    "image_encoder_compare",
    "svd_compare",
    "svd_bottom_weight_compare",
    "loss_compare",
    "subject_compare",
]


EXPERIMENT_PROFILES = {
    "image_encoder_compare": [
        "image_encoder",
        "checkpoint_criterion",
    ],

    "svd_compare": [
        "image_encoder",
        "feature_transform",
        "checkpoint_criterion",
    ],

    "svd_bottom_weight_compare": [
        "svd_bottom_k",
        "svd_bottom_weight",
        "checkpoint_criterion",
    ],

    "loss_compare": [
        "rsa_loss_type",
        "rsa_weight",
        "checkpoint_criterion",
    ],

    "subject_compare": [
        "subjects",
    ],
}


def format_weight_for_path(weight):
    """Convert a numeric weight to a filename-safe string."""
    return f"{weight:g}".replace(".", "p")

def build_svd_cache_tag(config):
    """Build the suffix used by an SVD-transformed feature cache."""

    centering = config["svd_centering"]
    mode = config["svd_component_mode"]

    if mode == "keep_top_rank":
        rank = config["svd_rank"]

        # Keep the old global-SVD filename for backward compatibility.
        if centering == "global":
            return f"svd{rank}"

        return f"svd_{centering}_keep{rank}"

    remove_count = config["svd_remove_count"]

    if mode == "remove_range":
        start = config["svd_remove_start"]
        end = start + remove_count - 1
        return f"svd_{centering}_remove_pc{start}-{end}"

    position = mode.removeprefix("remove_")

    return (
        f"svd_{centering}_"
        f"remove_{position}{remove_count}"
    )


def build_svd_path_name(config):
    """Build the directory name for one SVD feature variant."""

    centering = config["svd_centering"]
    mode = config["svd_component_mode"]

    if mode == "keep_top_rank":
        rank = config["svd_rank"]

        # Preserve the directory layout of existing global-rank experiments.
        if centering == "global":
            return f"svd_{rank}"

        return f"svd_{centering}_keep_{rank}"

    remove_count = config["svd_remove_count"]

    if mode == "remove_range":
        start = config["svd_remove_start"]
        end = start + remove_count - 1
        return f"svd_{centering}_remove_pc_{start}-{end}"

    position = mode.removeprefix("remove_")

    return (
        f"svd_{centering}_"
        f"remove_{position}_{remove_count}"
    )


def build_feature_filename(config):
    """Return the training feature-cache filename for this experiment."""

    image_encoder = config["image_encoder"]

    source_filename = FEATURE_SOURCE_FILES[image_encoder]

    if config["feature_transform"] != "svd":
        return source_filename

    source_path = Path(source_filename)
    svd_tag = build_svd_cache_tag(config)

    return (
        f"{source_path.stem}_"
        f"{svd_tag}"
        f"{source_path.suffix}"
    )

def build_loss_name(config):
    """Build a directory name representing the training loss."""

    if config.get("svd_mse_weighting", False):
        bottom_k = config["svd_bottom_k"]
        bottom_weight = format_weight_for_path(
            config["svd_bottom_weight"]
        )
        return (
            "mse_contrastive_"
            f"svd_bottom{bottom_k}_"
            f"w{bottom_weight}"
        )

    rsa_weight = config["rsa_weight"]

    # RSAを使わない場合
    if rsa_weight == 0:
        return "mse_contrastive"

    # RSAを使う場合
    rsa_loss_type = config["rsa_loss_type"]
    weight_str = format_weight_for_path(rsa_weight)

    return (
        f"mse_contrastive_"
        f"rsa_{rsa_loss_type}_"
        f"w{weight_str}"
    )

def build_experiment_path(config):
    """Build the directory hierarchy for one experiment."""

    image_encoder = config["image_encoder"]
    method = config["method"]
    loss_name = build_loss_name(config)
    checkpoint_criterion = config["checkpoint_criterion"]

    if config["feature_transform"] == "svd":
        feature_variant = build_svd_path_name(config)
    else:
        feature_variant = "full"

    return (
        f"{method}/"
        f"{image_encoder}/"
        f"{feature_variant}/"
        f"{loss_name}/"
        f"{checkpoint_criterion}"
    )