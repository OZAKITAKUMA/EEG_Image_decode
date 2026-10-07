#!/usr/bin/env python3
"""Grounding DINO + SAMで前景を残し、背景黒画像とCLIP教師特徴を作る。"""

import argparse
import csv
import gc
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path("/home/moepy/ozakitakuma/data_image/training_images")
DEFAULT_MASKED = Path("/home/moepy/ozakitakuma/data_image/training_images_sam_black")
DEFAULT_SOURCE = REPO_ROOT / "features" / "ViT-H-14_features_train.pt"
DEFAULT_OUTPUT = REPO_ROOT / "features" / "ViT-H-14_features_train_sam_black.pt"

N_CLASSES = 1654
CONDITIONS_PER_CLASS = 10
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(root):
    root = Path(root)
    paths = []

    for class_dir in sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
    ):
        paths.extend(
            sorted(
                path
                for path in class_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTS
                )
            )
        )

    return paths


def class_name_from_path(image_path):
    """THINGSのクラスフォルダ名からGrounding DINO用の語を作る。"""
    folder_name = image_path.parent.name

    if "_" in folder_name:
        class_name = folder_name.split("_", 1)[1]
    else:
        class_name = folder_name

    return class_name.replace("_", " ").strip()


def load_grounding_dino(model_name, device):
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    print(
        f"Loading Grounding DINO: "
        f"{model_name} on {device}"
    )

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(model_name)
        .to(device)
    )

    model.eval()

    return model, processor


def load_sam(model_name, device):
    from transformers import SamModel, SamProcessor

    print(
        f"Loading SAM: "
        f"{model_name} on {device}"
    )

    processor = SamProcessor.from_pretrained(
        model_name
    )

    model = (
        SamModel
        .from_pretrained(model_name)
        .to(device)
    )

    model.eval()

    return model, processor


def expand_box(
    box,
    width,
    height,
    padding_ratio,
):
    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    ]


@torch.no_grad()
def detect_object_box(
    image,
    class_name,
    model,
    processor,
    device,
    threshold,
    text_threshold,
    box_padding,
):
    """
    クラス名をtext promptとしてGrounding DINOでbboxを取得する。
    複数候補がある場合はconfidence最大のbboxを使う。
    """
    text_labels = [[class_name]]

    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    results = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[
                (
                    image.height,
                    image.width,
                )
            ],
        )[0]
    )

    boxes = results["boxes"]
    scores = results["scores"]

    if len(boxes) == 0:
        return None, None

    best_index = int(
        torch.argmax(scores).item()
    )

    box = boxes[
        best_index
    ].detach().cpu().tolist()

    box = expand_box(
        box,
        width=image.width,
        height=image.height,
        padding_ratio=box_padding,
    )

    return (
        box,
        float(
            scores[
                best_index
            ].detach().cpu().item()
        ),
    )


def mask_bounding_box(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]


def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def mask_box_coverage(mask, box):
    x1, y1, x2, y2 = box

    box_area = max(
        1.0,
        (x2 - x1) * (y2 - y1),
    )

    return float(mask.sum()) / box_area


def rectangle_mask(image, box):
    x1, y1, x2, y2 = box

    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(image.width, int(np.ceil(x2)))
    y2 = min(image.height, int(np.ceil(y2)))

    mask = np.zeros(
        (image.height, image.width),
        dtype=bool,
    )
    mask[y1:y2, x1:x2] = True

    return mask


@torch.no_grad()
def segment_center_with_sam(
    image,
    model,
    processor,
    device,
):
    """
    旧方式。画像中心のpositive pointからSAMの3候補を出し、
    predicted IoU最大のmaskを返す。
    """
    inputs = processor(
        images=image,
        input_points=[[[image.width / 2.0, image.height / 2.0]]],
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    masks = (
        processor
        .image_processor
        .post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        .reshape(
            -1,
            image.height,
            image.width,
        )
        .bool()
    )

    scores = (
        outputs.iou_scores
        .detach()
        .cpu()
        .reshape(-1)
    )

    best = int(
        torch.argmax(scores).item()
    )

    mask = masks[best].numpy()

    return (
        mask,
        float(scores[best].item()),
    )


@torch.no_grad()
def segment_box_with_sam(
    image,
    box,
    model,
    processor,
    device,
):
    """
    Grounding DINOのbboxをSAMへ渡す。
    3候補のうち、best scoreから0.05以内なら
    面積が最大のmaskを優先して、細切れmaskを避ける。
    """
    inputs = processor(
        images=image,
        input_boxes=[[box]],
        return_tensors="pt",
    ).to(device)

    outputs = model(
        **inputs,
        multimask_output=True,
    )

    masks = (
        processor
        .image_processor
        .post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        .reshape(
            -1,
            image.height,
            image.width,
        )
        .bool()
    )

    scores = (
        outputs.iou_scores
        .detach()
        .cpu()
        .reshape(-1)
    )

    best_score = float(
        scores.max().item()
    )

    candidate_indices = [
        index
        for index, score in enumerate(scores.tolist())
        if score >= best_score - 0.05
    ]

    best = max(
        candidate_indices,
        key=lambda index: int(masks[index].sum().item()),
    )

    mask = masks[best].numpy()

    return (
        mask,
        float(scores[best].item()),
        float(mask.mean()),
    )



def apply_black_background(
    image,
    mask,
):
    array = np.asarray(
        image.convert("RGB")
    )

    if mask.shape != array.shape[:2]:
        raise RuntimeError(
            "mask/image shape mismatch: "
            f"mask={mask.shape}, "
            f"image={array.shape[:2]}"
        )

    output = np.zeros_like(
        array
    )

    output[mask] = array[mask]

    return Image.fromarray(
        output
    )


def save_preview(
    original,
    masked,
    path,
):
    canvas = Image.new(
        "RGB",
        (
            original.width * 2,
            original.height,
        ),
        color=(0, 0, 0),
    )

    canvas.paste(
        original,
        (0, 0),
    )

    canvas.paste(
        masked,
        (original.width, 0),
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(path)


def create_masked_images(
    args,
    device,
):
    input_root = (
        args.input_dir.resolve()
    )

    output_root = (
        args.output_image_dir.resolve()
    )

    images = list_images(
        input_root
    )

    print(
        "Input images:",
        len(images),
    )

    if args.max_images is not None:
        images = images[
            :args.max_images
        ]

    (
        grounding_model,
        grounding_processor,
    ) = load_grounding_dino(
        args.grounding_model,
        device,
    )

    (
        sam_model,
        sam_processor,
    ) = load_sam(
        args.sam_model,
        device,
    )

    stats = []

    preview_root = Path(
        str(output_root)
        + "_preview"
    )

    no_detection_count = 0

    for index, source_path in enumerate(
        tqdm(
            images,
            desc="Grounded SAM black background",
        )
    ):
        relative = (
            source_path.relative_to(
                input_root
            )
        )

        output_path = (
            output_root
            / relative
        )

        image = (
            Image.open(
                source_path
            )
            .convert("RGB")
        )

        class_name = (
            class_name_from_path(
                source_path
            )
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            masked = (
                Image.open(
                    output_path
                )
                .convert("RGB")
            )

            detection_score = float("nan")
            sam_score = float("nan")
            area_ratio = float("nan")
            fallback = "cached"

        else:
            (
                box,
                detection_score,
            ) = detect_object_box(
                image=image,
                class_name=class_name,
                model=grounding_model,
                processor=grounding_processor,
                device=device,
                threshold=(
                    args.detection_threshold
                ),
                text_threshold=(
                    args.text_threshold
                ),
                box_padding=(
                    args.box_padding
                ),
            )

            if box is None:
                # detection失敗時は誤った部分maskを作らず元画像を残す。
                mask = np.ones(
                    (
                        image.height,
                        image.width,
                    ),
                    dtype=bool,
                )

                masked = image.copy()

                sam_score = float("nan")
                area_ratio = 1.0
                center_box_iou = float("nan")
                mask_coverage = float("nan")
                strategy = "keep_original"
                fallback = "no_detection_keep_original"
                no_detection_count += 1

            else:
                # まず、動物などでうまく働いていた旧center-point maskを試す。
                (
                    center_mask,
                    center_score,
                ) = segment_center_with_sam(
                    image=image,
                    model=sam_model,
                    processor=sam_processor,
                    device=device,
                )

                center_box_iou = box_iou(
                    mask_bounding_box(center_mask),
                    box,
                )

                if (
                    center_box_iou
                    >= args.center_mask_box_iou
                ):
                    mask = center_mask
                    sam_score = center_score
                    area_ratio = float(mask.mean())
                    mask_coverage = mask_box_coverage(
                        mask,
                        box,
                    )
                    strategy = "center_point"
                    fallback = ""

                else:
                    # center-pointが棒1本など局所部分を取った場合はbbox promptへ切り替える。
                    (
                        mask,
                        sam_score,
                        area_ratio,
                    ) = segment_box_with_sam(
                        image=image,
                        box=box,
                        model=sam_model,
                        processor=sam_processor,
                        device=device,
                    )

                    mask_coverage = (
                        mask_box_coverage(
                            mask,
                            box,
                        )
                    )

                    strategy = "box_sam"
                    fallback = ""

                    # abacusのようにSAMがbbox内の一部分しか残さない場合は、
                    # 物体を欠損させるよりbbox内を全て残す方を優先する。
                    if (
                        mask_coverage
                        < args.min_mask_box_coverage
                    ):
                        mask = rectangle_mask(
                            image,
                            box,
                        )

                        area_ratio = float(
                            mask.mean()
                        )

                        mask_coverage = (
                            mask_box_coverage(
                                mask,
                                box,
                            )
                        )

                        strategy = "box_rectangle"
                        fallback = (
                            "sparse_sam_mask_use_box"
                        )

                masked = (
                    apply_black_background(
                        image,
                        mask,
                    )
                )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_kwargs = {}

            if (
                output_path
                .suffix
                .lower()
                in (
                    ".jpg",
                    ".jpeg",
                )
            ):
                save_kwargs[
                    "quality"
                ] = 95

            masked.save(
                output_path,
                **save_kwargs,
            )

        if index < args.preview_count:
            save_preview(
                image,
                masked,
                (
                    preview_root
                    / (
                        f"{index:04d}_"
                        f"{class_name}_"
                        f"{source_path.stem}.jpg"
                    )
                ),
            )

        stats.append({
            "class_name":
                class_name,
            "source":
                str(source_path),
            "masked":
                str(output_path),
            "grounding_score":
                detection_score,
            "sam_iou_score":
                sam_score,
            "foreground_area_ratio":
                area_ratio,
            "center_mask_box_iou":
                center_box_iou,
            "mask_box_coverage":
                mask_coverage,
            "mask_strategy":
                strategy,
            "fallback":
                fallback,
        })

    stats_path = (
        output_root.parent
        / (
            f"{output_root.name}"
            "_mask_stats.csv"
        )
    )

    with open(
        stats_path,
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                stats[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            stats
        )

    print(
        "Preview:",
        preview_root,
    )

    print(
        "Mask stats:",
        stats_path,
    )

    print(
        "No-detection fallbacks:",
        f"{no_detection_count}/"
        f"{len(images)}",
    )

    del grounding_model
    del grounding_processor
    del sam_model
    del sam_processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return list_images(
        output_root
    )


@torch.no_grad()
def extract_clip_features(
    image_paths,
    device,
    batch_size,
):
    import open_clip

    print(
        "Loading OpenCLIP "
        "ViT-H-14 / "
        "laion2b_s32b_b79k"
    )

    (
        model,
        _,
        preprocess,
    ) = (
        open_clip
        .create_model_and_transforms(
            "ViT-H-14",
            pretrained=(
                "laion2b_s32b_b79k"
            ),
            precision="fp32",
            device=device,
            cache_dir=os.environ.get(
                "OPEN_CLIP_CACHE_DIR"
            ),
        )
    )

    model.eval()

    batches = []

    for start in tqdm(
        range(
            0,
            len(image_paths),
            batch_size,
        ),
        desc="CLIP projected features",
    ):
        batch_paths = image_paths[
            start:
            start + batch_size
        ]

        images = torch.stack([
            preprocess(
                Image.open(path)
                .convert("RGB")
            )
            for path
            in batch_paths
        ]).to(device)

        features = (
            model
            .encode_image(
                images
            )
        )

        batches.append(
            features
            .detach()
            .float()
            .cpu()
        )

    return torch.cat(
        batches,
        dim=0,
    )


def save_feature_cache(
    args,
    img_features,
):
    source = torch.load(
        args.source_features_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        tuple(
            source[
                "img_features"
            ].shape
        )
        != tuple(
            img_features.shape
        )
    ):
        raise RuntimeError(
            "Feature shape mismatch: "
            f"source="
            f"{tuple(source['img_features'].shape)}, "
            f"sam_black="
            f"{tuple(img_features.shape)}"
        )

    output = dict(
        source
    )

    output[
        "img_features"
    ] = img_features

    output[
        "feature_metadata"
    ] = {
        "transform":
            "grounded_sam_black_background",
        "grounding_model":
            args.grounding_model,
        "grounding_prompt":
            "class_folder_name",
        "detection_threshold":
            args.detection_threshold,
        "text_threshold":
            args.text_threshold,
        "box_padding":
            args.box_padding,
        "sam_model":
            args.sam_model,
        "sam_prompt":
            "grounding_dino_box",
        "no_detection_fallback":
            "keep_original_image",
        "background_rgb":
            [0, 0, 0],
        "clip_model":
            "ViT-H-14",
        "clip_pretrained":
            "laion2b_s32b_b79k",
        "clip_features_normalized":
            False,
        "source_image_dir":
            str(
                args
                .input_dir
                .resolve()
            ),
        "masked_image_dir":
            str(
                args
                .output_image_dir
                .resolve()
            ),
    }

    (
        args
        .output_features_path
        .parent
        .mkdir(
            parents=True,
            exist_ok=True,
        )
    )

    torch.save(
        output,
        args.output_features_path,
    )

    print(
        "Saved feature cache:",
        args.output_features_path,
    )

    print(
        "img_features:",
        tuple(
            img_features.shape
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output_image_dir",
        type=Path,
        default=DEFAULT_MASKED,
    )

    parser.add_argument(
        "--source_features_path",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--output_features_path",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--grounding_model",
        default=(
            "IDEA-Research/"
            "grounding-dino-tiny"
        ),
    )

    parser.add_argument(
        "--sam_model",
        default=(
            "facebook/"
            "sam-vit-base"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--box_padding",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--center_mask_box_iou",
        type=float,
        default=0.45,
        help=(
            "If the old center-point SAM mask agrees with the "
            "Grounding DINO box at least this much, reuse it."
        ),
    )

    parser.add_argument(
        "--min_mask_box_coverage",
        type=float,
        default=0.20,
        help=(
            "If box-prompt SAM keeps less than this fraction of "
            "the detected box area, fall back to keeping the whole box."
        ),
    )

    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--preview_count",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    masked_paths = (
        create_masked_images(
            args,
            device,
        )
    )

    if args.max_images is not None:
        print(
            "Preview run only: "
            "CLIP feature extraction "
            "was skipped."
        )
        return

    expected = (
        N_CLASSES
        * CONDITIONS_PER_CLASS
    )

    if (
        len(masked_paths)
        != expected
    ):
        raise RuntimeError(
            f"Expected {expected} "
            "masked images, got "
            f"{len(masked_paths)}"
        )

    img_features = (
        extract_clip_features(
            masked_paths,
            device,
            args.clip_batch_size,
        )
    )

    save_feature_cache(
        args,
        img_features,
    )


if __name__ == "__main__":
    main()
