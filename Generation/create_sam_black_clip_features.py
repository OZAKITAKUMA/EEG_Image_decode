#!/usr/bin/env python3
"""SAMで前景だけを残して背景を黒化し、CLIP教師特徴を再抽出する。"""

import argparse
import csv
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
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ))
    return paths


def load_sam(model_name, device):
    from transformers import SamModel, SamProcessor

    print(f"Loading SAM: {model_name} on {device}")
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def predict_center_mask(image, model, processor, device):
    """画像中心をpositive pointとして3候補を出し、predicted IoU最大を採用する。"""
    width, height = image.size
    input_points = [[[width / 2.0, height / 2.0]]]

    inputs = processor(
        images=image,
        input_points=input_points,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
    )[0][0]

    scores = outputs.iou_scores.detach().cpu()[0][0]
    best = int(torch.argmax(scores).item())

    mask = masks[best].bool().numpy()
    score = float(scores[best].item())
    area_ratio = float(mask.mean())

    return mask, score, area_ratio


def apply_black_background(image, mask):
    array = np.asarray(image.convert("RGB"))
    if mask.shape != array.shape[:2]:
        raise RuntimeError(
            f"mask/image shape mismatch: mask={mask.shape}, image={array.shape[:2]}"
        )

    output = np.zeros_like(array)
    output[mask] = array[mask]
    return Image.fromarray(output)


def save_preview(original, masked, path):
    canvas = Image.new(
        "RGB",
        (original.width * 2, original.height),
        color=(0, 0, 0),
    )
    canvas.paste(original, (0, 0))
    canvas.paste(masked, (original.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def create_masked_images(args, device):
    input_root = args.input_dir.resolve()
    output_root = args.output_image_dir.resolve()

    images = list_images(input_root)
    print("Input images:", len(images))

    if args.max_images is not None:
        images = images[:args.max_images]

    model, processor = load_sam(args.sam_model, device)

    stats = []
    preview_root = Path(str(output_root) + "_preview")

    for index, source_path in enumerate(tqdm(images, desc="SAM black background")):
        relative = source_path.relative_to(input_root)
        output_path = output_root / relative

        image = Image.open(source_path).convert("RGB")

        if output_path.exists() and not args.overwrite:
            masked = Image.open(output_path).convert("RGB")
            score = float("nan")
            area_ratio = float("nan")
        else:
            mask, score, area_ratio = predict_center_mask(
                image,
                model,
                processor,
                device,
            )
            masked = apply_black_background(image, mask)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            if output_path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            masked.save(output_path, **save_kwargs)

        if index < args.preview_count:
            save_preview(
                image,
                masked,
                preview_root / f"{index:04d}_{source_path.stem}.jpg",
            )

        stats.append({
            "source": str(source_path),
            "masked": str(output_path),
            "sam_iou_score": score,
            "foreground_area_ratio": area_ratio,
        })

    stats_path = output_root.parent / f"{output_root.name}_mask_stats.csv"
    with open(stats_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    print("Preview:", preview_root)
    print("Mask stats:", stats_path)

    return list_images(output_root)


@torch.no_grad()
def extract_clip_features(image_paths, device, batch_size):
    import open_clip

    print("Loading OpenCLIP ViT-H-14 / laion2b_s32b_b79k")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        precision="fp32",
        device=device,
        cache_dir=os.environ.get("OPEN_CLIP_CACHE_DIR"),
    )
    model.eval()

    batches = []
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="CLIP projected features",
    ):
        batch_paths = image_paths[start:start + batch_size]
        images = torch.stack([
            preprocess(Image.open(path).convert("RGB"))
            for path in batch_paths
        ]).to(device)

        features = model.encode_image(images)
        batches.append(features.detach().float().cpu())

    return torch.cat(batches, dim=0)


def save_feature_cache(args, img_features):
    source = torch.load(
        args.source_features_path,
        map_location="cpu",
        weights_only=False,
    )

    if tuple(source["img_features"].shape) != tuple(img_features.shape):
        raise RuntimeError(
            "Feature shape mismatch: "
            f"source={tuple(source['img_features'].shape)}, "
            f"sam_black={tuple(img_features.shape)}"
        )

    output = dict(source)
    output["img_features"] = img_features
    output["feature_metadata"] = {
        "transform": "sam_black_background",
        "sam_model": args.sam_model,
        "sam_prompt": "image_center_positive_point",
        "background_rgb": [0, 0, 0],
        "clip_model": "ViT-H-14",
        "clip_pretrained": "laion2b_s32b_b79k",
        "clip_features_normalized": False,
        "source_image_dir": str(args.input_dir.resolve()),
        "masked_image_dir": str(args.output_image_dir.resolve()),
    }

    args.output_features_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_features_path)

    print("Saved feature cache:", args.output_features_path)
    print("img_features:", tuple(img_features.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_image_dir", type=Path, default=DEFAULT_MASKED)
    parser.add_argument("--source_features_path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_features_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam_model", default="facebook/sam-vit-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_batch_size", type=int, default=20)
    parser.add_argument("--preview_count", type=int, default=20)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    masked_paths = create_masked_images(args, device)

    if args.max_images is not None:
        print("Preview run only: CLIP feature extraction was skipped.")
        return

    expected = N_CLASSES * CONDITIONS_PER_CLASS
    if len(masked_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} masked images, got {len(masked_paths)}"
        )

    img_features = extract_clip_features(
        masked_paths,
        device,
        args.clip_batch_size,
    )
    save_feature_cache(args, img_features)


if __name__ == "__main__":
    main()
