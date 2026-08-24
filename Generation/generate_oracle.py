import argparse
from pathlib import Path

import torch
from pipeline import Generator4Embeds
from tqdm import tqdm

from evaluate import (
    load_gt_images_as_tensor,
    load_generated_images_grouped,
    compute_metrics,
)

def load_class_names(image_directory):
    class_dirs = sorted(
        path for path in image_directory.iterdir() 
        if path.is_dir()
    )

    return [path.name.split("_",1)[1]
           if "_" in path.name
           else path.name
           for path in class_dirs
      ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=Path, required=True)
    parser.add_argument("--img_directory_test", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--sdxl_model_path", default="stabilityai/sdxl-turbo")
    parser.add_argument("--ip_adapter_path", default="h94/IP-Adapter")
    parser.add_argument("--sdxl_steps", type=int, default=4)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--num_gen_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle_dir", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.features_path, map_location="cpu",)
    image_features = saved["img_features"].float()
    class_names = load_class_names(args.img_directory_test)

    if image_features.shape != (len(class_names), 1024):
        raise RuntimeError(
            f"特徴量とクラス数が一致しません: "
            f"features={image_features.shape}, "
            f"classes={len(class_names)}"
        )

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    generation_dir = (
        args.output_dir
        / "generated_imgs_oracle"
        / args.subject
    )
    for class_name in class_names:
        (generation_dir / class_name).mkdir(
            parents=True,
            exist_ok=True,
        )
    print("Loading IP-Adapter + SDXL-Turbo...")

    generator = Generator4Embeds(
        num_inference_steps=args.sdxl_steps,
        device=str(device),
        sdxl_model_path=args.sdxl_model_path,
        ip_adapter_path=args.ip_adapter_path,
    )

    all_features = image_features.repeat_interleave(args.num_gen_per_class, dim=0,)
    torch_generator = torch.Generator(device=device).manual_seed(args.seed)

    for start in tqdm(range(0, len(all_features), args.gen_batch_size), desc="Generating Oracle",):
        batch = all_features[start:start + args.gen_batch_size].to(device=device, dtype=torch.float16,)
        images = generator.generate_batch(batch,generator=torch_generator,)

        for offset, image in enumerate(images):
            index = start + offset
            class_index = index // args.num_gen_per_class
            sample_index = index % args.num_gen_per_class

            save_path = (
                generation_dir
                / class_names[class_index]
                / f"{sample_index}.png"
            )
            image.save(save_path)

        print(f"Oracle images saved: {generation_dir}")

    print(f"Oracle images saved: {generation_dir}")

    # SDXLをGPUから下ろして、評価モデル用の容量を空ける
    del generator
    torch.cuda.empty_cache()

    print("\nLoading images for Oracle evaluation...")

    gt_images = load_gt_images_as_tensor(
        str(args.img_directory_test),
        device,
    )

    recons_grouped, n_per_class = load_generated_images_grouped(
        str(generation_dir),
        device,
    )

    print("Generated:", recons_grouped.shape)
    print("Ground truth:", gt_images.shape)

    oracle_metrics, oracle_stds = compute_metrics(
        recons_grouped,
        gt_images,
        device,
        n_per_class=n_per_class,
    )

    print("\n" + "=" * 50)
    print("Oracle reconstruction metrics")
    print("=" * 50)

    for metric_name, score in oracle_metrics.items():
        std = oracle_stds[metric_name]
        print(f"{metric_name:<14} {score:.4f} ± {std:.4f}")

    print("=" * 50)
        
    

if __name__ ==  "__main__":
        main()