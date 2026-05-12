from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from common import list_image_paths, now_str, resolve_path, save_json


def build_transform(image_size: int, resize_size: int, mean: list[float], std: list[float]):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_images(
    input_path: str | Path,
    image_size: int = 224,
    resize_size: int = 256,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    image_paths = list_image_paths(input_path)
    transform = build_transform(image_size=image_size, resize_size=resize_size, mean=mean, std=std)

    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            tensors.append(transform(rgb_image).numpy())

    batch = np.stack(tensors, axis=0).astype(np.float32)
    return batch, image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess sample images into a batched NCHW numpy file.")
    parser.add_argument("--input", default="assets/sample_images", help="Image file or directory.")
    parser.add_argument("--output", default="results/raw/input_batch.npy", help="Output .npy file path.")
    parser.add_argument("--meta-output", default="results/raw/input_batch_meta.json", help="Output metadata JSON path.")
    parser.add_argument("--image-size", type=int, default=224, help="Center crop size.")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side before crop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch, image_paths = preprocess_images(
        input_path=args.input,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, batch)

    save_json(
        args.meta_output,
        {
            "timestamp": now_str(),
            "input": str(resolve_path(args.input)),
            "output": str(output_path),
            "batch_shape": list(batch.shape),
            "image_paths": [str(path) for path in image_paths],
            "image_size": args.image_size,
            "resize_size": args.resize_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )

    print(f"Saved preprocessed batch to: {output_path}")
    print(f"Batch shape: {batch.shape}")


if __name__ == "__main__":
    main()

