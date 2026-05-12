from __future__ import annotations

import argparse

import numpy as np
import torch

from common import benchmark_callable, load_torchvision_model, now_str, resolve_path, save_json, summarize_latencies, topk_predictions
from preprocess import preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyTorch image classification inference and benchmark it.")
    parser.add_argument("--input", default="assets/sample_images", help="Image file or directory.")
    parser.add_argument("--model-name", default="resnet50", help="Torchvision model name.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side before crop.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations.")
    parser.add_argument("--amp", action="store_true", help="Enable torch autocast(fp16) on CUDA.")
    parser.add_argument("--include-h2d", action="store_true", help="Include CPU->GPU copy in each timed iteration.")
    parser.add_argument("--output-prefix", default="results/raw/pytorch", help="Prefix for outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, categories = load_torchvision_model(args.model_name, pretrained=True)
    model.eval().to(device)

    batch_np, image_paths = preprocess_images(
        input_path=args.input,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )

    host_batch = torch.from_numpy(batch_np)
    static_batch = host_batch.to(device) if device == "cpu" or not args.include_h2d else None

    def infer_once():
        input_batch = host_batch.to(device) if args.include_h2d and device == "cuda" else static_batch
        with torch.inference_mode():
            if args.amp and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return model(input_batch)
            return model(input_batch)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    logits_tensor, latencies_ms = benchmark_callable(
        infer_once,
        warmup=args.warmup,
        iterations=args.iterations,
        synchronize=sync,
    )
    logits = logits_tensor.detach().cpu().numpy().astype(np.float32)

    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_prefix) + "_logits.npy", logits)

    predictions = topk_predictions(logits, topk=5, categories=categories)
    benchmark = summarize_latencies(latencies_ms)
    throughput_qps = (len(image_paths) * args.iterations) / (sum(latencies_ms) / 1000.0)

    save_json(
        str(output_prefix) + "_predictions.json",
        {
            "timestamp": now_str(),
            "runner": "pytorch",
            "model_name": args.model_name,
            "device": device,
            "precision": "fp16" if args.amp else "fp32",
            "image_paths": [str(path) for path in image_paths],
            "predictions": predictions,
        },
    )
    save_json(
        str(output_prefix) + "_benchmark.json",
        {
            "timestamp": now_str(),
            "runner": "pytorch",
            "model_name": args.model_name,
            "device": device,
            "precision": "fp16" if args.amp else "fp32",
            "batch_size": int(batch_np.shape[0]),
            "image_size": args.image_size,
            "include_h2d": bool(args.include_h2d),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_ms": benchmark,
            "throughput_qps": float(throughput_qps),
            "output_logits": str(output_prefix) + "_logits.npy",
        },
    )

    print(f"Saved PyTorch outputs with prefix: {output_prefix}")
    print(f"Mean latency: {benchmark['mean']:.4f} ms")
    print(f"Throughput: {throughput_qps:.4f} images/s")


if __name__ == "__main__":
    main()

