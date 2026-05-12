from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort

from common import benchmark_callable, now_str, resolve_path, save_json, summarize_latencies, topk_predictions
from preprocess import preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime inference and benchmark it.")
    parser.add_argument("--input", default="assets/sample_images", help="Image file or directory.")
    parser.add_argument("--onnx-path", default="models/model.onnx", help="ONNX model path.")
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"], help="Execution provider.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side before crop.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations.")
    parser.add_argument("--output-prefix", default="results/raw/onnx", help="Prefix for outputs.")
    return parser.parse_args()


def pick_providers(provider: str) -> list[str]:
    available = ort.get_available_providers()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this ONNX Runtime install.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _is_dynamic_dim(dim: int | str | None) -> bool:
    return not isinstance(dim, int)


def _split_or_validate_batch(batch_np: np.ndarray, input_shape: list[int | str | None]) -> list[np.ndarray]:
    if not input_shape:
        return [batch_np]

    expected_batch = input_shape[0]
    if _is_dynamic_dim(expected_batch):
        return [batch_np]

    expected_batch = int(expected_batch)
    actual_batch = int(batch_np.shape[0])
    if actual_batch == expected_batch:
        return [batch_np]

    if expected_batch == 1:
        return [batch_np[index : index + 1] for index in range(actual_batch)]

    raise ValueError(
        "ONNX model batch dimension does not match input batch. "
        f"Model expects batch={expected_batch}, got batch={actual_batch}. "
        "Re-export the model with dynamic batch or provide a matching number of images."
    )


def main() -> None:
    args = parse_args()
    batch_np, image_paths = preprocess_images(
        input_path=args.input,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )

    providers = pick_providers(args.provider)
    session = ort.InferenceSession(str(resolve_path(args.onnx_path)), providers=providers)

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name
    output_name = output_meta.name
    input_shape = list(input_meta.shape)
    input_batches = _split_or_validate_batch(batch_np, input_shape)

    def infer_once():
        outputs = [session.run([output_name], {input_name: current_batch})[0] for current_batch in input_batches]
        return np.concatenate(outputs, axis=0)

    logits, latencies_ms = benchmark_callable(
        infer_once,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    logits = np.asarray(logits, dtype=np.float32)

    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_prefix) + "_logits.npy", logits)

    benchmark = summarize_latencies(latencies_ms)
    throughput_qps = (len(image_paths) * args.iterations) / (sum(latencies_ms) / 1000.0)

    save_json(
        str(output_prefix) + "_predictions.json",
        {
            "timestamp": now_str(),
            "runner": "onnxruntime",
            "provider": session.get_providers(),
            "input_name": input_name,
            "input_shape": input_shape,
            "output_name": output_name,
            "image_paths": [str(path) for path in image_paths],
            "predictions": topk_predictions(logits, topk=5),
        },
    )
    save_json(
        str(output_prefix) + "_benchmark.json",
        {
            "timestamp": now_str(),
            "runner": "onnxruntime",
            "provider": session.get_providers(),
            "onnx_path": str(resolve_path(args.onnx_path)),
            "batch_size": int(batch_np.shape[0]),
            "model_input_shape": input_shape,
            "session_runs_per_iteration": len(input_batches),
            "image_size": args.image_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_ms": benchmark,
            "throughput_qps": float(throughput_qps),
            "output_logits": str(output_prefix) + "_logits.npy",
        },
    )

    print(f"Saved ONNX Runtime outputs with prefix: {output_prefix}")
    print(f"Mean latency: {benchmark['mean']:.4f} ms")
    print(f"Throughput: {throughput_qps:.4f} images/s")


if __name__ == "__main__":
    main()
