from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import tensorrt as trt
from cuda import cudart

from common import benchmark_callable, now_str, resolve_path, save_json, summarize_latencies, topk_predictions
from preprocess import preprocess_images


def cuda_call(call_result):
    err = call_result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime call failed: {err}")
    if len(call_result) == 1:
        return None
    if len(call_result) == 2:
        return call_result[1]
    return call_result[1:]


@dataclass
class TensorBinding:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    nbytes: int
    device_ptr: int
    host_array: np.ndarray


class TensorRTRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = resolve_path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.stream = cuda_call(cudart.cudaStreamCreate())
        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly one input tensor, got {self.input_names}")
        if len(self.output_names) != 1:
            raise RuntimeError(f"Expected exactly one output tensor, got {self.output_names}")

        self.bindings: dict[str, TensorBinding] = {}
        self.prepared_shape: tuple[int, ...] | None = None

    def expected_input_shape(self) -> tuple[int, ...]:
        input_name = self.input_names[0]
        return tuple(int(dim) for dim in self.engine.get_tensor_shape(input_name))

    def prepare(self, input_batch: np.ndarray) -> None:
        input_name = self.input_names[0]
        input_shape = tuple(int(x) for x in input_batch.shape)
        if self.prepared_shape == input_shape:
            return

        self.release_bindings()

        if not self.context.set_input_shape(input_name, input_shape):
            raise RuntimeError(f"Failed to set TensorRT input shape: {input_shape}")

        for name in self.input_names + self.output_names:
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            nbytes = int(np.prod(shape) * dtype.itemsize)
            device_ptr = int(cuda_call(cudart.cudaMalloc(nbytes)))
            host_array = np.empty(shape, dtype=dtype)
            if not self.context.set_tensor_address(name, device_ptr):
                raise RuntimeError(f"Failed to bind tensor address for {name}")
            self.bindings[name] = TensorBinding(
                name=name,
                dtype=dtype,
                shape=shape,
                nbytes=nbytes,
                device_ptr=device_ptr,
                host_array=host_array,
            )

        self.prepared_shape = input_shape

    def infer(self, input_batch: np.ndarray) -> np.ndarray:
        self.prepare(input_batch)
        input_name = self.input_names[0]
        output_name = self.output_names[0]

        input_binding = self.bindings[input_name]
        output_binding = self.bindings[output_name]

        casted = input_batch.astype(input_binding.dtype, copy=False)
        np.copyto(input_binding.host_array, casted)

        cuda_call(
            cudart.cudaMemcpyAsync(
                input_binding.device_ptr,
                input_binding.host_array.ctypes.data,
                input_binding.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
        )

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed.")

        cuda_call(
            cudart.cudaMemcpyAsync(
                output_binding.host_array.ctypes.data,
                output_binding.device_ptr,
                output_binding.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        )
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return output_binding.host_array.copy()

    def synchronize(self) -> None:
        cuda_call(cudart.cudaStreamSynchronize(self.stream))

    def release_bindings(self) -> None:
        for binding in self.bindings.values():
            cuda_call(cudart.cudaFree(binding.device_ptr))
        self.bindings.clear()
        self.prepared_shape = None

    def close(self) -> None:
        self.release_bindings()
        if self.stream is not None:
            cuda_call(cudart.cudaStreamDestroy(self.stream))
            self.stream = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TensorRT engine inference via Python runtime API.")
    parser.add_argument("--engine", default="models/model_fp32.engine", help="TensorRT engine path.")
    parser.add_argument("--input", default="assets/sample_images", help="Image file or directory.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side before crop.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations.")
    parser.add_argument("--output-prefix", default="results/raw/trt_fp32", help="Prefix for outputs.")
    return parser.parse_args()


def infer_precision_name(engine_path: str) -> str:
    lower_name = engine_path.lower()
    if "fp16" in lower_name:
        return "fp16"
    if "int8" in lower_name:
        return "int8"
    return "fp32"


def split_or_validate_batch(batch_np: np.ndarray, expected_shape: tuple[int, ...]) -> list[np.ndarray]:
    if not expected_shape:
        return [batch_np]

    actual_shape = tuple(int(x) for x in batch_np.shape)
    if actual_shape == expected_shape:
        return [batch_np]

    if len(expected_shape) == len(actual_shape) and expected_shape[0] == 1 and actual_shape[1:] == expected_shape[1:]:
        return [batch_np[index : index + 1] for index in range(actual_shape[0])]

    raise ValueError(
        "TensorRT engine input shape does not match preprocessed batch. "
        f"Engine expects {expected_shape}, got {actual_shape}. "
        "Rebuild the engine with a matching optimization profile or provide compatible inputs."
    )


def main() -> None:
    args = parse_args()
    batch_np, image_paths = preprocess_images(
        input_path=args.input,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )

    runner = TensorRTRunner(args.engine)
    engine_input_shape = list(runner.expected_input_shape())
    try:
        input_batches = split_or_validate_batch(batch_np, runner.expected_input_shape())

        logits, latencies_ms = benchmark_callable(
            lambda: np.concatenate([runner.infer(current_batch) for current_batch in input_batches], axis=0),
            warmup=args.warmup,
            iterations=args.iterations,
            synchronize=runner.synchronize,
        )
    finally:
        runner.close()

    logits = np.asarray(logits, dtype=np.float32)
    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_prefix) + "_logits.npy", logits)

    benchmark = summarize_latencies(latencies_ms)
    throughput_qps = (len(image_paths) * args.iterations) / (sum(latencies_ms) / 1000.0)
    precision = infer_precision_name(args.engine)

    save_json(
        str(output_prefix) + "_predictions.json",
        {
            "timestamp": now_str(),
            "runner": "tensorrt_python",
            "engine_path": str(resolve_path(args.engine)),
            "precision": precision,
            "engine_input_shape": engine_input_shape,
            "image_paths": [str(path) for path in image_paths],
            "predictions": topk_predictions(logits, topk=5),
        },
    )
    save_json(
        str(output_prefix) + "_benchmark.json",
        {
            "timestamp": now_str(),
            "runner": "tensorrt_python",
            "engine_path": str(resolve_path(args.engine)),
            "precision": precision,
            "batch_size": int(batch_np.shape[0]),
            "engine_input_shape": engine_input_shape,
            "session_runs_per_iteration": len(input_batches),
            "image_size": args.image_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_ms": benchmark,
            "throughput_qps": float(throughput_qps),
            "output_logits": str(output_prefix) + "_logits.npy",
        },
    )

    print(f"Saved TensorRT outputs with prefix: {output_prefix}")
    print(f"Mean latency: {benchmark['mean']:.4f} ms")
    print(f"Throughput: {throughput_qps:.4f} images/s")


if __name__ == "__main__":
    main()
