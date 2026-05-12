from __future__ import annotations

import argparse
import shutil

import onnx

from common import now_str, parse_trtexec_log, resolve_path, run_command, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engines and run trtexec benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build TensorRT engines.")
    add_shared_model_args(build_parser)
    build_parser.add_argument("--build-fp16", action="store_true", help="Build FP16 engine in addition to FP32.")
    build_parser.add_argument("--build-int8", action="store_true", help="Attempt to build INT8 engine.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark TensorRT engines with trtexec.")
    add_shared_model_args(benchmark_parser)
    benchmark_parser.add_argument("--benchmark-fp16", action="store_true", help="Benchmark FP16 engine as well.")
    benchmark_parser.add_argument("--benchmark-int8", action="store_true", help="Benchmark INT8 engine as well.")
    benchmark_parser.add_argument("--warmup-ms", type=int, default=500, help="Warmup duration in milliseconds.")
    benchmark_parser.add_argument("--iterations", type=int, default=100, help="Minimum number of iterations.")
    benchmark_parser.add_argument("--duration", type=int, default=10, help="Minimum duration in seconds.")
    benchmark_parser.add_argument("--no-data-transfers", action="store_true", help="Pass --noDataTransfers to trtexec.")

    full_parser = subparsers.add_parser("full", help="Build and benchmark FP32/FP16 engines.")
    add_shared_model_args(full_parser)
    full_parser.add_argument("--warmup-ms", type=int, default=500, help="Warmup duration in milliseconds.")
    full_parser.add_argument("--iterations", type=int, default=100, help="Minimum number of iterations.")
    full_parser.add_argument("--duration", type=int, default=10, help="Minimum duration in seconds.")
    full_parser.add_argument("--no-data-transfers", action="store_true", help="Pass --noDataTransfers to trtexec.")
    full_parser.add_argument("--build-fp16", action="store_true", default=True, help="Build FP16 engine as well.")

    return parser.parse_args()


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--onnx-path", default="models/model.onnx", help="ONNX model path.")
    parser.add_argument("--input-name", default="input", help="Input tensor name in ONNX / TensorRT.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for engine build and benchmark.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--trtexec", default="trtexec", help="Path to trtexec binary.")


def trtexec_path(value: str) -> str:
    executable = shutil.which(value) if value == "trtexec" else value
    if not executable:
        raise FileNotFoundError("Could not find trtexec in PATH. Install TensorRT first.")
    return executable


def shape_string(input_name: str, batch_size: int, image_size: int) -> str:
    return f"{input_name}:{batch_size}x3x{image_size}x{image_size}"


def _onnx_input_dims(onnx_path: str, input_name: str) -> list[int | str | None]:
    model = onnx.load(str(resolve_path(onnx_path)))
    graph_inputs = {value.name: value for value in model.graph.input}
    if input_name not in graph_inputs:
        raise ValueError(f"Input tensor '{input_name}' not found in ONNX model: {resolve_path(onnx_path)}")

    dims: list[int | str | None] = []
    tensor_type = graph_inputs[input_name].type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _trtexec_shape_args(
    onnx_path: str,
    input_name: str,
    batch_size: int,
    image_size: int,
) -> list[str]:
    dims = _onnx_input_dims(onnx_path, input_name)
    requested_dims = [batch_size, 3, image_size, image_size]
    has_dynamic_dim = any(not isinstance(dim, int) for dim in dims)

    if has_dynamic_dim:
        shape = shape_string(input_name, batch_size, image_size)
        return [
            f"--minShapes={shape}",
            f"--optShapes={shape}",
            f"--maxShapes={shape}",
        ]

    if dims != requested_dims:
        raise ValueError(
            "Static ONNX input shape does not match requested build shape. "
            f"Model shape: {dims}, requested: {requested_dims}. "
            "Re-export ONNX with matching dimensions or use --dynamic-batch during export."
        )

    return []


def build_engine(
    trtexec_bin: str,
    onnx_path: str,
    engine_path: str,
    input_name: str,
    batch_size: int,
    image_size: int,
    precision: str,
) -> None:
    command = [
        trtexec_bin,
        f"--onnx={resolve_path(onnx_path)}",
        f"--saveEngine={resolve_path(engine_path)}",
        "--skipInference",
        "--verbose",
    ]
    command.extend(_trtexec_shape_args(onnx_path, input_name, batch_size, image_size))
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")

    log_path = f"results/raw/build_{precision}.log"
    run_command(command, log_path=log_path)
    save_json(
        f"results/raw/build_{precision}.json",
        {
            "timestamp": now_str(),
            "runner": "trtexec_build",
            "precision": precision,
            "onnx_path": str(resolve_path(onnx_path)),
            "engine_path": str(resolve_path(engine_path)),
            "input_name": input_name,
            "batch_size": batch_size,
            "image_size": image_size,
            "log_path": str(resolve_path(log_path)),
        },
    )


def benchmark_engine(
    trtexec_bin: str,
    onnx_path: str,
    engine_path: str,
    input_name: str,
    batch_size: int,
    image_size: int,
    precision: str,
    warmup_ms: int,
    iterations: int,
    duration: int,
    no_data_transfers: bool,
) -> None:
    command = [
        trtexec_bin,
        f"--loadEngine={resolve_path(engine_path)}",
        f"--warmUp={warmup_ms}",
        f"--iterations={iterations}",
        f"--duration={duration}",
    ]
    command.extend(_trtexec_shape_args(onnx_path, input_name, batch_size, image_size))
    if no_data_transfers:
        command.append("--noDataTransfers")

    log_path = f"results/raw/benchmark_{precision}.log"
    output = run_command(command, log_path=log_path)
    metrics = parse_trtexec_log(output)

    save_json(
        f"results/raw/trtexec_benchmark_{precision}.json",
        {
            "timestamp": now_str(),
            "runner": "trtexec",
            "precision": precision,
            "engine_path": str(resolve_path(engine_path)),
            "input_name": input_name,
            "batch_size": batch_size,
            "image_size": image_size,
            "warmup": warmup_ms,
            "iterations": iterations,
            "duration_s": duration,
            "no_data_transfers": no_data_transfers,
            "log_path": str(resolve_path(log_path)),
            **metrics,
        },
    )


def main() -> None:
    args = parse_args()
    trtexec_bin = trtexec_path(args.trtexec)

    if args.command in {"build", "full"}:
        build_engine(
            trtexec_bin=trtexec_bin,
            onnx_path=args.onnx_path,
            engine_path="models/model_fp32.engine",
            input_name=args.input_name,
            batch_size=args.batch_size,
            image_size=args.image_size,
            precision="fp32",
        )
        if getattr(args, "build_fp16", False):
            build_engine(
                trtexec_bin=trtexec_bin,
                onnx_path=args.onnx_path,
                engine_path="models/model_fp16.engine",
                input_name=args.input_name,
                batch_size=args.batch_size,
                image_size=args.image_size,
                precision="fp16",
            )
        if getattr(args, "build_int8", False):
            build_engine(
                trtexec_bin=trtexec_bin,
                onnx_path=args.onnx_path,
                engine_path="models/model_int8.engine",
                input_name=args.input_name,
                batch_size=args.batch_size,
                image_size=args.image_size,
                precision="int8",
            )

    if args.command in {"benchmark", "full"}:
        benchmark_engine(
            trtexec_bin=trtexec_bin,
            onnx_path=args.onnx_path,
            engine_path="models/model_fp32.engine",
            input_name=args.input_name,
            batch_size=args.batch_size,
            image_size=args.image_size,
            precision="fp32",
            warmup_ms=args.warmup_ms,
            iterations=args.iterations,
            duration=args.duration,
            no_data_transfers=args.no_data_transfers,
        )
        if getattr(args, "benchmark_fp16", False) or args.command == "full":
            benchmark_engine(
                trtexec_bin=trtexec_bin,
                onnx_path=args.onnx_path,
                engine_path="models/model_fp16.engine",
                input_name=args.input_name,
                batch_size=args.batch_size,
                image_size=args.image_size,
                precision="fp16",
                warmup_ms=args.warmup_ms,
                iterations=args.iterations,
                duration=args.duration,
                no_data_transfers=args.no_data_transfers,
            )
        if getattr(args, "benchmark_int8", False):
            benchmark_engine(
                trtexec_bin=trtexec_bin,
                onnx_path=args.onnx_path,
                engine_path="models/model_int8.engine",
                input_name=args.input_name,
                batch_size=args.batch_size,
                image_size=args.image_size,
                precision="int8",
                warmup_ms=args.warmup_ms,
                iterations=args.iterations,
                duration=args.duration,
                no_data_transfers=args.no_data_transfers,
            )


if __name__ == "__main__":
    main()
