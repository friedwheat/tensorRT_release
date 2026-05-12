from __future__ import annotations

import argparse

import onnx
import torch

from common import load_torchvision_model, now_str, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a torchvision image classification model to ONNX.")
    parser.add_argument("--model-name", default="resnet50", help="Torchvision model name.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used during export.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--output", default="models/model.onnx", help="Output ONNX path.")
    parser.add_argument("--input-name", default="input", help="Input tensor name.")
    parser.add_argument("--output-name", default="logits", help="Output tensor name.")
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export a fixed-batch ONNX model instead of the default dynamic-batch model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, _ = load_torchvision_model(args.model_name, pretrained=True)
    model.eval()

    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, dtype=torch.float32)
    dynamic_batch = not args.static_batch

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            args.input_name: {0: "batch"},
            args.output_name: {0: "batch"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            opset_version=args.opset,
            input_names=[args.input_name],
            output_names=[args.output_name],
            dynamic_axes=dynamic_axes,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    input_shape = [args.batch_size, 3, args.image_size, args.image_size]
    if dynamic_batch:
        input_shape[0] = "dynamic"

    save_json(
        "results/raw/onnx_export.json",
        {
            "timestamp": now_str(),
            "model_name": args.model_name,
            "onnx_path": str(output_path),
            "input_name": args.input_name,
            "output_name": args.output_name,
            "input_shape": input_shape,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "opset": args.opset,
            "dynamic_batch": dynamic_batch,
            "static_batch": args.static_batch,
            "file_size_bytes": output_path.stat().st_size,
        },
    )

    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
