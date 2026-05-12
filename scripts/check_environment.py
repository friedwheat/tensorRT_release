from __future__ import annotations

import json
import platform
import shutil
import subprocess

from common import now_str, resolve_path, save_json


def run_shell(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    except Exception:
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def import_version(module_name: str, attr_name: str = "__version__") -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, attr_name, None))


def main() -> None:
    torch_info = {}
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": int(torch.cuda.device_count()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}

    payload = {
        "timestamp": now_str(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch": torch_info,
        "onnx_version": import_version("onnx"),
        "onnxruntime_version": import_version("onnxruntime"),
        "tensorrt_version": import_version("tensorrt"),
        "trtexec_path": shutil.which("trtexec"),
        "nvidia_smi": run_shell(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc_version": run_shell(["nvcc", "--version"]),
    }

    output_path = resolve_path("results/raw/environment.json")
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved environment snapshot to: {output_path}")


if __name__ == "__main__":
    main()

