from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def list_image_paths(input_path: str | Path) -> list[Path]:
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]

    image_paths = [
        child
        for child in sorted(path.rglob("*"))
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {path}")
    return image_paths


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("Latency list is empty.")
    return {
        "min": float(min(latencies_ms)),
        "max": float(max(latencies_ms)),
        "mean": float(statistics.mean(latencies_ms)),
        "median": float(statistics.median(latencies_ms)),
        "p95": percentile(latencies_ms, 95),
        "p99": percentile(latencies_ms, 99),
    }


def benchmark_callable(
    fn: Callable[[], Any],
    warmup: int,
    iterations: int,
    synchronize: Callable[[], None] | None = None,
) -> tuple[Any, list[float]]:
    output = None
    for _ in range(warmup):
        output = fn()
        if synchronize is not None:
            synchronize()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = fn()
        if synchronize is not None:
            synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return output, latencies_ms


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=str(resolve_path(cwd) if cwd is not None else PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    combined = result.stdout + ("\n" if result.stdout and result.stderr else "") + result.stderr
    if log_path is not None:
        log_file = resolve_path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(combined, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n{combined}"
        )
    return combined


def parse_trtexec_log(text: str) -> dict[str, float | None]:
    throughput = _match_float(r"Throughput:\s*([0-9.]+)\s*qps", text)

    latency_match = re.search(
        r"Latency:\s*min = ([0-9.]+) ms,\s*max = ([0-9.]+) ms,\s*mean = ([0-9.]+) ms,\s*median = ([0-9.]+) ms,\s*percentile\(99%\) = ([0-9.]+) ms",
        text,
    )
    enqueue_match = re.search(
        r"Enqueue Time:\s*min = ([0-9.]+) ms,\s*max = ([0-9.]+) ms,\s*mean = ([0-9.]+) ms,\s*median = ([0-9.]+) ms,\s*percentile\(99%\) = ([0-9.]+) ms",
        text,
    )

    metrics: dict[str, float | None] = {
        "throughput_qps": throughput,
        "latency_min_ms": None,
        "latency_max_ms": None,
        "latency_mean_ms": None,
        "latency_median_ms": None,
        "latency_p99_ms": None,
        "enqueue_mean_ms": None,
    }

    if latency_match:
        metrics["latency_min_ms"] = float(latency_match.group(1))
        metrics["latency_max_ms"] = float(latency_match.group(2))
        metrics["latency_mean_ms"] = float(latency_match.group(3))
        metrics["latency_median_ms"] = float(latency_match.group(4))
        metrics["latency_p99_ms"] = float(latency_match.group(5))

    if enqueue_match:
        metrics["enqueue_mean_ms"] = float(enqueue_match.group(3))

    return metrics


def _match_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def topk_predictions(
    logits: np.ndarray,
    topk: int = 5,
    categories: list[str] | None = None,
) -> list[list[dict[str, Any]]]:
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)

    probs = softmax(logits.astype(np.float64))
    topk_indices = np.argsort(-probs, axis=1)[:, :topk]

    predictions: list[list[dict[str, Any]]] = []
    for sample_idx, indices in enumerate(topk_indices):
        sample_results: list[dict[str, Any]] = []
        for rank, class_idx in enumerate(indices, start=1):
            class_index = int(class_idx)
            item: dict[str, Any] = {
                "rank": rank,
                "class_id": class_index,
                "score": float(probs[sample_idx, class_index]),
            }
            if categories and class_index < len(categories):
                item["label"] = categories[class_index]
            sample_results.append(item)
        predictions.append(sample_results)
    return predictions


def load_torchvision_model(model_name: str, pretrained: bool = True):
    from torchvision.models import get_model, get_model_weights

    weights = None
    categories = None
    if pretrained:
        weights = get_model_weights(model_name).DEFAULT
        categories = list(weights.meta.get("categories", []))

    model = get_model(model_name, weights=weights)
    return model, categories


def markdown_table(rows: Iterable[dict[str, Any]], headers: list[str]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join([header_line, separator, *body])
