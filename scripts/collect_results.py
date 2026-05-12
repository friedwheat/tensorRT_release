from __future__ import annotations

import csv
from pathlib import Path

from common import PROJECT_ROOT, load_json, markdown_table, now_str, resolve_path


def collect_benchmarks(raw_dir: Path) -> list[dict]:
    rows = []
    for file_path in sorted(raw_dir.glob("*benchmark*.json")):
        payload = load_json(file_path)
        if "runner" not in payload:
            continue
        row = {
            "timestamp": payload.get("timestamp", ""),
            "runner": payload.get("runner", ""),
            "model_name": payload.get("model_name", ""),
            "provider_or_engine": payload.get("provider", payload.get("engine_path", "")),
            "precision": payload.get("precision", ""),
            "batch_size": payload.get("batch_size", ""),
            "image_size": payload.get("image_size", ""),
            "warmup": payload.get("warmup", ""),
            "iterations": payload.get("iterations", ""),
            "latency_mean_ms": payload.get("latency_ms", {}).get("mean", payload.get("latency_mean_ms", "")),
            "latency_p95_ms": payload.get("latency_ms", {}).get("p95", payload.get("latency_p95_ms", "")),
            "latency_p99_ms": payload.get("latency_ms", {}).get("p99", payload.get("latency_p99_ms", "")),
            "throughput_qps": payload.get("throughput_qps", ""),
            "source_file": str(file_path.relative_to(PROJECT_ROOT)),
        }
        rows.append(row)
    return rows


def collect_validations(raw_dir: Path) -> list[dict]:
    rows = []
    for file_path in sorted(raw_dir.glob("validation_*.json")):
        payload = load_json(file_path)
        rows.append(
            {
                "timestamp": payload.get("timestamp", ""),
                "name": payload.get("name", ""),
                "max_abs_diff": payload.get("max_abs_diff", ""),
                "mean_abs_diff": payload.get("mean_abs_diff", ""),
                "top1_agreement": payload.get("top1_agreement", ""),
                "top5_overlap": payload.get("top5_overlap", ""),
                "mean_cosine_similarity": payload.get("mean_cosine_similarity", ""),
                "source_file": str(file_path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw_dir = resolve_path("results/raw")
    benchmark_rows = collect_benchmarks(raw_dir)
    validation_rows = collect_validations(raw_dir)

    benchmark_csv = resolve_path("results/benchmark_results.csv")
    validation_csv = resolve_path("results/accuracy_compare.csv")

    write_csv(benchmark_csv, benchmark_rows)
    write_csv(validation_csv, validation_rows)

    summary_lines = [
        f"# Experiment Summary",
        "",
        f"Generated at: {now_str()}",
        "",
        "## Benchmark Results",
        "",
    ]
    if benchmark_rows:
        summary_lines.append(
            markdown_table(
                benchmark_rows,
                headers=[
                    "runner",
                    "precision",
                    "batch_size",
                    "latency_mean_ms",
                    "throughput_qps",
                ],
            )
        )
    else:
        summary_lines.append("No benchmark files found.")

    summary_lines.extend(["", "## Accuracy Comparison", ""])
    if validation_rows:
        summary_lines.append(
            markdown_table(
                validation_rows,
                headers=[
                    "name",
                    "max_abs_diff",
                    "mean_abs_diff",
                    "top1_agreement",
                    "top5_overlap",
                ],
            )
        )
    else:
        summary_lines.append("No validation files found.")

    summary_path = resolve_path("results/raw/summary.md")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote benchmark CSV: {benchmark_csv}")
    print(f"Wrote accuracy CSV: {validation_csv}")
    print(f"Wrote summary markdown: {summary_path}")


if __name__ == "__main__":
    main()
