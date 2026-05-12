from __future__ import annotations

import argparse

import numpy as np

from common import now_str, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare logits from two inference backends.")
    parser.add_argument("--baseline", required=True, help="Baseline logits .npy file.")
    parser.add_argument("--candidate", required=True, help="Candidate logits .npy file.")
    parser.add_argument("--name", required=True, help="Comparison name, used in output filename.")
    parser.add_argument("--output-dir", default="results/raw", help="Directory for validation output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = np.load(resolve_path(args.baseline))
    candidate = np.load(resolve_path(args.candidate))

    if baseline.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {baseline.shape} vs {candidate.shape}")

    diff = np.abs(baseline - candidate)
    baseline_top1 = np.argmax(baseline, axis=1)
    candidate_top1 = np.argmax(candidate, axis=1)

    baseline_top5 = np.argsort(-baseline, axis=1)[:, :5]
    candidate_top5 = np.argsort(-candidate, axis=1)[:, :5]

    top1_agreement = float(np.mean(baseline_top1 == candidate_top1))
    top5_overlap = float(
        np.mean(
            [
                len(set(base_row.tolist()) & set(cand_row.tolist())) / 5.0
                for base_row, cand_row in zip(baseline_top5, candidate_top5, strict=True)
            ]
        )
    )

    cosine_similarity = []
    for base_row, cand_row in zip(baseline, candidate, strict=True):
        denom = np.linalg.norm(base_row) * np.linalg.norm(cand_row)
        cosine_similarity.append(float(np.dot(base_row, cand_row) / denom if denom else 0.0))

    payload = {
        "timestamp": now_str(),
        "name": args.name,
        "baseline": str(resolve_path(args.baseline)),
        "candidate": str(resolve_path(args.candidate)),
        "shape": list(baseline.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agreement": top1_agreement,
        "top5_overlap": top5_overlap,
        "mean_cosine_similarity": float(np.mean(cosine_similarity)),
    }

    output_path = resolve_path(args.output_dir) / f"validation_{args.name}.json"
    save_json(output_path, payload)

    print(f"Saved validation report to: {output_path}")
    print(f"Top-1 agreement: {top1_agreement:.4f}")
    print(f"Max abs diff: {payload['max_abs_diff']:.6f}")


if __name__ == "__main__":
    main()

