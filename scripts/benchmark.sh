#!/usr/bin/env bash
set -euo pipefail

python scripts/build_and_benchmark.py benchmark \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --benchmark-fp16

