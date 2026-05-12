#!/usr/bin/env bash
set -euo pipefail

python scripts/build_and_benchmark.py build \
  --onnx-path models/model.onnx \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --build-fp16

