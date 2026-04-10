#!/bin/bash
set -e
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

"$VENV/bin/iree-benchmark-module" \
  --module=vit_b_16_O3.vmfb \
  --device=hip \
  --function=main \
  --input=@../input/x.npy \
  --benchmark_repetitions=5 \
  --benchmark_min_warmup_time=1.0
