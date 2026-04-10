#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv
time "$VENV/bin/iree-run-module" \
  --module=vit_b_16.vmfb \
  --device=hip \
  --function=main \
  --input=@../input/x.npy
