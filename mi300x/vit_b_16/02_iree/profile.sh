#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

rocprofv3 --kernel-trace --output-format csv -d iree_vit_b_16 -- \
  "$VENV/bin/iree-run-module" \
  --module=vit_b_16.vmfb \
  --device=hip \
  --function=main \
  --input=@../input/x.npy
