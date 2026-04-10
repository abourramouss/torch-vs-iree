#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

rocprofv3 --kernel-trace --output-format csv -d iree_mobilenetv2_O3 -- \
  "$VENV/bin/iree-run-module" \
  --module=mobilenetv2_O3.vmfb \
  --device=hip \
  --function=main \
  --input=@../input/x.npy
