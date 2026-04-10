#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

rocprofv3 --kernel-trace --output-format csv -d iree_mobilenetv2 -- \
  "$VENV/bin/iree-run-module" \
  --module=mobilenetv2.vmfb \
  --device=hip \
  --function=main \
  --input=@../input/x.npy
