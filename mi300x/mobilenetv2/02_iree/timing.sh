#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv
time "$VENV/bin/iree-run-module" \
  --module=mobilenetv2.vmfb \
  --device=hip \
  --function=main \
  --input=@../input/x.npy
