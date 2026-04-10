#!/bin/bash
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

rocprofv3 --kernel-trace --output-format csv -d torch_vit_b_16 -- \
  "$VENV/bin/python" bench.py
