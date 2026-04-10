#!/bin/bash
set -e
cd "$(dirname "$0")"

VENV=/root/torch-vs-iree/.venv

# Export MLIR from PyTorch
"$VENV/bin/python" -c "
import torch
import torchvision
import iree.turbine.aot as aot

model = torchvision.models.vit_b_16(weights='DEFAULT').eval()
example_input = torch.randn(1, 3, 224, 224)
exported = aot.export(model, example_input)
exported.save_mlir('vit_b_16.mlir')
print('Saved vit_b_16.mlir')
"

# Compile at O0 — no optimizations, baseline for comparison.
# Dump per-dispatch MLIR sources to dumps/ for diffing against O3.
rm -rf dumps && mkdir -p dumps
"$VENV/bin/iree-compile" vit_b_16.mlir \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx942 \
  --iree-opt-level=O0 \
  --iree-hal-dump-executable-sources-to=dumps \
  -o vit_b_16.vmfb

if [ ! -s vit_b_16.vmfb ]; then
  echo "ERROR: vit_b_16.vmfb not produced"
  exit 1
fi
echo "Done! Saved vit_b_16.vmfb"
