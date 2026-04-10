#!/bin/bash
set -e
cd "$(dirname "$0")"

VENV=/root/torch-vs-iree/.venv

# Export MLIR from PyTorch
"$VENV/bin/python" -c "
import torch
import torchvision
import iree.turbine.aot as aot

model = torchvision.models.mobilenet_v2(weights='DEFAULT').eval()
example_input = torch.randn(1, 3, 224, 224)
exported = aot.export(model, example_input)
exported.save_mlir('mobilenetv2.mlir')
print('Saved mobilenetv2.mlir')
"

# Compile with iree-compile for ROCm / MI300X (gfx942)
"$VENV/bin/iree-compile" mobilenetv2.mlir \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx942 \
  -o mobilenetv2.vmfb

echo "Done! Saved mobilenetv2.vmfb"
