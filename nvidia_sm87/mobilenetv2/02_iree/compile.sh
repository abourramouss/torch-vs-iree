#!/bin/bash
cd "$(dirname "$0")"

# Export MLIR from PyTorch
python3.11 -c "
import torch
import torchvision
import iree.turbine.aot as aot

model = torchvision.models.mobilenet_v2(weights='DEFAULT').eval()
example_input = torch.randn(1, 3, 224, 224)
exported = aot.export(model, example_input)
exported.save_mlir('mobilenetv2.mlir')
print('Saved mobilenetv2.mlir')
"

# Compile with iree-compile
/home/bourram/iree-install/bin/iree-compile mobilenetv2.mlir \
  --iree-hal-target-device=cuda \
  --iree-cuda-target=sm_87 \
  --iree-cuda-target-features=+ptx74 \
  -o mobilenetv2.vmfb

echo "Done! Saved mobilenetv2.vmfb"
