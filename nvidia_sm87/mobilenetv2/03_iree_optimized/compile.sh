#!/bin/bash
cd "$(dirname "$0")"

# Reuse MLIR from 02_iree
MLIR="../02_iree/mobilenetv2.mlir"
if [ ! -f "$MLIR" ]; then
  echo "Run 02_iree/compile.sh first to generate mobilenetv2.mlir"
  exit 1
fi

/home/bourram/iree-install/bin/iree-compile "$MLIR" \
  --iree-hal-target-device=cuda \
  --iree-cuda-target=sm_87 \
  --iree-cuda-target-features=+ptx74 \
  --iree-opt-level=O3 \
  --iree-opt-const-expr-hoisting \
  --iree-opt-data-tiling \
  --iree-dispatch-creation-enable-aggressive-fusion \
  --iree-codegen-llvmgpu-use-vector-distribution \
  --iree-llvmgpu-enable-shared-memory-reuse \
  -o mobilenetv2_O3.vmfb

echo "Done! Saved mobilenetv2_O3.vmfb"
