#!/bin/bash
set -e
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

# Reuse MLIR from 02_iree
MLIR="../02_iree/mobilenetv2.mlir"
if [ ! -f "$MLIR" ]; then
  echo "Run 02_iree/compile.sh first to generate mobilenetv2.mlir"
  exit 1
fi

rm -f mobilenetv2_O3.vmfb

# Note: --iree-opt-data-tiling crashes iree-compile 3.11 on ROCm/gfx942
# for this model, so it's omitted here.
"$VENV/bin/iree-compile" "$MLIR" \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx942 \
  --iree-opt-level=O3 \
  --iree-opt-const-expr-hoisting \
  --iree-dispatch-creation-enable-aggressive-fusion \
  --iree-codegen-llvmgpu-use-vector-distribution \
  --iree-llvmgpu-enable-shared-memory-reuse \
  -o mobilenetv2_O3.vmfb

# iree-compile may exit 0 even after a stack dump, so verify the artifact.
if [ ! -s mobilenetv2_O3.vmfb ]; then
  echo "ERROR: mobilenetv2_O3.vmfb not produced"
  exit 1
fi
echo "Done! Saved mobilenetv2_O3.vmfb"
