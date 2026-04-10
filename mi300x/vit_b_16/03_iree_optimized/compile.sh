#!/bin/bash
set -e
cd "$(dirname "$0")"
VENV=/root/torch-vs-iree/.venv

# Reuse MLIR from 02_iree
MLIR="../02_iree/vit_b_16.mlir"
if [ ! -f "$MLIR" ]; then
  echo "Run 02_iree/compile.sh first to generate vit_b_16.mlir"
  exit 1
fi

rm -f vit_b_16_O3.vmfb
rm -rf dumps && mkdir -p dumps

"$VENV/bin/iree-compile" "$MLIR" \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx942 \
  --iree-opt-level=O3 \
  --iree-opt-const-expr-hoisting \
  --iree-dispatch-creation-enable-aggressive-fusion \
  --iree-codegen-llvmgpu-use-vector-distribution \
  --iree-llvmgpu-enable-shared-memory-reuse \
  --iree-hal-dump-executable-sources-to=dumps \
  -o vit_b_16_O3.vmfb

# iree-compile may exit 0 even after a stack dump, so verify the artifact.
if [ ! -s vit_b_16_O3.vmfb ]; then
  echo "ERROR: vit_b_16_O3.vmfb not produced"
  exit 1
fi
echo "Done! Saved vit_b_16_O3.vmfb"
