#!/bin/bash
cd "$(dirname "$0")"

iree-compile \
--iree-hal-target-device=local \
--iree-hal-local-target-device-backends=llvm-cpu \
--iree-llvmcpu-target-cpu=host \
--iree-opt-level=O3 \
--iree-opt-data-tiling \
mobilenetv2.mlir -o mobilenet_cpu_03.vmfb

