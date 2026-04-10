#!/bin/bash
cd "$(dirname "$0")"

nsys profile -o iree_mobilenetv2_O3 --force-overwrite true \
  /home/bourram/iree-install/bin/iree-run-module \
  --module=mobilenetv2_O3.vmfb \
  --device=cuda \
  --function=main \
  --input=@../input/x.npy

nsys stats --report cuda_gpu_kern_sum --force-export true iree_mobilenetv2_O3.nsys-rep
