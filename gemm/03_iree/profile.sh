#!/bin/bash
cd "$(dirname "$0")"

nsys profile -o iree_gemm --force-overwrite true \
  /home/bourram/iree-install/bin/iree-run-module \
  --module=gemm.vmfb \
  --device=cuda \
  --function=main \
  --input=@../inputs/a.npy \
  --input=@../inputs/b.npy

nsys stats --report gputrace --force-export true iree_gemm.nsys-rep
