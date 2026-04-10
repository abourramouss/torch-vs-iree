#!/bin/bash
cd "$(dirname "$0")"
time /home/bourram/iree-install/bin/iree-run-module \
  --module=gemm.vmfb \
  --device=cuda \
  --function=main \
  --input=@../inputs/a.npy \
  --input=@../inputs/b.npy
