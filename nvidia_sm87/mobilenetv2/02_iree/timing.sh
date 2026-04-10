#!/bin/bash
cd "$(dirname "$0")"
time /home/bourram/iree-install/bin/iree-run-module \
  --module=mobilenetv2.vmfb \
  --device=cuda \
  --function=main \
  --input=@../input/x.npy
