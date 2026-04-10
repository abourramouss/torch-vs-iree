#!/bin/bash
cd "$(dirname "$0")"

nsys profile -o iree_mobilenetv2 --force-overwrite true \
  /home/bourram/iree-install/bin/iree-run-module \
  --module=mobilenetv2.vmfb \
  --device=cuda \
  --function=main \
  --input=@../input/x.npy

nsys stats --report gputrace --force-export true iree_mobilenetv2.nsys-rep
