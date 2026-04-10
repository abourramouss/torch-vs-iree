#!/bin/bash
cd "$(dirname "$0")"

nsys profile -o torch_mobilenetv2 --force-overwrite true python3.8 bench.py

nsys stats --report gputrace --force-export true torch_mobilenetv2.nsys-rep
