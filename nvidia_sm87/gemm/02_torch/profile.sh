#!/bin/bash
cd "$(dirname "$0")"

nsys profile -o torch_gemm --force-overwrite true python3.8 gemm.py

nsys stats --report gputrace --force-export true torch_gemm.nsys-rep
