#!/bin/bash
cd "$(dirname "$0")"
/opt/rocm/bin/hipcc hipblasgemm.cpp -o hipblasgemm -lhipblas --offload-arch=gfx942
