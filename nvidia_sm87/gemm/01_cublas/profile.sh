#!/bin/bash
cd "$(dirname "$0")"

nsys profile -o cublas_gemm --force-overwrite true ./cublasgemm

nsys stats --report gputrace --force-export true cublas_gemm.nsys-rep
