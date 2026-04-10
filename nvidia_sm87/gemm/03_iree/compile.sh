#!/bin/bash
cd "$(dirname "$0")"

# First export the MLIR from Python
python3.11 -c "
import torch
import iree.turbine.aot as aot

class GEMM(torch.nn.Module):
    def forward(self, A, B):
        return torch.mm(A, B)

model = GEMM()
a = torch.randn(4096, 4096)
b = torch.randn(4096, 4096)
exported = aot.export(model, a, b)
exported.save_mlir('gemm.mlir')
print('Saved gemm.mlir')
"

# Then compile with iree-compile and tuning flags
/home/bourram/iree-install/bin/iree-compile gemm.mlir \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_87 \
  --iree-cuda-target-features=+ptx74 \
  --iree-codegen-llvmgpu-use-vector-distribution \
  --iree-llvmgpu-enable-shared-memory-reuse \
  -o gemm.vmfb

echo "Done! Saved gemm.vmfb"
