import torch
import iree.turbine.aot as aot

# Define compute-bound matrix sizes
M, K, N = 4096, 4096, 4096

class GEMM(torch.nn.Module):
    def forward(self, A, B):
        return torch.mm(A, B)

model = GEMM()
example_a = torch.randn(M, K)
example_b = torch.randn(K, N)

exported = aot.export(model, example_a, example_b)
exported.session.set_flags(
    "--iree-cuda-target=sm_87",
    "--iree-cuda-target-features=+ptx74",
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-llvmgpu-enable-shared-memory-reuse",
)
exported.compile(save_to="gemm.vmfb", target_backends=["cuda"])
print("Done! Saved gemm.vmfb")
