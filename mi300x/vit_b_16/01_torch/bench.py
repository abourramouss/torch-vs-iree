import torch
import torchvision
import numpy as np
import time

model = torchvision.models.vit_b_16(weights="DEFAULT").eval().cuda()
x = torch.from_numpy(np.load("../input/x.npy")).cuda()

# Warm-up
with torch.no_grad():
    model(x)
torch.cuda.synchronize()

# Benchmark 1000 inferences
start = time.perf_counter()
with torch.no_grad():
    for i in range(1000):
        result = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"1000 inferences: {elapsed:.3f}s")
print(f"Average: {elapsed/1000*1000:.2f}ms per inference")
