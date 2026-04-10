"""Unified torch runner: load ViT-B/16, warmup, time N iterations wall-clock.

Used both standalone (wall-clock only) and under rocprofv3 (GPU time).
"""
import torch
import torchvision
import numpy as np
import time
import sys

input_path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

model = torchvision.models.vit_b_16(weights="DEFAULT").eval().cuda()
x = torch.from_numpy(np.load(input_path)).cuda()

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(x)
torch.cuda.synchronize()

start = time.perf_counter()
with torch.no_grad():
    for _ in range(N):
        result = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"N={N}")
print(f"wall_total_ms={elapsed*1000:.3f}")
print(f"wall_per_iter_ms={elapsed/N*1000:.3f}")
