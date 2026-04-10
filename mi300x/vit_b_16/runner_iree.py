"""Unified IREE runner: load vmfb, warmup, time N iterations wall-clock.

Used both standalone (wall-clock only) and under rocprofv3 (GPU time).
"""
import numpy as np
import time
import sys
import iree.runtime as ireert

vmfb_path = sys.argv[1]
input_path = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100

device = ireert.get_device("hip")
config = ireert.Config(device=device)
ctx = ireert.SystemContext(config=config)
with open(vmfb_path, "rb") as f:
    vm_module = ireert.VmModule.from_flatbuffer(
        ctx.instance, f.read(), warn_if_copy=False
    )
ctx.add_vm_module(vm_module)

x = np.load(input_path)
fn = ctx.modules.module.main

# Warmup
for _ in range(5):
    fn(x)

start = time.perf_counter()
for _ in range(N):
    fn(x)
elapsed = time.perf_counter() - start

print(f"N={N}")
print(f"wall_total_ms={elapsed*1000:.3f}")
print(f"wall_per_iter_ms={elapsed/N*1000:.3f}")
