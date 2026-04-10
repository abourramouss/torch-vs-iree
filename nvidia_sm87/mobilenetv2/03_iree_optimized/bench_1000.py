import numpy as np
import time
import iree.runtime as ireert

device = ireert.get_device("cuda")
config = ireert.Config(device=device)
ctx = ireert.SystemContext(config=config)
with open("mobilenetv2_O3.vmfb", "rb") as f:
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, f.read())
ctx.add_vm_module(vm_module)

x = np.load("../input/x.npy")
iree_fn = ctx.modules.module.main

# Warm-up
iree_fn(x)

# Benchmark 1000 inferences
start = time.perf_counter()
for i in range(1000):
    iree_fn(x)
elapsed = time.perf_counter() - start

print(f"1000 inferences: {elapsed:.3f}s")
print(f"Average: {elapsed/1000*1000:.2f}ms per inference")
