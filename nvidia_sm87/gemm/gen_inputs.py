import numpy as np
import os

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
os.makedirs(out_dir, exist_ok=True)

a = np.random.randn(4096, 4096).astype(np.float32)
b = np.random.randn(4096, 4096).astype(np.float32)

# Save as .npy for IREE and PyTorch
np.save(os.path.join(out_dir, "a.npy"), a)
np.save(os.path.join(out_dir, "b.npy"), b)

# Save as raw binary for cuBLAS C++
a.tofile(os.path.join(out_dir, "a.bin"))
b.tofile(os.path.join(out_dir, "b.bin"))

print(f"Saved inputs to {out_dir}")
