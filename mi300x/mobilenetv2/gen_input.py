import numpy as np
import os

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
os.makedirs(out_dir, exist_ok=True)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)
np.save(os.path.join(out_dir, "x.npy"), x)
x.tofile(os.path.join(out_dir, "x.bin"))

print(f"Saved input to {out_dir}")
