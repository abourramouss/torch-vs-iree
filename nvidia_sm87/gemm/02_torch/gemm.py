import torch
import numpy as np

# Load pre-generated inputs
A = torch.from_numpy(np.load("../inputs/a.npy")).cuda()
B = torch.from_numpy(np.load("../inputs/b.npy")).cuda()

C = torch.mm(A, B)

print(f"Result shape: {C.shape}")
