import torch
import torchvision
import numpy as np

model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval().cuda()

x = torch.from_numpy(np.load("../input/x.npy")).cuda()

with torch.no_grad():
    result = model(x)
