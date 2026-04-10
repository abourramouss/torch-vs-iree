
import torch, torchvision, numpy as np, sys
x = torch.from_numpy(np.load(sys.argv[1])).cuda()
model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval().cuda()
with torch.no_grad():
    out = model(x)
np.save(sys.argv[2], out.cpu().numpy())
