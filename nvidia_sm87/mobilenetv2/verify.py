"""Run MobileNetV2 on a real image via both PyTorch and IREE, compare outputs."""
import numpy as np
import urllib.request
import json
import os
import subprocess
import sys

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
os.makedirs(data_dir, exist_ok=True)

img_path = os.path.join(data_dir, "dog.jpg")
labels_path = os.path.join(data_dir, "imagenet_labels.json")

if not os.path.exists(img_path):
    print("Downloading test image...")
    req = urllib.request.Request(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req) as resp, open(img_path, "wb") as f:
        f.write(resp.read())

if not os.path.exists(labels_path):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
        labels_path,
    )

# Preprocess image
from PIL import Image
img = Image.open(img_path).convert("RGB").resize((224, 224))
x = np.array(img, dtype=np.float32) / 255.0
x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = np.transpose(x, (2, 0, 1))[np.newaxis].astype(np.float32)  # NCHW
np.save(os.path.join(data_dir, "real_input.npy"), x)

with open(labels_path) as f:
    labels = json.load(f)

def print_top5(logits, title):
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    top5_idx = np.argsort(probs[0])[-5:][::-1]
    print(f"\n=== {title} ===")
    print("Top-5 predictions:")
    for idx in top5_idx:
        print(f"  {labels[idx]:30s} {probs[0][idx]*100:.2f}%")
    return logits

# --- PyTorch (python3.8) ---
torch_script = os.path.join(data_dir, "_torch_infer.py")
with open(torch_script, "w") as f:
    f.write("""
import torch, torchvision, numpy as np, sys
x = torch.from_numpy(np.load(sys.argv[1])).cuda()
model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval().cuda()
with torch.no_grad():
    out = model(x)
np.save(sys.argv[2], out.cpu().numpy())
""")
torch_out_path = os.path.join(data_dir, "torch_out.npy")
print("Running PyTorch inference...")
subprocess.run(["python3.8", torch_script, os.path.join(data_dir, "real_input.npy"), torch_out_path], check=True)
torch_out = np.load(torch_out_path)
print_top5(torch_out, "PyTorch")

# --- IREE (iree-run-module) ---
vmfb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_iree", "mobilenetv2.vmfb")
if not os.path.exists(vmfb_path):
    print(f"\nIREE module not found at {vmfb_path}, run compile.sh first")
    sys.exit(1)

iree_out_path = os.path.join(data_dir, "iree_out.npy")
print("\nRunning IREE inference...")
result = subprocess.run(
    ["/home/bourram/iree-install/bin/iree-run-module",
     f"--module={vmfb_path}", "--device=cuda", "--function=main",
     f"--input=@{os.path.join(data_dir, 'real_input.npy')}",
     f"--output=@{iree_out_path}"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"IREE failed: {result.stderr}")
    sys.exit(1)

iree_out = np.load(iree_out_path)
print_top5(iree_out, "IREE")

# --- Compare ---
diff = np.abs(torch_out - iree_out).max()
print(f"\nMax absolute difference: {diff:.6f}")
