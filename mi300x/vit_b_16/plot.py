"""Plot per-iteration time comparison for IREE O0, IREE O3, and PyTorch (ROCm)
on ViT-B/16 bs=1, MI300X (gfx942).

Numbers come from:
  - GPU kernel time: rocprofv3 SQLite results.db (SUM of kernels.duration).
  - Wall clock: python loop runner_iree.py / runner_torch.py (N=100).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Per-iteration measurements (ms)
configs = ["IREE O0", "IREE O3", "PyTorch\nrocm6.4"]
gpu_time = [94.80, 9.41, 2.19]          # from rocprofv3 kernel duration sum / 55 iters
wall_clock = [98.80, 13.73, 2.44]       # from runner_*.py wall clock / 100 iters
py_overhead = [w - g for w, g in zip(wall_clock, gpu_time)]

# Top-kernel breakdown (ms per iter) for O3 vs torch
kernels = {
    "attention":     {"O3": 6.16, "torch": 0.41},
    "Q/K/V + proj":  {"O3": 0.82, "torch": 0.55},  # Q/K/V matmul + output proj
    "MLP up":        {"O3": 1.76, "torch": 0.36},
    "MLP down":      {"O3": 0.47, "torch": 0.29},
    "other":         {"O3": 0.20, "torch": 0.58},  # layernorm, elementwise, etc.
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# ---------- Panel 1: per-iter time, log scale ----------
x = np.arange(len(configs))
w = 0.55

bars_gpu = ax1.bar(x, gpu_time, w, label="GPU kernel time",
                   color="#3b82f6", edgecolor="black", linewidth=0.7)
bars_ov = ax1.bar(x, py_overhead, w, bottom=gpu_time, label="Python harness overhead",
                  color="#e5e7eb", edgecolor="black", linewidth=0.7, hatch="//")

ax1.set_yscale("log")
ax1.set_ylabel("Time per inference (ms)", fontsize=11)
ax1.set_title("ViT-B/16 bs=1 on MI300X — per-iteration time\n(log scale)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=10)
ax1.grid(axis="y", which="both", linestyle=":", alpha=0.5)
ax1.set_axisbelow(True)
ax1.legend(loc="upper right", framealpha=0.95)

for i, (g, w_) in enumerate(zip(gpu_time, wall_clock)):
    ax1.annotate(f"{g:.2f} ms", xy=(i, g), xytext=(0, -12),
                 textcoords="offset points", ha="center", fontsize=9,
                 color="white", fontweight="bold")
    ax1.annotate(f"wall {w_:.2f} ms", xy=(i, w_), xytext=(0, 4),
                 textcoords="offset points", ha="center", fontsize=9)

# Speedup annotations
ax1.annotate("", xy=(1, 94.80), xytext=(0, 94.80),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5))
ax1.text(0.5, 120, "O3: 10.1× faster", ha="center", color="#16a34a",
         fontsize=10, fontweight="bold")
ax1.text(1.5, 12, "torch: 4.3× faster\nthan O3", ha="center", color="#dc2626",
         fontsize=10, fontweight="bold")

# ---------- Panel 2: kernel breakdown for O3 vs torch ----------
kernel_names = list(kernels.keys())
o3_vals = [kernels[k]["O3"] for k in kernel_names]
torch_vals = [kernels[k]["torch"] for k in kernel_names]

x2 = np.arange(len(kernel_names))
w2 = 0.38

palette_o3 = "#f97316"
palette_torch = "#14b8a6"

ax2.bar(x2 - w2/2, o3_vals, w2, label="IREE O3", color=palette_o3,
        edgecolor="black", linewidth=0.6)
ax2.bar(x2 + w2/2, torch_vals, w2, label="PyTorch", color=palette_torch,
        edgecolor="black", linewidth=0.6)

ax2.set_ylabel("GPU time per inference (ms)", fontsize=11)
ax2.set_title("Per-kernel GPU time breakdown: IREE O3 vs PyTorch", fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(kernel_names, fontsize=10)
ax2.grid(axis="y", linestyle=":", alpha=0.5)
ax2.set_axisbelow(True)
ax2.legend(loc="upper right", framealpha=0.95)

# Annotate bars with values
for i, (a, b) in enumerate(zip(o3_vals, torch_vals)):
    ax2.annotate(f"{a:.2f}", xy=(i - w2/2, a), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=8)
    ax2.annotate(f"{b:.2f}", xy=(i + w2/2, b), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=8)

ax2.axhline(0, color="black", lw=0.6)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
