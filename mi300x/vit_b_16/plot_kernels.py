"""Plot kernel launches per inference for IREE O0, IREE O3, and PyTorch.

Numbers come from rocprofv3 SQLite results.db (COUNT of kernel dispatches),
divided by 55 iterations (50 measured + 5 warmup).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

configs = ["IREE O0", "IREE O3", "PyTorch\nrocm6.4"]
kernels_per_iter = [139, 91, 129]
colors = ["#3b82f6", "#f97316", "#14b8a6"]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(configs))
ax.bar(x, kernels_per_iter, 0.55, color=colors,
       edgecolor="black", linewidth=0.8)

ax.set_ylabel("Kernel launches per inference", fontsize=11)
ax.set_title("ViT-B/16 bs=1 on MI300X — GPU kernel launches per inference",
             fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0, max(kernels_per_iter) * 1.15)

for i, n in enumerate(kernels_per_iter):
    ax.annotate(f"{n}", xy=(i, n), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=12,
                fontweight="bold")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
