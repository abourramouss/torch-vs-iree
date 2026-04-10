"""Per-dispatch GPU time for IREE O0 vs O3 on ViT-B/16 bs=1, MI300X.

Two vertical bar charts, one per config. Each bar = one kernel (by name),
height = ms per iteration. Numbers from rocprofv3 SQLite results.db
(SUM(duration) / 55 iterations).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# (name, calls_per_iter, ms_per_iter)  — sorted by ms descending
o0 = [
    ("dispatch_5_batch_matmul_197x1x2304x768",     12, 51.375),
    ("dispatch_4_elementwise_broadcast_197x768x2304", 12, 34.419),
    ("dispatch_7_attention_12x197x64x64x197",       12,  6.420),
    ("dispatch_12_batch_matmul_1x197x768x3072",     12,  1.388),
    ("dispatch_11_batch_matmul_1x197x3072x768",     12,  0.428),
    ("dispatch_8_matmul_197x768x768",               12,  0.388),
    ("dispatch_3_reduction_197x768",                24,  0.142),
    ("dispatch_2_reduction_197x768",                25,  0.127),
    ("dispatch_6_elementwise_transpose_197x3x768",  12,  0.062),
    ("dispatch_0_conv_768x14x14x3x16x16",            1,  0.027),
    ("dispatch_1_elementwise_151296",                1,  0.005),
    ("__amd_rocclr_copyBuffer",                      1,  0.005),
    ("dispatch_135_reduction_197x768",               1,  0.005),
    ("dispatch_136_elementwise_1x768",               1,  0.005),
    ("dispatch_137_matmul_1x1000x768",               1,  0.004),
]

o3 = [
    ("dispatch_4_attention_12x197x64x64x197",       12,  6.164),
    ("dispatch_8_matmul_like_197x768x3072",         12,  1.756),
    ("dispatch_7_matmul_like_197x3072x768",         12,  0.466),
    ("dispatch_5_matmul_like_197x768x768",          12,  0.465),
    ("dispatch_3_matmul_like_3x197x768x768",        12,  0.361),
    ("dispatch_6_reduction_197x768",                12,  0.087),
    ("dispatch_9_reduction_197x768",                11,  0.054),
    ("dispatch_0_conv_768x14x14x3x16x16",            1,  0.027),
    ("dispatch_2_reduction_197x768",                 1,  0.005),
    ("dispatch_1_elementwise_151296",                1,  0.005),
    ("dispatch_86_reduction_197x768",                1,  0.005),
    ("dispatch_87_reduction_197x768",                1,  0.005),
    ("__amd_rocclr_copyBuffer",                      1,  0.005),
    ("dispatch_88_elementwise_768",                  1,  0.005),
    ("dispatch_89_matvec_like_1000x768",             1,  0.004),
]

def short(name):
    return name.replace("main$async_", "").replace("_f32", "")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))

def draw(ax, entries, title, color):
    names = [short(n) for n, _, _ in entries]
    times = [ms for _, _, ms in entries]
    calls = [c for _, c, _ in entries]
    x = np.arange(len(entries))
    ax.bar(x, times, 0.7, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("GPU time per inference (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(times) * 1.15)
    for i, (t, c) in enumerate(zip(times, calls)):
        label = f"{t:.2f}\n×{c}"
        ax.annotate(label, xy=(i, t), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7)

draw(ax1, o0, "IREE O0 — total 94.80 ms/iter, 15 kernels, 139 launches/iter",
     "#3b82f6")
draw(ax2, o3, "IREE O3 — total 9.41 ms/iter, 15 kernels, 91 launches/iter",
     "#f97316")

fig.suptitle(
    "ViT-B/16 bs=1 on MI300X — per-dispatch GPU time",
    fontsize=14, fontweight="bold", y=1.00,
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dispatches.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
