"""Per-semantic-role GPU time for IREE O0 vs O3 on ViT-B/16 bs=1, MI300X.

Groups individual dispatches by what they actually compute (Q/K/V projection,
attention, MLP up, etc.) so O0 and O3 can be compared apples-to-apples, even
when the compiler has fused/split dispatches.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Semantic role → list of (config, dispatch_name, ms_per_iter)
# Totals from rocprofv3 / 55 iterations.
mapping = {
    "patch_embed": [
        ("O0", "dispatch_0_conv_768x14x14x3x16x16", 0.027),
        ("O0", "dispatch_1_elementwise_151296",     0.005),
        ("O3", "dispatch_0_conv_768x14x14x3x16x16", 0.027),
        ("O3", "dispatch_1_elementwise_151296",     0.005),
    ],
    "layernorm": [
        ("O0", "dispatch_2_reduction_197x768",      0.127),
        ("O0", "dispatch_3_reduction_197x768",      0.142),
        ("O3", "dispatch_2_reduction_197x768",      0.005),
        ("O3", "dispatch_6_reduction_197x768",      0.087),
        ("O3", "dispatch_9_reduction_197x768",      0.054),
    ],
    "qkv_proj + bias": [
        ("O0", "dispatch_4_elementwise_broadcast_197x768x2304", 34.419),
        ("O0", "dispatch_5_batch_matmul_197x1x2304x768",        51.375),
        ("O0", "dispatch_6_elementwise_transpose_197x3x768",     0.062),
        ("O3", "dispatch_3_matmul_like_3x197x768x768",           0.361),
    ],
    "attention": [
        ("O0", "dispatch_7_attention_12x197x64x64x197", 6.420),
        ("O3", "dispatch_4_attention_12x197x64x64x197", 6.164),
    ],
    "attn_out_proj": [
        ("O0", "dispatch_8_matmul_197x768x768",       0.388),
        ("O3", "dispatch_5_matmul_like_197x768x768",  0.465),
    ],
    "mlp_up (768→3072)": [
        ("O0", "dispatch_11_batch_matmul_1x197x3072x768", 0.428),
        ("O3", "dispatch_7_matmul_like_197x3072x768",     0.466),
    ],
    "mlp_down (3072→768)": [
        ("O0", "dispatch_12_batch_matmul_1x197x768x3072", 1.388),
        ("O3", "dispatch_8_matmul_like_197x768x3072",     1.756),
    ],
    "classifier_head": [
        ("O0", "dispatch_135_reduction_197x768",   0.005),
        ("O0", "dispatch_136_elementwise_1x768",   0.005),
        ("O0", "dispatch_137_matmul_1x1000x768",   0.004),
        ("O3", "dispatch_86_reduction_197x768",    0.005),
        ("O3", "dispatch_87_reduction_197x768",    0.005),
        ("O3", "dispatch_88_elementwise_768",      0.005),
        ("O3", "dispatch_89_matvec_like_1000x768", 0.004),
    ],
    "runtime_copy": [
        ("O0", "__amd_rocclr_copyBuffer", 0.005),
        ("O3", "__amd_rocclr_copyBuffer", 0.005),
    ],
}

roles = list(mapping.keys())
o0_totals = [sum(ms for cfg, _, ms in mapping[r] if cfg == "O0") for r in roles]
o3_totals = [sum(ms for cfg, _, ms in mapping[r] if cfg == "O3") for r in roles]
o0_counts = [sum(1 for cfg, _, _ in mapping[r] if cfg == "O0") for r in roles]
o3_counts = [sum(1 for cfg, _, _ in mapping[r] if cfg == "O3") for r in roles]

x = np.arange(len(roles))
w = 0.38

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

# ---- Panel 1: linear scale ----
ax1.bar(x - w/2, o0_totals, w, label="IREE O0", color="#3b82f6",
        edgecolor="black", linewidth=0.6)
ax1.bar(x + w/2, o3_totals, w, label="IREE O3", color="#f97316",
        edgecolor="black", linewidth=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(roles, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("GPU time per inference (ms)", fontsize=11)
ax1.set_title("Per semantic role — linear scale (O0 dominated by Q/K/V)",
              fontsize=12)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.set_axisbelow(True)
ax1.legend(loc="upper right")

for i, (a, b, ac, bc) in enumerate(zip(o0_totals, o3_totals,
                                       o0_counts, o3_counts)):
    ax1.annotate(f"{a:.2f}\n{ac}k", xy=(i - w/2, a), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=8)
    ax1.annotate(f"{b:.2f}\n{bc}k", xy=(i + w/2, b), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=8)

# ---- Panel 2: log scale (shows small roles too) ----
ax2.bar(x - w/2, o0_totals, w, label="IREE O0", color="#3b82f6",
        edgecolor="black", linewidth=0.6)
ax2.bar(x + w/2, o3_totals, w, label="IREE O3", color="#f97316",
        edgecolor="black", linewidth=0.6)
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(roles, rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("GPU time per inference (ms, log)", fontsize=11)
ax2.set_title("Per semantic role — log scale (shows all roles clearly)",
              fontsize=12)
ax2.grid(axis="y", which="both", linestyle=":", alpha=0.4)
ax2.set_axisbelow(True)
ax2.legend(loc="upper right")

# Speedup annotations on log panel
for i, (a, b) in enumerate(zip(o0_totals, o3_totals)):
    if a > 0 and b > 0:
        ratio = a / b
        if ratio > 1.5 or ratio < 0.67:
            color = "#16a34a" if ratio > 1 else "#dc2626"
            sign = f"{ratio:.0f}× faster" if ratio > 1 else f"{1/ratio:.2f}× slower"
            ax2.annotate(sign, xy=(i, max(a, b) * 1.4), ha="center",
                         fontsize=8, color=color, fontweight="bold")

fig.suptitle(
    "ViT-B/16 bs=1 on MI300X — GPU time by semantic role (O0 vs O3)",
    fontsize=14, fontweight="bold", y=0.995,
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "roles.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
