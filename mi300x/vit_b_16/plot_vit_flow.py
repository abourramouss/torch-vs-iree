"""Architectural flow diagram of ViT-B/16 showing the execution path
end-to-end, color-coded to match the semantic-role panels.

Left side: high-level pipeline (input → patch embed → ×12 transformer blocks
→ final LN → classifier). Right side: zoomed-in view of a single transformer
block, showing LayerNorm → Multi-Head Self-Attention → residual add →
LayerNorm → MLP → residual add.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Colors matching the semantic roles used in roles_detail.png
color = {
    "io":            "#0f172a",
    "patch_embed":   "#64748b",
    "layernorm":     "#eab308",
    "qkv_proj":      "#2563eb",
    "attention":     "#7c3aed",
    "attn_out_proj": "#10b981",
    "mlp_up":        "#f97316",
    "mlp_down":      "#e11d48",
    "classifier":   "#94a3b8",
    "residual":      "#475569",
    "block":         "#f1f5f9",
}

# Dispatch mapping: (label, O0_dispatches_ms, O3_dispatches_ms)
# Shown as sidebar annotations next to each architectural box.
dispatches = {
    "patch":    ("O0: d0 + d1  (0.03 ms)",    "O3: d0 + d1  (0.03 ms)"),
    "ln1":      ("O0: d2 + d3  (0.27 ms)*",   "O3: d2 | d6 | d9  (0.15 ms)*"),
    "qkv":      ("O0: d5 + d4 + d6  (85.86 ms)", "O3: d3  (0.36 ms)"),
    "att":      ("O0: d7  (6.42 ms)",         "O3: d4  (6.16 ms)"),
    "outp":     ("O0: d8  (0.39 ms)",         "O3: d5  (0.47 ms)"),
    "ln2":      ("O0: d2 + d3  (0.27 ms)*",   "O3: d2 | d6 | d9  (0.15 ms)*"),
    "mlp_up":   ("O0: d11  (0.43 ms)",        "O3: d7  (0.47 ms)"),
    "mlp_down": ("O0: d12  (1.39 ms)",        "O3: d8  (1.76 ms)"),
    "final_ln": ("O0: d2  (~0.005 ms)",       "O3: d2  (~0.005 ms)"),
    "cls_head": ("O0: d135 + d136 + d137  (0.014 ms)",
                 "O3: d86 + d87 + d88 + d89  (0.019 ms)"),
}

fig = plt.figure(figsize=(26, 20))
ax_l = fig.add_subplot(1, 2, 1)
ax_r = fig.add_subplot(1, 2, 2)
ax_l.set_xlim(0, 14)
ax_l.set_ylim(0, 100)
ax_l.axis("off")
ax_r.set_xlim(-1, 18)
ax_r.set_ylim(0, 100)
ax_r.axis("off")


def box(ax, x, y, w, h, text, face, edge="black", txt_color="white",
        fontsize=14, fontweight="bold", radius=0.35):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=txt_color)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.8, style="->"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=22, color=color, lw=lw,
    ))


# ============================================================
#                         LEFT: pipeline
# ============================================================
ax_l.set_title("ViT-B/16 — overall execution pipeline",
               fontsize=18, fontweight="bold", pad=16, loc="left")

L_X = 6.0
positions = {}

def add_stage(key, y, w, h, text, face, txt="white", fs=14,
              dispatch_key=None):
    box(ax_l, L_X, y, w, h, text, face, txt_color=txt, fontsize=fs)
    positions[key] = (L_X, y - h / 2, y + h / 2)
    if dispatch_key and dispatch_key in dispatches:
        o0, o3 = dispatches[dispatch_key]
        ax_l.text(L_X + w / 2 + 0.4, y + 0.55, o0, ha="left", va="center",
                  fontsize=11, color="#1e3a8a", fontfamily="monospace",
                  fontweight="bold")
        ax_l.text(L_X + w / 2 + 0.4, y - 0.55, o3, ha="left", va="center",
                  fontsize=11, color="#9a3412", fontfamily="monospace",
                  fontweight="bold")


# Input
add_stage("input", 94, 6.2, 4.2, "Input image\n1 × 3 × 224 × 224",
          color["io"], fs=14)

# Patch embed conv
add_stage("patch", 84, 6.8, 5.6,
          "Patch embed  (conv 16×16, stride 16)\n→ 196 patches × 768",
          color["patch_embed"], fs=13, dispatch_key="patch")

# Class token + positional embedding
add_stage("cls", 75.5, 6.8, 4.2,
          "+ class token + pos embedding\n→ 197 × 768",
          color["patch_embed"], fs=13)

# Transformer block loop (represent as one big box with ×12)
block_top = 68
block_bot = 22
box_h = block_top - block_bot
ax_l.add_patch(mpatches.Rectangle(
    (L_X - 3.8, block_bot), 7.6, box_h,
    facecolor=color["block"], edgecolor="#334155",
    linewidth=1.8, linestyle="--",
))
ax_l.text(L_X, block_top - 2.5, "Transformer block   × 12",
          ha="center", va="center", fontsize=15, fontweight="bold",
          color="#0f172a")

# Simplified inside-block representation (left panel stays compact; full
# dispatch annotations appear in the right panel)
add_stage("ln1", 62, 6.2, 3.2, "LayerNorm",
          color["layernorm"], txt="black", fs=13)
add_stage("mhsa", 55, 6.2, 5.4,
          "Multi-Head Self-Attention\n(12 heads × 64)",
          color["attention"], fs=13)
add_stage("res1", 47.6, 3.2, 2.6, "+  residual",
          color["residual"], fs=12)
add_stage("ln2", 40.6, 6.2, 3.2, "LayerNorm",
          color["layernorm"], txt="black", fs=13)
add_stage("mlp", 33.3, 6.2, 5.4, "MLP\n768 → 3072 → 768",
          color["mlp_up"], fs=13)
add_stage("res2", 26.0, 3.2, 2.6, "+  residual",
          color["residual"], fs=12)
# Note on the left panel's block: detail is on the right.
ax_l.text(L_X - 4.0, (block_top + block_bot) / 2 - 4,
          "see right panel\nfor per-box\ndispatches",
          ha="right", va="center", fontsize=11, color="#64748b",
          style="italic")

# Loopback arrow (visual hint that block repeats)
ax_l.add_patch(FancyArrowPatch(
    (L_X + 3.9, 26.0), (L_X + 3.9, 62),
    arrowstyle="->", mutation_scale=22, color="#334155", lw=1.8,
    connectionstyle="arc3,rad=0.55",
))
ax_l.text(L_X + 4.8, 44, "×12", fontsize=16, fontweight="bold",
          color="#334155")

# Final LN
add_stage("final_ln", 16, 6.6, 3.2, "Final LayerNorm",
          color["layernorm"], txt="black", fs=13, dispatch_key="final_ln")
# Classifier
add_stage("cls_head", 8.8, 6.8, 3.6,
          "Classifier head  (Linear 768 → 1000)",
          color["classifier"], txt="black", fs=13, dispatch_key="cls_head")
# Output
add_stage("output", 2.0, 5.4, 3.2, "Logits  1 × 1000",
          color["io"], fs=14)

# Connect stages with arrows
stage_chain = [
    "input", "patch", "cls", "ln1", "mhsa", "res1",
    "ln2", "mlp", "res2", "final_ln", "cls_head", "output",
]
for a, b in zip(stage_chain[:-1], stage_chain[1:]):
    xa, ya_bot, ya_top = positions[a]
    xb, yb_bot, yb_top = positions[b]
    arrow(ax_l, L_X, ya_bot, L_X, yb_top, color="#1f2937", lw=1.3)

# ============================================================
#              RIGHT: zoomed single transformer block
# ============================================================
ax_r.set_title("One transformer block — internal dispatches",
               fontsize=18, fontweight="bold", pad=16, loc="left")

R_X = 6.0
rpos = {}

def radd(key, y, w, h, text, face, txt="white", fs=13, dispatch_key=None):
    box(ax_r, R_X, y, w, h, text, face, txt_color=txt, fontsize=fs)
    rpos[key] = (R_X, y - h / 2, y + h / 2)
    if dispatch_key and dispatch_key in dispatches:
        o0, o3 = dispatches[dispatch_key]
        ax_r.text(R_X + w / 2 + 0.4, y + 0.9, o0, ha="left", va="center",
                  fontsize=12, color="#1e3a8a", fontfamily="monospace",
                  fontweight="bold")
        ax_r.text(R_X + w / 2 + 0.4, y - 0.9, o3, ha="left", va="center",
                  fontsize=12, color="#9a3412", fontfamily="monospace",
                  fontweight="bold")


# Input: 197 × 768
radd("x_in", 94, 5.2, 3.4, "x   (197 × 768)", color["io"], fs=14)

# LN1
radd("ln1", 85.5, 6.0, 3.4, "LayerNorm",
     color["layernorm"], txt="black", fs=14, dispatch_key="ln1")

# Q/K/V projection (+bias)
radd("qkv", 76, 7.6, 4.6,
     "Q, K, V projection\n(Linear 768 → 768×3 + bias)",
     color["qkv_proj"], fs=13, dispatch_key="qkv")

# Attention
radd("att", 66.5, 7.6, 4.6,
     "softmax(Q·Kᵀ / √d)·V\n12 heads × 64", color["attention"], fs=13,
     dispatch_key="att")

# Out projection
radd("outp", 57, 7.6, 3.4,
     "Output projection  (Linear 768 → 768)",
     color["attn_out_proj"], fs=13, dispatch_key="outp")

# Residual 1
radd("res1", 49, 4.4, 2.8, "+  residual",
     color["residual"], fs=13)

# LN2
radd("ln2", 40, 6.0, 3.4, "LayerNorm",
     color["layernorm"], txt="black", fs=14, dispatch_key="ln2")

# MLP up
radd("mlp_up", 31, 7.6, 4.2,
     "MLP up  (Linear 768 → 3072 + bias)",
     color["mlp_up"], fs=13, dispatch_key="mlp_up")

# GELU (piece of MLP, same color as mlp_up lightened)
radd("gelu", 23, 4.4, 2.8, "GELU", "#fdba74", txt="black", fs=13)

# MLP down
radd("mlp_down", 14, 7.6, 4.2,
     "MLP down  (Linear 3072 → 768 + bias)",
     color["mlp_down"], fs=13, dispatch_key="mlp_down")

# Residual 2
radd("res2", 6, 4.4, 2.8, "+  residual",
     color["residual"], fs=13)

rchain = ["x_in", "ln1", "qkv", "att", "outp", "res1",
          "ln2", "mlp_up", "gelu", "mlp_down", "res2"]
for a, b in zip(rchain[:-1], rchain[1:]):
    xa, ya_bot, ya_top = rpos[a]
    xb, yb_bot, yb_top = rpos[b]
    arrow(ax_r, R_X, ya_bot, R_X, yb_top, color="#1f2937", lw=1.3)

# Residual skip lines on the left side of the column (to avoid the
# dispatch annotations on the right).
skip_x = -0.3
# First skip: x_in → res1
ax_r.plot([R_X - 2.6, skip_x], [rpos["x_in"][1], rpos["x_in"][1]],
          color="#475569", lw=1.8)
ax_r.plot([skip_x, skip_x], [rpos["res1"][1] + 1.5, rpos["x_in"][1]],
          color="#475569", lw=1.8)
arrow(ax_r, skip_x, rpos["res1"][1] + 1.5, R_X - 2.4, rpos["res1"][1] + 1.5,
      color="#475569", lw=1.8)
ax_r.text(skip_x - 0.15, (rpos["x_in"][1] + rpos["res1"][1]) / 2,
          "skip", fontsize=12, color="#475569", rotation=90, va="center",
          ha="right", fontweight="bold")

# Second skip: res1 → res2
ax_r.plot([R_X - 2.4, skip_x], [rpos["res1"][1], rpos["res1"][1]],
          color="#475569", lw=1.8)
ax_r.plot([skip_x, skip_x], [rpos["res2"][1] + 1.5, rpos["res1"][1]],
          color="#475569", lw=1.8)
arrow(ax_r, skip_x, rpos["res2"][1] + 1.5, R_X - 2.4, rpos["res2"][1] + 1.5,
      color="#475569", lw=1.8)
ax_r.text(skip_x - 0.15, (rpos["res1"][1] + rpos["res2"][1]) / 2,
          "skip", fontsize=12, color="#475569", rotation=90, va="center",
          ha="right", fontweight="bold")

# Legend (shared): map each box color back to the semantic role name used
# in roles_detail.png
legend_handles = [
    mpatches.Patch(facecolor=color["patch_embed"],   label="patch_embed"),
    mpatches.Patch(facecolor=color["layernorm"],     label="layernorm"),
    mpatches.Patch(facecolor=color["qkv_proj"],      label="qkv_proj"),
    mpatches.Patch(facecolor=color["attention"],     label="attention"),
    mpatches.Patch(facecolor=color["attn_out_proj"], label="attn_out_proj"),
    mpatches.Patch(facecolor=color["mlp_up"],        label="mlp_up"),
    mpatches.Patch(facecolor=color["mlp_down"],      label="mlp_down"),
    mpatches.Patch(facecolor=color["classifier"],    label="classifier"),
    mpatches.Patch(facecolor=color["residual"],      label="residual add"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=9,
           fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.0))

fig.suptitle(
    "ViT-B/16 architecture with IREE dispatch mapping  (O0 vs O3)\n"
    "blue = O0 dispatches ·  orange = O3 dispatches ·  * starred lines "
    "show role totals over all 25 LN executions",
    fontsize=18, fontweight="bold", y=0.995,
)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vit_flow.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
