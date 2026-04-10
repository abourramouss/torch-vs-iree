"""Sankey-style flow diagram showing how O0 dispatches map to O3 dispatches.

Each dispatch is a rectangle sized by GPU time per iteration. Ribbons connect
each O0 dispatch to its semantic destination in O3. Heights are normalized
per side (absolute totals shown separately) so the 10× speedup doesn't squash
the O3 side into invisibility.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
import os

# Per-dispatch data: (name, ms, role)
# Roles are the bridge between O0 and O3.
o0 = [
    ("dispatch_0_conv",                    0.027, "patch_embed"),
    ("dispatch_1_elementwise",             0.005, "patch_embed"),
    ("dispatch_2_reduction",               0.127, "layernorm"),
    ("dispatch_3_reduction",               0.142, "layernorm"),
    ("dispatch_4_bias_broadcast_197x2304", 34.419, "qkv_proj"),
    ("dispatch_5_batch_matmul_2304x768",   51.375, "qkv_proj"),
    ("dispatch_6_transpose_3x768",          0.062, "qkv_proj"),
    ("dispatch_7_attention",                6.420, "attention"),
    ("dispatch_8_matmul_197x768x768",       0.388, "attn_out_proj"),
    ("dispatch_11_bmm_3072x768",            0.428, "mlp_up"),
    ("dispatch_12_bmm_768x3072",            1.388, "mlp_down"),
    ("dispatch_135_reduction",              0.005, "classifier"),
    ("dispatch_136_elementwise",            0.005, "classifier"),
    ("dispatch_137_matmul_1x1000x768",      0.004, "classifier"),
    ("__amd_rocclr_copyBuffer",             0.005, "runtime"),
]

o3 = [
    ("dispatch_0_conv",                    0.027, "patch_embed"),
    ("dispatch_1_elementwise",             0.005, "patch_embed"),
    ("dispatch_2_reduction",               0.005, "layernorm"),
    ("dispatch_3_matmul_3x197x768x768",    0.361, "qkv_proj"),
    ("dispatch_4_attention",               6.164, "attention"),
    ("dispatch_5_matmul_197x768x768",      0.465, "attn_out_proj"),
    ("dispatch_6_reduction",               0.087, "layernorm"),
    ("dispatch_7_matmul_197x3072x768",     0.466, "mlp_up"),
    ("dispatch_8_matmul_197x768x3072",     1.756, "mlp_down"),
    ("dispatch_9_reduction",               0.054, "layernorm"),
    ("dispatch_86_reduction",              0.005, "classifier"),
    ("dispatch_87_reduction",              0.005, "classifier"),
    ("dispatch_88_elementwise",            0.005, "classifier"),
    ("dispatch_89_matvec_1000x768",        0.004, "classifier"),
    ("__amd_rocclr_copyBuffer",            0.005, "runtime"),
]

role_colors = {
    "patch_embed":   "#64748b",
    "layernorm":     "#fbbf24",
    "qkv_proj":      "#2563eb",
    "attention":     "#7c3aed",
    "attn_out_proj": "#10b981",
    "mlp_up":        "#f97316",
    "mlp_down":      "#e11d48",
    "classifier":    "#94a3b8",
    "runtime":       "#cbd5e1",
}

o0_total = sum(ms for _, ms, _ in o0)
o3_total = sum(ms for _, ms, _ in o3)

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(-0.02, 1.05)
ax.axis("off")

LEFT_X = 1.2
RIGHT_X = 8.8
BOX_W = 0.6
GAP = 0.003  # vertical gap between stacked rectangles

def draw_column(entries, x, total, label):
    y = 0.0
    positions = {}  # name → (y_bottom, y_top) in normalized coords
    for name, ms, role in entries:
        h = ms / total  # normalize
        rect = mpatches.Rectangle(
            (x - BOX_W / 2, y), BOX_W, h,
            facecolor=role_colors[role], edgecolor="black", linewidth=0.5,
        )
        ax.add_patch(rect)
        # Label — inside if big enough, outside if small
        if h > 0.025:
            ax.text(x, y + h / 2, f"{name}\n{ms:.2f} ms",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if role in {"qkv_proj", "attention", "mlp_down", "patch_embed"} else "black",
                    fontweight="bold")
        positions[name] = (y, y + h)
        y += h + GAP
    # Header
    ax.text(x, 1.035, f"{label}\n{total:.2f} ms/iter",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    return positions

o0_pos = draw_column(o0, LEFT_X, o0_total, "IREE O0")
o3_pos = draw_column(o3, RIGHT_X, o3_total, "IREE O3")

# Draw ribbons from each O0 dispatch to its O3 destination(s) by role
# Multiple O0 dispatches in the same role flow into the same O3 slot(s).
o3_by_role = {}
for name, ms, role in o3:
    o3_by_role.setdefault(role, []).append((name, ms))

# Allocate flow within destination role: each O0 dispatch gets a proportional
# slice of the total O3 role capacity (visual only — represents fusion).
o3_role_totals = {r: sum(ms for _, ms in lst) for r, lst in o3_by_role.items()}
o0_role_totals = {}
for _, ms, role in o0:
    o0_role_totals[role] = o0_role_totals.get(role, 0) + ms

# For each source, allocate offset within source rect and destination rect
o0_role_used = {r: 0.0 for r in o0_role_totals}
o3_role_used = {r: 0.0 for r in o3_role_totals}

for name, ms, role in o0:
    if role not in o3_by_role:
        continue
    # Source y span (from o0_pos) — use full box; allocate within role
    y0_bot, y0_top = o0_pos[name]
    # Destination: full span of all O3 dispatches in this role (cumulative)
    dest_y_bot = min(o3_pos[n][0] for n, _ in o3_by_role[role])
    dest_y_top = max(o3_pos[n][1] for n, _ in o3_by_role[role])
    # Allocate proportional slice of destination
    src_frac = ms / o0_role_totals[role]
    o3_used = o3_role_used[role]
    dest_height = dest_y_top - dest_y_bot
    d_bot = dest_y_bot + o3_used
    d_top = d_bot + src_frac * dest_height
    o3_role_used[role] += src_frac * dest_height

    # Bezier ribbon from (LEFT_X + BOX_W/2, [y0_bot, y0_top])
    # to (RIGHT_X - BOX_W/2, [d_bot, d_top])
    x0 = LEFT_X + BOX_W / 2
    x1 = RIGHT_X - BOX_W / 2
    xm = (x0 + x1) / 2

    verts_top = [(x0, y0_top), (xm, y0_top), (xm, d_top), (x1, d_top)]
    verts_bot = [(x1, d_bot), (xm, d_bot), (xm, y0_bot), (x0, y0_bot)]
    codes_top = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    codes_bot = [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts_top + verts_bot + [(x0, y0_top)],
                codes_top + codes_bot + [Path.CLOSEPOLY])
    patch = mpatches.PathPatch(
        path,
        facecolor=role_colors[role],
        edgecolor="none",
        alpha=0.28,
    )
    ax.add_patch(patch)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=r)
    for r, c in role_colors.items()
]
ax.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.5, -0.04),
          ncol=5, fontsize=9, frameon=False, title="semantic role")

fig.suptitle(
    "ViT-B/16 bs=1 on MI300X — dispatch flow from O0 to O3\n"
    "(height normalized per side; totals shown above columns)",
    fontsize=14, fontweight="bold", y=0.995,
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sankey.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
