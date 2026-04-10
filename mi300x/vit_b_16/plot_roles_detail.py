"""Per-role breakdown showing individual dispatches that compose each role.

One subplot per semantic role. Each subplot has two side-by-side mini-bars
(O0 and O3) rendered on *independent y-axes*, so a role where O0 is 85 ms
and O3 is 0.36 ms still shows both compositions clearly.

Segment labels: dispatch_name / ms per iter × calls per iter.
Subplot title: role name and × roles per inference.
"""
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import to_rgb

# (dispatch_name, ms_per_iter, calls_per_iter)
data = {
    "patch_embed": {
        "O0": [
            ("dispatch_0_conv_768x14x14x3x16x16", 0.027, 1),
            ("dispatch_1_elementwise_151296",     0.005, 1),
        ],
        "O3": [
            ("dispatch_0_conv_768x14x14x3x16x16", 0.027, 1),
            ("dispatch_1_elementwise_151296",     0.005, 1),
        ],
    },
    "layernorm": {
        "O0": [
            ("dispatch_2_reduction_197x768", 0.127, 25),
            ("dispatch_3_reduction_197x768", 0.142, 24),
        ],
        "O3": [
            ("dispatch_2_reduction_197x768", 0.005,  1),
            ("dispatch_6_reduction_197x768", 0.087, 12),
            ("dispatch_9_reduction_197x768", 0.054, 11),
        ],
    },
    "qkv_proj + bias": {
        "O0": [
            ("dispatch_5_batch_matmul_197x1x2304x768",        51.375, 12),
            ("dispatch_4_elementwise_broadcast_197x768x2304", 34.419, 12),
            ("dispatch_6_elementwise_transpose_197x3x768",     0.062, 12),
        ],
        "O3": [
            ("dispatch_3_matmul_like_3x197x768x768", 0.361, 12),
        ],
    },
    "attention": {
        "O0": [("dispatch_7_attention_12x197x64x64x197", 6.420, 12)],
        "O3": [("dispatch_4_attention_12x197x64x64x197", 6.164, 12)],
    },
    "attn_out_proj": {
        "O0": [("dispatch_8_matmul_197x768x768",       0.388, 12)],
        "O3": [("dispatch_5_matmul_like_197x768x768",  0.465, 12)],
    },
    "mlp_up (768→3072)": {
        "O0": [("dispatch_11_batch_matmul_1x197x3072x768", 0.428, 12)],
        "O3": [("dispatch_7_matmul_like_197x3072x768",     0.466, 12)],
    },
    "mlp_down (3072→768)": {
        "O0": [("dispatch_12_batch_matmul_1x197x768x3072", 1.388, 12)],
        "O3": [("dispatch_8_matmul_like_197x768x3072",     1.756, 12)],
    },
    "classifier_head": {
        "O0": [
            ("dispatch_135_reduction_197x768", 0.005, 1),
            ("dispatch_136_elementwise_1x768", 0.005, 1),
            ("dispatch_137_matmul_1x1000x768", 0.004, 1),
        ],
        "O3": [
            ("dispatch_86_reduction_197x768",    0.005, 1),
            ("dispatch_87_reduction_197x768",    0.005, 1),
            ("dispatch_88_elementwise_768",      0.005, 1),
            ("dispatch_89_matvec_like_1000x768", 0.004, 1),
        ],
    },
    "runtime_copy": {
        "O0": [("__amd_rocclr_copyBuffer", 0.005, 1)],
        "O3": [("__amd_rocclr_copyBuffer", 0.005, 1)],
    },
}

role_base_color = {
    "patch_embed":         "#64748b",
    "layernorm":           "#eab308",
    "qkv_proj + bias":     "#2563eb",
    "attention":           "#7c3aed",
    "attn_out_proj":       "#10b981",
    "mlp_up (768→3072)":   "#f97316",
    "mlp_down (3072→768)": "#e11d48",
    "classifier_head":     "#94a3b8",
    "runtime_copy":        "#cbd5e1",
}

role_iters = {
    "patch_embed":         1,
    "layernorm":           25,
    "qkv_proj + bias":     12,
    "attention":           12,
    "attn_out_proj":       12,
    "mlp_up (768→3072)":   12,
    "mlp_down (3072→768)": 12,
    "classifier_head":     1,
    "runtime_copy":        1,
}

def shades(base_hex, n):
    r, g, b = to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n == 1:
        return [base_hex]
    lights = np.linspace(max(0.25, l - 0.15), min(0.82, l + 0.28), n)
    return [colorsys.hls_to_rgb(h, li, s) for li in lights]


def short_name(name):
    """Trim dispatch_N_role_shape to something compact."""
    n = name.replace("_f32", "")
    n = n.replace("main$async_", "")
    parts = n.split("_")
    if not parts[0].startswith("dispatch"):
        return n[:28]
    idx = parts[1] if len(parts) > 1 else "?"
    # Keep 1-2 descriptive tokens
    kind = "_".join(parts[2:4]) if len(parts) > 3 else "_".join(parts[2:])
    shape = parts[-1] if len(parts) > 4 else ""
    s = f"d{idx}_{kind}"
    if shape and shape not in kind:
        s += f"_{shape}"
    return s[:30]


def draw_stack(ax, entries, role_color, x, width=0.55):
    """Draw a single stacked bar and return (total_ms, label_items)."""
    entries = sorted(entries, key=lambda t: -t[1])
    palette = shades(role_color, max(len(entries), 1))
    total = sum(ms for _, ms, _ in entries)
    bot = 0.0
    label_items = []  # for side annotations
    for i, (name, ms, calls) in enumerate(entries):
        ax.bar(x, ms, width, bottom=bot, color=palette[i],
               edgecolor="black", linewidth=0.4)
        frac = ms / total if total > 0 else 0
        sn = short_name(name)
        if frac >= 0.20:
            # Inline label
            ax.text(x, bot + ms / 2, f"{sn}\n{ms:.3f} ms  ×{calls}",
                    ha="center", va="center", fontsize=7.2,
                    color="white" if i == 0 else "black",
                    fontweight="bold")
        label_items.append((sn, ms, calls, palette[i], bot + ms / 2))
        bot += ms
    return total, label_items


def draw_subplot(fig, gs_cell, role):
    """Draw one role panel with two independent-axis mini-bars."""
    d = data[role]
    base = role_base_color[role]
    inner = gs_cell.subgridspec(1, 2, wspace=0.05, width_ratios=[1, 1])
    ax0 = fig.add_subplot(inner[0, 0])
    ax1 = fig.add_subplot(inner[0, 1])

    t0, _ = draw_stack(ax0, d["O0"], base, x=0)
    t1, _ = draw_stack(ax1, d["O3"], base, x=0)

    for ax, total, lab in [(ax0, t0, "O0"), (ax1, t1, "O3")]:
        ymax = total * 1.35 if total > 0 else 0.01
        ax.set_ylim(0, ymax)
        ax.set_xticks([0])
        ax.set_xticklabels([lab], fontsize=9, fontweight="bold")
        ax.set_xlim(-0.5, 0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.text(0, total, f"{total:.3f} ms", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # Only the left axis shows the ylabel
    ax0.set_ylabel("ms/iter", fontsize=8)
    ax1.set_ylabel("ms/iter", fontsize=8)
    ax0.tick_params(axis="y", labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)

    # Title spanning the subplot pair — attach it to ax0 but centred between
    ax0.set_title(f"{role}   ×{role_iters[role]}/iter",
                  fontsize=10.5, fontweight="bold", color=base,
                  loc="left", pad=6)

    # Side legend for all dispatches (both O0 and O3) below the subplot
    all_entries = [("O0", e) for e in sorted(d["O0"], key=lambda t: -t[1])] + \
                  [("O3", e) for e in sorted(d["O3"], key=lambda t: -t[1])]
    palette_o0 = shades(base, max(len(d["O0"]), 1))
    palette_o3 = shades(base, max(len(d["O3"]), 1))
    lines = []
    for i, entry in enumerate(sorted(d["O0"], key=lambda t: -t[1])):
        name, ms, calls = entry
        lines.append((palette_o0[i], f"O0  {short_name(name)}  {ms:.3f} ms ×{calls}"))
    for i, entry in enumerate(sorted(d["O3"], key=lambda t: -t[1])):
        name, ms, calls = entry
        lines.append((palette_o3[i], f"O3  {short_name(name)}  {ms:.3f} ms ×{calls}"))

    # Text legend under the twin bars
    legend_ax = fig.add_subplot(inner[0, :])
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    legend_ax.set_frame_on(False)
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
    legend_ax.patch.set_alpha(0)
    # Push this behind the bars by setting zorder low
    legend_ax.set_zorder(-1)

    # Draw text beneath the bars
    y = -0.18
    for color, txt in lines:
        legend_ax.text(0.02, y, "■", color=color, fontsize=8, va="center")
        legend_ax.text(0.08, y, txt, fontsize=7, va="center")
        y -= 0.07


# --- Layout ---
roles = list(data.keys())
ncols = 3
nrows = (len(roles) + ncols - 1) // ncols

fig = plt.figure(figsize=(18, 16))
outer = fig.add_gridspec(nrows, ncols, hspace=0.75, wspace=0.28)

for i, role in enumerate(roles):
    r, c = divmod(i, ncols)
    draw_subplot(fig, outer[r, c], role)

fig.suptitle(
    "ViT-B/16 bs=1 on MI300X — dispatches per semantic role (O0 vs O3)",
    fontsize=14, fontweight="bold", y=0.995,
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "roles_detail.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
