import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. GEMM Kernel Time Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
backends = ['cuBLAS', 'PyTorch', 'IREE']
kernel_ms = [40.2, 40.2, 494.5]
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(backends, kernel_ms, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, kernel_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.1f} ms', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylabel('GPU Kernel Time (ms)', fontsize=13)
ax.set_title('GEMM 4096×4096 — GPU Kernel Time', fontsize=15, fontweight='bold')
ax.set_ylim(0, 580)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gemm_kernel_time.png'), dpi=150)
plt.close()

# ============================================================
# 2. GEMM Wall-Clock Time Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
wall_s = [1.094, 4.034, 0.781]
bars = ax.bar(backends, wall_s, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, wall_s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{val:.3f} s', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylabel('Wall-Clock Time (s)', fontsize=13)
ax.set_title('GEMM 4096×4096 — Total Execution Time', fontsize=15, fontweight='bold')
ax.set_ylim(0, 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gemm_wall_time.png'), dpi=150)
plt.close()

# ============================================================
# 3. NCU: cuBLAS vs IREE key metrics
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = [
    ('Compute (SM) %', [75.28, 11.14], '%'),
    ('IPC (executed)', [3.01, 0.27], ''),
    ('Occupancy %', [33.3, 16.7], '%'),
]
ncu_colors = ['#2ecc71', '#e74c3c']
ncu_labels = ['cuBLAS', 'IREE']
for ax, (title, vals, unit) in zip(axes, metrics):
    bars = ax.bar(ncu_labels, vals, color=ncu_colors, width=0.45, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, vals):
        label = f'{val}{unit}' if unit else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03,
                label, ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.suptitle('GEMM NCU Analysis — cuBLAS vs IREE', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gemm_ncu_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. NCU: Stall / Bottleneck comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# cuBLAS pie
cublas_stalls = [75.28, 24.71]
ax1.pie(cublas_stalls, labels=['Compute\n(active)', 'No eligible\nwarps'],
        colors=['#2ecc71', '#bdc3c7'], autopct='%.1f%%', startangle=90,
        textprops={'fontsize': 12})
ax1.set_title('cuBLAS — Scheduler', fontsize=13, fontweight='bold')

# IREE pie
iree_stalls = [6.74, 84.3, 8.96]
ax2.pie(iree_stalls, labels=['Compute\n(active)', 'Stalled on\nL1TEX', 'Other'],
        colors=['#e74c3c', '#e67e22', '#bdc3c7'], autopct='%.1f%%', startangle=90,
        textprops={'fontsize': 12})
ax2.set_title('IREE — Scheduler', fontsize=13, fontweight='bold')

fig.suptitle('GEMM — Where GPU Time Is Spent', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'gemm_stall_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. MobileNetV2 Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
mv2_labels = ['PyTorch', 'IREE']
mv2_colors = ['#3498db', '#e74c3c']

# GPU kernel time
ax = axes[0]
vals = [1.6, 13.8]
bars = ax.bar(mv2_labels, vals, color=mv2_colors, width=0.45, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val} ms', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_title('GPU Kernel Time', fontsize=13, fontweight='bold')
ax.set_ylabel('ms')
ax.set_ylim(0, 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Kernel launches
ax = axes[1]
vals = [160, 55]
bars = ax.bar(mv2_labels, vals, color=mv2_colors, width=0.45, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_title('Kernel Launches', fontsize=13, fontweight='bold')
ax.set_ylim(0, 200)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Wall-clock
ax = axes[2]
vals = [6.4, 0.1]
bars = ax.bar(mv2_labels, vals, color=mv2_colors, width=0.45, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val} s', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_title('Wall-Clock Time', fontsize=13, fontweight='bold')
ax.set_ylabel('s')
ax.set_ylim(0, 8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('MobileNetV2 — PyTorch vs IREE', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mobilenetv2_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. MobileNetV2 — 100 Inferences (Warm)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Per-inference latency
vals = [9.68, 10.17]
bars = ax1.bar(mv2_labels, vals, color=mv2_colors, width=0.45, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val} ms', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax1.set_title('Average Latency (1000 inferences)', fontsize=13, fontweight='bold')
ax1.set_ylabel('ms / inference')
ax1.set_ylim(0, 14)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Cold vs Warm comparison
categories = ['Cold Start\n(1 inference)', 'Warm\n(avg of 1000)']
torch_vals = [6400, 9.68]
iree_vals = [100, 10.17]
x_pos = np.arange(len(categories))
w = 0.3
bars1 = ax2.bar(x_pos - w/2, torch_vals, w, label='PyTorch', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax2.bar(x_pos + w/2, iree_vals, w, label='IREE', color='#e74c3c', edgecolor='black', linewidth=0.8)
ax2.set_yscale('log')
ax2.set_ylabel('Time (ms, log scale)')
ax2.set_title('Cold Start vs Warm Inference', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories)
ax2.legend()
for bar, val in zip(bars1, torch_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
             f'{val} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
for bar, val in zip(bars2, iree_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
             f'{val} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('MobileNetV2 — Startup Amortization', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mobilenetv2_warm_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. MobileNetV2 — Kernel Names (Top 10 by time)
# ============================================================

# PyTorch top kernels (from nsys cuda_gpu_kern_sum)
torch_kernels = [
    ('bn_fw_inf_1C11_kernel_NCHW', 440.4, 52),
    ('conv_depthwise2d_forward', 401.5, 17),
    ('clamp_scalar (ReLU6)', 204.3, 35),
    ('cutlass_tensorop_s1688gemm\n64x64_16x6', 119.9, 14),
    ('cutlass_tensorop_s1688gemm\n128x64_16x6', 51.8, 4),
    ('cutlass_tensorop_s1688gemm\n64x64_32x6', 48.7, 4),
    ('cutlass_tensorop_s1688gemm\n64x128_32x3', 44.8, 2),
    ('ampere_sgemm_32x128_nn', 42.4, 2),
    ('cutlass_tensorop_s1688gemm\n256x128_16x3', 40.9, 1),
    ('implicit_convolve_sgemm', 39.3, 1),
]

# IREE top kernels (from nsys cuda_gpu_kern_sum)
iree_kernels = [
    ('matmul_like\n160x49x960', 1745.9, 2),
    ('matmul_like\n96x112x112x16', 1472.9, 1),
    ('matmul_like\n96x196x576', 1145.6, 2),
    ('matmul_like\n320x49x960', 888.7, 1),
    ('conv_112x112x32\n3x3', 678.3, 1),
    ('matmul_like\n144x56x56x24', 651.1, 2),
    ('conv_56x56x96\n3x3', 629.4, 1),
    ('matmul_like\n160x49x576', 531.4, 1),
    ('matmul_like\n960x7x7x160', 499.3, 3),
    ('matmul_like\n1280x7x7x320', 422.8, 1),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# PyTorch
names = [k[0] for k in torch_kernels][::-1]
times = [k[1] for k in torch_kernels][::-1]
counts = [k[2] for k in torch_kernels][::-1]
y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, times, color='#3498db', edgecolor='black', linewidth=0.5, height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=9, fontfamily='monospace')
ax1.set_xlabel('Total Time (μs)', fontsize=11)
ax1.set_title(f'PyTorch — Top 10 Kernels\n({sum(c for _,_,c in torch_kernels)} calls shown)', fontsize=13, fontweight='bold')
for bar, t, c in zip(bars, times, counts):
    ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{t:.0f}μs ×{c}', va='center', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# IREE
names = [k[0] for k in iree_kernels][::-1]
times = [k[1] for k in iree_kernels][::-1]
counts = [k[2] for k in iree_kernels][::-1]
y_pos = np.arange(len(names))
bars = ax2.barh(y_pos, times, color='#e74c3c', edgecolor='black', linewidth=0.5, height=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names, fontsize=9, fontfamily='monospace')
ax2.set_xlabel('Total Time (μs)', fontsize=11)
ax2.set_title(f'IREE — Top 10 Dispatches\n({sum(c for _,_,c in iree_kernels)} calls shown)', fontsize=13, fontweight='bold')
for bar, t, c in zip(bars, times, counts):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'{t:.0f}μs ×{c}', va='center', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('MobileNetV2 — Kernel Breakdown', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'mobilenetv2_kernels.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 8. IREE Default vs O3 — Side by Side
# ============================================================

# Top 10 dispatches by time for both
dispatch_names = [
    'matmul_like\n160x49x960',
    'matmul_like\n96x112x112x16',
    'matmul_like\n96x196x576',
    'matmul_like\n320x49x960',
    'conv_112x112x32\n3x3',
    'matmul_like\n144x56x56x24',
    'conv_56x56x96\n3x3',
    'matmul_like\n160x49x576',
    'matmul_like\n960x7x7x160',
    'matmul_like\n1280x7x7x320',
]

default_us = [1745.9, 1472.9, 1145.6, 888.7, 678.3, 651.1, 629.4, 531.4, 499.3, 422.8]
o3_us =      [1763.6, 1502.5, 1145.7, 889.5, 596.8, 643.2, 668.6, 534.8, 496.5, 421.7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Side-by-side horizontal bars
y_pos = np.arange(len(dispatch_names))
names_rev = dispatch_names[::-1]

# Default
bars1 = ax1.barh(y_pos, default_us[::-1], color='#e74c3c', edgecolor='black', linewidth=0.5, height=0.7, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names_rev, fontsize=9, fontfamily='monospace')
ax1.set_xlabel('Time (μs)', fontsize=11)
ax1.set_title('IREE Default', fontsize=13, fontweight='bold')
for bar, val in zip(bars1, default_us[::-1]):
    ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va='center', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, 2100)

# O3
bars2 = ax2.barh(y_pos, o3_us[::-1], color='#9b59b6', edgecolor='black', linewidth=0.5, height=0.7, alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names_rev, fontsize=9, fontfamily='monospace')
ax2.set_xlabel('Time (μs)', fontsize=11)
ax2.set_title('IREE O3 + Aggressive Fusion', fontsize=13, fontweight='bold')
for bar, val in zip(bars2, o3_us[::-1]):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va='center', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, 2100)

fig.suptitle(f'IREE Default vs O3 — Top 10 Dispatches\n'
             f'Total: Default {sum(default_us)/1000:.2f}ms vs O3 {sum(o3_us)/1000:.2f}ms (1.00× speedup)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'iree_default_vs_o3.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 9. IREE Default vs O3 — IR Comparison Summary
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

ir_text = """IREE Default vs O3 — IR & Codegen Comparison

┌─────────────────────────┬──────────────────────┬──────────────────────┐
│                         │      Default         │       O3             │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Dispatch count          │         37           │         36           │
│ Pipeline                │  LLVMGPUTileAndFuse  │  LLVMGPUTileAndFuse  │
│ Workgroup size          │     [32, 8, 1]       │     [32, 8, 1]       │
│ Subgroup size           │         32           │         32           │
│ Shared memory           │        None          │        None          │
│ Tensor cores (MMA)      │        No            │        No            │
│ Bank conflict avoidance │        Yes           │        Yes           │
│ Total GPU kernel time   │     13.84 ms         │     13.82 ms         │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Differences             │ dispatch_18 (matmul) │ dispatch_32, 50      │
│                         │   merged into        │   (conv 14×14, 7×7)  │
│                         │   dispatch_24        │   split out separate │
└─────────────────────────┴──────────────────────┴──────────────────────┘

Conclusion: O3 flags only reorganized 2-3 dispatches (split/merge).
Same pipeline, same tiling, same workgroup sizes, no shared memory,
no tensor cores. Net effect on performance: ~0%."""

ax.text(0.05, 0.95, ir_text, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'iree_ir_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Charts saved:")
print("  gemm_kernel_time.png")
print("  gemm_wall_time.png")
print("  gemm_ncu_comparison.png")
print("  gemm_stall_analysis.png")
print("  mobilenetv2_comparison.png")
print("  mobilenetv2_warm_comparison.png")
print("  mobilenetv2_kernels.png")
print("  iree_default_vs_o3.png")
print("  iree_ir_comparison.png")
