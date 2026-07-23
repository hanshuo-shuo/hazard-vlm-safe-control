"""
Compare all methods: VLA variants vs VLM+PIVOT+Physics variants.
Generates a comprehensive figure for advisor meeting.
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
import numpy as np

# ========== Consolidated Results ==========
# All results from hazard_result_gpu experiments (1000-episode evals)

methods = [
    # --- VLA baselines (all Qwen2-VL-7B) ---
    "VLA (head only)\n1k demos",
    "VLA (head, img+state)\n1k demos",
    "VLA (head only)\n3k demos",
    "VLA (head only)\n5k demos",
    "VLA + LoRA\n1k demos",
    "VLA + Flow head\n1k demos",
    "VLA + Flow + LoRA\n1k demos",
    # --- Our methods ---
    "VLM+PIVOT\n(no physics, 7B)",
    "VLM+PIVOT+Physics\n(7B, no LoRA)",
    "VLM+PIVOT+Physics\n(32B, no LoRA)",
    "VLM+PIVOT+Physics\n+LoRA (7B)",
]

# success rates (heldout seeds where available, otherwise best)
success = [
    0.220,   # VLA head 1k (nosame1000 v1)
    0.369,   # VLA head img+state 1k (nosame1000 v2)
    0.413,   # VLA head 3k (data sweep heldout)
    0.360,   # VLA head 5k (nosame5000)
    0.634,   # VLA + LoRA 1k (nosame)
    0.100,   # VLA + Flow head only (heldout)
    0.661,   # VLA + Flow + LoRA (heldout)
    0.280,   # historical VLM+PIVOT no-physics run, 7B
    0.740,   # VLM+PIVOT+Physics 7B no LoRA (200ep, run 20260504_161135)
    0.850,   # VLM+PIVOT+Physics 32B no LoRA (100ep, run 20260504_002548)
    0.878,   # VLM+PIVOT+Physics+LoRA 7B (1000ep, run 20260504_193226)
]

hazard_hit = [
    0.712,
    0.604,
    0.580,
    0.628,
    0.366,
    0.675,
    0.339,
    0.720,
    0.260,   # 7B no LoRA
    0.140,   # 32B no LoRA
    0.121,   # 7B + LoRA
]

timeout = [
    0.068,
    0.027,
    0.007,
    0.012,
    0.000,
    0.225,
    0.000,
    0.060,
    0.000,   # 7B no LoRA
    0.010,   # 32B no LoRA
    0.001,   # 7B + LoRA
]

mean_clearance = [
    0.141,
    0.220,
    0.232,
    0.186,
    0.446,
    0.215,
    0.463,
    None,     # not recorded for plain PIVOT
    0.306,   # 7B no LoRA
    0.521,   # 32B no LoRA
    0.535,   # 7B + LoRA
]

mean_return = [
    -14.3,
    6.2,
    11.9,
    4.1,
    44.9,
    -24.9,
    48.9,
    30.4,
    60.5,    # 7B no LoRA
    77.8,    # 32B no LoRA
    81.5,    # 7B + LoRA
]

# Colors
colors_vla = ['#d9534f'] * 7   # red family for VLA
colors_ours = ['#5bc0de', '#66b3e0', '#5cb85c', '#2e8b57']  # blue=no physics, lightblue=7B physics, green=32B, darkgreen=LoRA
colors = colors_vla + colors_ours

# Category labels
is_ours = [False]*7 + [True]*4

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle("VLA vs. VLM+PIVOT+Physics: Comprehensive Comparison\n(VLA all Qwen2-VL-7B; Our methods: 7B unless noted)",
             fontsize=15, fontweight='bold', y=0.98)

x = np.arange(len(methods))
bar_width = 0.65

# --- Panel 1: Success Rate ---
ax = axes[0, 0]
bars = ax.bar(x, [s*100 for s in success], bar_width, color=colors, edgecolor='black', linewidth=0.5)
for i, (bar, s) in enumerate(zip(bars, success)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{s*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Success Rate (higher is better)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5)
ax.set_ylim(0, 108)
ax.axvspan(-0.5, 6.5, alpha=0.06, color='red', label='VLA methods')
ax.axvspan(6.5, 10.5, alpha=0.06, color='green', label='Our methods')
ax.legend(loc='upper left', fontsize=9)

# --- Panel 2: Hazard Hit Rate ---
ax = axes[0, 1]
bars = ax.bar(x, [h*100 for h in hazard_hit], bar_width, color=colors, edgecolor='black', linewidth=0.5)
for i, (bar, h) in enumerate(zip(bars, hazard_hit)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{h*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Hazard Hit Rate (%)', fontsize=12)
ax.set_title('Hazard Hit Rate (lower is better)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5)
ax.set_ylim(0, 85)
ax.axvspan(-0.5, 6.5, alpha=0.06, color='red')
ax.axvspan(6.5, 10.5, alpha=0.06, color='green')

# --- Panel 3: Mean Return ---
ax = axes[1, 0]
bars = ax.bar(x, mean_return, bar_width, color=colors, edgecolor='black', linewidth=0.5)
for i, (bar, r) in enumerate(zip(bars, mean_return)):
    ypos = max(bar.get_height(), 0) + 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{r:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Mean Return', fontsize=12)
ax.set_title('Mean Return (higher is better)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
ax.axvspan(-0.5, 6.5, alpha=0.06, color='red')
ax.axvspan(6.5, 10.5, alpha=0.06, color='green')

# --- Panel 4: Min Clearance ---
ax = axes[1, 1]
clearance_vals = [c if c is not None else 0 for c in mean_clearance]
clearance_colors = list(colors)
bars = ax.bar(x, clearance_vals, bar_width, color=clearance_colors, edgecolor='black', linewidth=0.5)
for i, (bar, c) in enumerate(zip(bars, mean_clearance)):
    if c is not None:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{c:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.02,
                'N/A', ha='center', va='bottom', fontsize=9, fontweight='bold', color='gray')
ax.set_ylabel('Mean Min Clearance', fontsize=12)
ax.set_title('Mean Min Clearance from Hazards (higher is safer)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5)
ax.set_ylim(0, 0.7)
ax.axvspan(-0.5, 6.5, alpha=0.06, color='red')
ax.axvspan(6.5, 10.5, alpha=0.06, color='green')

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = '/Users/hanshuo/Desktop/hazard/vla_vs_pivot_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

# ========== Figure 2: VLA Data Scaling vs Our Method ==========
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# VLA head only data scaling
data_sizes = [1000, 3000, 5000]
vla_head_success = [0.369, 0.413, 0.360]  # img+state version for 1k, rest are head only
vla_head_hazard = [0.604, 0.580, 0.628]

ax.plot(data_sizes, [s*100 for s in vla_head_success], 'ro-', linewidth=2, markersize=10, label='VLA (action head only)')
ax.axhline(y=63.4, color='orange', linestyle='--', linewidth=2, label='VLA + LoRA (1k demos): 63.4%')
ax.axhline(y=66.1, color='salmon', linestyle='-.', linewidth=2, label='VLA + Flow + LoRA (1k demos): 66.1%')
ax.axhline(y=74.0, color='#66b3e0', linestyle='--', linewidth=2.5, label='Ours: VLM+PIVOT+Physics 7B (no LoRA): 74.0%')
ax.axhline(y=85.0, color='#5cb85c', linestyle=':', linewidth=2, label='Ours: VLM+PIVOT+Physics 32B (no LoRA): 85.0%')
ax.axhline(y=87.8, color='#2e8b57', linestyle='-', linewidth=2.5, label='Ours: VLM+PIVOT+Physics+LoRA 7B: 87.8%')

ax.set_xlabel('Number of Expert Demonstrations', fontsize=13)
ax.set_ylabel('Success Rate (%)', fontsize=13)
ax.set_title('VLA Data Scaling vs. Our Method\n(More expert data barely helps VLA; our method needs zero expert demos)',
             fontsize=13, fontweight='bold')
ax.set_xticks(data_sizes)
ax.set_xlim(500, 5500)
ax.set_ylim(0, 100)
ax.legend(fontsize=9.5, loc='center left')
ax.grid(True, alpha=0.3)

# Add annotation
ax.annotate('7B+Physics+LoRA: 87.8%\n(0 expert demos, online learning)',
            xy=(3000, 87.8), xytext=(3500, 50),
            fontsize=11, fontweight='bold', color='#2e8b57',
            arrowprops=dict(arrowstyle='->', color='#2e8b57', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
ax.annotate('7B+Physics (no LoRA): 74%\nalready beats best VLA (66.1%)',
            xy=(1500, 74.0), xytext=(1800, 30),
            fontsize=10, fontweight='bold', color='#66b3e0',
            arrowprops=dict(arrowstyle='->', color='#66b3e0', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.3))

out_path2 = '/Users/hanshuo/Desktop/hazard/vla_data_scaling.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path2}")
plt.close()

# ========== Figure 3: LoRA Learning Curve (our method) ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# From analysis report
windows = ['First 100\neps', 'Middle\n(overall)', 'Last 100\neps']
success_curve = [68.0, 87.8, 90.0]
hazard_curve = [32.0, 12.1, 10.0]
teacher_agree = [31.3, 57.3, 84.4]
clearance_curve = [0.286, 0.535, 0.674]

ax = axes[0]
x_w = np.arange(len(windows))
w = 0.35
bars1 = ax.bar(x_w - w/2, success_curve, w, color='#5cb85c', label='Success %', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_w + w/2, hazard_curve, w, color='#d9534f', label='Hazard Hit %', edgecolor='black', linewidth=0.5)
for bar, val in zip(bars1, success_curve):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}%', ha='center', fontweight='bold', fontsize=10)
for bar, val in zip(bars2, hazard_curve):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}%', ha='center', fontweight='bold', fontsize=10)
ax.set_xticks(x_w)
ax.set_xticklabels(windows, fontsize=11)
ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Our Method: Online Learning Progress\n(VLM+PIVOT+Physics+LoRA)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 105)

ax2 = axes[1]
ax2.plot(x_w, teacher_agree, 'bo-', linewidth=2, markersize=10, label='VLM-Teacher Agreement %')
ax2.plot(x_w, [c*100 for c in clearance_curve], 'gs-', linewidth=2, markersize=10, label='Mean Min Clearance (×100)')
for i, (ta, cl) in enumerate(zip(teacher_agree, clearance_curve)):
    ax2.text(i, ta+2, f'{ta:.1f}%', ha='center', fontweight='bold', color='blue', fontsize=10)
    ax2.text(i, cl*100-4, f'{cl:.3f}', ha='center', fontweight='bold', color='green', fontsize=10)
ax2.set_xticks(x_w)
ax2.set_xticklabels(windows, fontsize=11)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_title('LoRA Distillation: Teacher Alignment & Safety\nImprove Over Time', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='center right')
ax2.set_ylim(0, 105)

plt.tight_layout()
out_path3 = '/Users/hanshuo/Desktop/hazard/lora_learning_curve.png'
plt.savefig(out_path3, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path3}")
plt.close()

# ========== Figure 4: Summary Table as Image ==========
fig, ax = plt.subplots(figsize=(30, 7))
ax.axis('off')

table_data = [
    ['VLA (head only, 1k demos)',           '1000', '7B',  '22.0%', '71.2%', '6.8%',  '-14.3', '0.141'],
    ['VLA (head, img+state, 1k)',           '1000', '7B',  '36.9%', '60.4%', '2.7%',  '6.2',   '0.220'],
    ['VLA (head only, 3k demos)',           '3000', '7B',  '41.3%', '58.0%', '0.7%',  '11.9',  '0.232'],
    ['VLA (head only, 5k demos)',           '5000', '7B',  '36.0%', '62.8%', '1.2%',  '4.1',   '0.186'],
    ['VLA + LoRA (1k demos)',               '1000', '7B',  '63.4%', '36.6%', '0.0%',  '44.9',  '0.446'],
    ['VLA + Flow head (1k demos)',          '1000', '7B',  '10.0%', '67.5%', '22.5%', '-24.9', '0.215'],
    ['VLA + Flow + LoRA (1k demos)',        '1000', '7B',  '66.1%', '33.9%', '0.0%',  '48.9',  '0.463'],
    ['VLM+PIVOT (no physics)',              '0',    '7B',  '28.0%', '72.0%', '6.0%',  '30.4',  'N/A'],
    ['VLM+PIVOT+Physics (no LoRA)',         '0',    '7B',  '74.0%', '26.0%', '0.0%',  '60.5',  '0.306'],
    ['VLM+PIVOT+Physics (no LoRA)',         '0',    '32B', '85.0%', '14.0%', '1.0%',  '77.8',  '0.521'],
    ['VLM+PIVOT+Physics+LoRA',              '0',    '7B',  '87.8%', '12.1%', '0.1%',  '81.5',  '0.535'],
]

col_labels = ['Method', 'Expert\nDemos', 'VLM\nSize', 'Success\nRate', 'Hazard\nHit', 'Timeout', 'Mean\nReturn', 'Min\nClearance']

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4a4a4a')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color VLA rows (red-ish) and our rows (green-ish)
for i in range(1, 8):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor('#ffe0e0')
for i in range(8, 12):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor('#e0ffe0')

# Bold best results (row 11 = VLM+PIVOT+Physics+LoRA 7B)
table[11, 3].set_text_props(fontweight='bold', color='darkgreen')  # best success
table[11, 4].set_text_props(fontweight='bold', color='darkgreen')  # best hazard
table[11, 6].set_text_props(fontweight='bold', color='darkgreen')  # best return
# Also highlight 7B no-LoRA row (row 9) to show it already beats best VLA
table[9, 3].set_text_props(fontweight='bold', color='darkblue')

# Note about expert demos for our method
ax.text(0.5, 0.02,
        "Note: Our VLM+PIVOT methods require 0 expert demonstrations — the physics model is learned online.\n"
        "VLA methods require 1k-5k expert demos collected from a hand-tuned expert policy.\n"
        "Red rows = VLA approaches | Green rows = Our approach (VLM+PIVOT+Physics)",
        ha='center', va='bottom', fontsize=10, style='italic',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_title("Complete Results Summary: VLA vs. VLM+PIVOT+Physics\n(Hazard Navigation, Qwen2-VL-7B Backbone)",
             fontsize=14, fontweight='bold', pad=20)

out_path4 = '/Users/hanshuo/Desktop/hazard/results_summary_table.png'
plt.savefig(out_path4, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path4}")
plt.close()

print("\nAll figures generated successfully!")
