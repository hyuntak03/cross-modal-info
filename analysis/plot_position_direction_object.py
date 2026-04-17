"""
Position/Direction/Object dim 분석 시각화.

Layout:
  Row 1: Within-task probe accuracy (Pos R², Dir acc, Obj acc per task, Vision Encoder)
  Row 2: Cross-task top-50 dim overlap matrices (Position / Direction / Object)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

with open("analysis/position_direction_object_dims.json") as f:
    data = json.load(f)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_LABELS = ["Shape-Color", "Obj-Color", "Shape-Place", "Obj-Place"]

# Use Vanilla Vision Encoder (모든 모델 동일 — frozen)
res = data["Vanilla"]["vision_encoder"]

TOP_K = 50
D_VE = 1152
RANDOM_EXPECT = TOP_K * TOP_K / D_VE  # 2.17


def overlap(dims_a, dims_b):
    return len(set(dims_a) & set(dims_b))


def build_overlap_matrix(attr):
    n = len(TASKS)
    M = np.zeros((n, n))
    for i, t1 in enumerate(TASKS):
        for j, t2 in enumerate(TASKS):
            d1 = res[t1][attr]["top_dims"]
            d2 = res[t2][attr]["top_dims"]
            M[i, j] = overlap(d1, d2)
    return M


# ============================================================
#  Figure
# ============================================================

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.2], hspace=0.45, wspace=0.3)

# Row 1: Within-task probe accuracy — 3 metrics side by side
attrs_info = [
    ("position", "Position R² (x, y avg)", "R²", "#2ecc71"),
    ("direction", "Direction Probe Acc (4-class, chance=25%)", "Acc (%)", "#e74c3c"),
    ("object", "Object/Shape Probe Acc", "Acc (%)", "#3498db"),
]

for col, (attr, title, ylabel, color) in enumerate(attrs_info):
    ax = fig.add_subplot(gs[0, col])
    vals = []
    for t in TASKS:
        r = res[t][attr]
        if attr == "position":
            v = (r["R2_x"] + r["R2_y"]) / 2
        else:
            v = r["acc"]
        vals.append(v)

    bars = ax.bar(range(len(TASKS)), vals, color=color, edgecolor="black", linewidth=1.2)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.2f}" if attr == "position" else f"{v:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")

    # chance line for direction
    if attr == "direction":
        ax.axhline(25, color="gray", linestyle=":", alpha=0.6, label="chance=25%")
        ax.legend(loc="lower left", fontsize=8)

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASK_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if attr == "position":
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

# Row 2: Cross-task overlap heatmaps
overlap_info = [
    ("position", "Position dims cross-task overlap", "Greens"),
    ("direction", "Direction dims cross-task overlap", "Reds"),
    ("object", "Object dims cross-task overlap", "Blues"),
]

for col, (attr, title, cmap) in enumerate(overlap_info):
    ax = fig.add_subplot(gs[1, col])
    M = build_overlap_matrix(attr)

    # Mask diagonal for better color scale (diag is always 50)
    mask = np.eye(len(TASKS), dtype=bool)
    off_vals = M[~mask]

    # Full matrix with values
    sns.heatmap(M, annot=True, fmt=".0f", cmap=cmap, vmin=0, vmax=30,
                xticklabels=TASK_LABELS, yticklabels=TASK_LABELS,
                linewidths=0.5, linecolor="white", cbar_kws={"label": f"overlap / {TOP_K}"},
                annot_kws={"size": 12, "weight": "bold"}, ax=ax)
    ax.set_xlabel("Task B", fontweight="bold", fontsize=10)
    ax.set_ylabel("Task A", fontweight="bold", fontsize=10)

    off_mean = off_vals.mean()
    ax.set_title(f"{title}\nOff-diag mean = {off_mean:.1f} (random ≈ {RANDOM_EXPECT:.1f})",
                 fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

# Super title
fig.suptitle(
    "Position / Direction / Object Dim Analysis — Vision Encoder output (SigLIP, mean-pooled)\n"
    "Top-50 dims per task × attribute → cross-task overlap\n"
    f"R2R_4way_1500 (6000 samples/task, 3 models → Vision Encoder frozen same across)",
    fontsize=13, fontweight="bold", y=0.995
)

for ext in [".png", ".pdf"]:
    plt.savefig(f"analysis/position_direction_object_dims{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] analysis/position_direction_object_dims.png / .pdf")


# ============================================================
#  Second figure: interpretation summary
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(13, 7))

off_pos = build_overlap_matrix("position")
off_dir = build_overlap_matrix("direction")
off_obj = build_overlap_matrix("object")

# Off-diag pairs sorted
pairs = []
for i, t1 in enumerate(TASKS):
    for j, t2 in enumerate(TASKS):
        if i >= j: continue
        pairs.append((f"{TASK_LABELS[i]}\n↔\n{TASK_LABELS[j]}",
                      off_pos[i, j], off_dir[i, j], off_obj[i, j]))

labels = [p[0] for p in pairs]
pos_vals = [p[1] for p in pairs]
dir_vals = [p[2] for p in pairs]
obj_vals = [p[3] for p in pairs]

x = np.arange(len(labels))
width = 0.25

ax.bar(x - width, pos_vals, width, label="Position", color="#2ecc71", edgecolor="black")
ax.bar(x, dir_vals, width, label="Direction", color="#e74c3c", edgecolor="black")
ax.bar(x + width, obj_vals, width, label="Object", color="#3498db", edgecolor="black")
ax.axhline(RANDOM_EXPECT, color="black", linestyle="--", linewidth=2, label=f"Random ≈ {RANDOM_EXPECT:.1f}")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Top-50 dim overlap", fontweight="bold", fontsize=11)
ax.set_title(
    "Cross-Task Top-50 Dim Overlap by Attribute (Vision Encoder)\n"
    "→ Position overlap ≈ random (task-specific dim usage). "
    "Object overlap high when identity label space shared (shape↔shape, obj↔obj).",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", alpha=0.3)

for ext in [".png", ".pdf"]:
    plt.savefig(f"analysis/position_direction_object_summary{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] analysis/position_direction_object_summary.png / .pdf")
