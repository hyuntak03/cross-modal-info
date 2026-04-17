"""
Within-task position/direction/object feature dim 공유 분석 시각화.

Question: Task 내에서 position과 direction이 같은 dim을 쓰는가?
  - P ∩ D 높으면 → 같은 dim 공유 (direction ≈ position delta)
  - P ∩ O 낮으면 → identity와 분리
  - D ∩ O 낮으면 → direction과 identity 분리
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis/position_direction_object_dims.json") as f:
    data = json.load(f)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_LABELS = ["Shape-Color", "Obj-Color", "Shape-Place", "Obj-Place"]
TOP_K = 50

RANDOM_VE = TOP_K * TOP_K / 1152  # 2.17
RANDOM_AP = TOP_K * TOP_K / 3584  # 0.70


def overlap(a, b):
    return len(set(a) & set(b))


def compute_within_task_overlaps(stage_res):
    """For each task, compute P∩D, P∩O, D∩O."""
    results = []
    for task in TASKS:
        r = stage_res[task]
        P = r["position"]["top_dims"]
        D = r["direction"]["top_dims"]
        O = r["object"]["top_dims"]
        results.append({
            "task": task,
            "P∩D": overlap(P, D),
            "P∩O": overlap(P, O),
            "D∩O": overlap(D, O),
        })
    return results


# ============================================================
#  Figure 1: Vision Encoder within-task overlaps (3 models same)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Vanilla VE
ax = axes[0]
ve_res = compute_within_task_overlaps(data["Vanilla"]["vision_encoder"])

x = np.arange(len(TASKS))
width = 0.25

vals_pd = [r["P∩D"] for r in ve_res]
vals_po = [r["P∩O"] for r in ve_res]
vals_do = [r["D∩O"] for r in ve_res]

ax.bar(x - width, vals_pd, width, label="P ∩ D (Position vs Direction)",
       color="#e67e22", edgecolor="black", linewidth=1.2)
ax.bar(x, vals_po, width, label="P ∩ O (Position vs Object)",
       color="#9b59b6", edgecolor="black", linewidth=1.2)
ax.bar(x + width, vals_do, width, label="D ∩ O (Direction vs Object)",
       color="#16a085", edgecolor="black", linewidth=1.2)

# Value labels
for i, (pd, po, do) in enumerate(zip(vals_pd, vals_po, vals_do)):
    for xi, v in zip([i - width, i, i + width], [pd, po, do]):
        ax.text(xi, v + 0.3, f"{v}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

ax.axhline(RANDOM_VE, color="red", linestyle="--", linewidth=2,
           label=f"Random ≈ {RANDOM_VE:.1f}")

ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, fontsize=10, fontweight="bold")
ax.set_ylabel(f"Top-{TOP_K} dim overlap", fontsize=11, fontweight="bold")
ax.set_title("Vision Encoder (SigLIP, D=1152)\nWithin-task dim sharing",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 14)
ax.grid(axis="y", alpha=0.3)

# Right: Vanilla AP
ax = axes[1]
ap_res = compute_within_task_overlaps(data["Vanilla"]["after_projector"])

vals_pd = [r["P∩D"] for r in ap_res]
vals_po = [r["P∩O"] for r in ap_res]
vals_do = [r["D∩O"] for r in ap_res]

ax.bar(x - width, vals_pd, width, label="P ∩ D",
       color="#e67e22", edgecolor="black", linewidth=1.2)
ax.bar(x, vals_po, width, label="P ∩ O",
       color="#9b59b6", edgecolor="black", linewidth=1.2)
ax.bar(x + width, vals_do, width, label="D ∩ O",
       color="#16a085", edgecolor="black", linewidth=1.2)

for i, (pd, po, do) in enumerate(zip(vals_pd, vals_po, vals_do)):
    for xi, v in zip([i - width, i, i + width], [pd, po, do]):
        ax.text(xi, v + 0.1, f"{v}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

ax.axhline(RANDOM_AP, color="red", linestyle="--", linewidth=2,
           label=f"Random ≈ {RANDOM_AP:.1f}")

ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, fontsize=10, fontweight="bold")
ax.set_ylabel(f"Top-{TOP_K} dim overlap", fontsize=11, fontweight="bold")
ax.set_title("After Projector (D=3584)\nWithin-task dim sharing",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 14)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Within-Task Feature Dim Sharing: Position / Direction / Object\n"
    "Higher P∩D = direction and position share dims (direction ≈ position delta)\n"
    "Lower P∩O / D∩O = identity uses separate dims",
    fontsize=13, fontweight="bold", y=1.03
)
plt.tight_layout()
for ext in [".png", ".pdf"]:
    plt.savefig(f"analysis/within_task_dim_sharing{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] analysis/within_task_dim_sharing.png / .pdf")


# ============================================================
#  Figure 2: Top dim explicit listing (for paper readability)
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(16, 11))

for row, task in enumerate(TASKS):
    ve = data["Vanilla"]["vision_encoder"][task]
    P_dims = set(ve["position"]["top_dims"])
    D_dims = set(ve["direction"]["top_dims"])
    O_dims = set(ve["object"]["top_dims"])

    for col, (attr_a, attr_b, dims_a, dims_b, color) in enumerate([
        ("Position", "Direction", P_dims, D_dims, "#e67e22"),
        ("Position", "Object", P_dims, O_dims, "#9b59b6"),
        ("Direction", "Object", D_dims, O_dims, "#16a085"),
    ]):
        ax = axes[row, col]
        shared = dims_a & dims_b
        only_a = dims_a - dims_b
        only_b = dims_b - dims_a

        counts = [len(only_a), len(shared), len(only_b)]
        labels = [f"{attr_a}\nonly", "shared", f"{attr_b}\nonly"]
        colors = ["lightgray", color, "lightgray"]
        ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.2)

        for i, v in enumerate(counts):
            ax.text(i, v + 1, f"{v}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_ylim(0, 60)
        ax.set_yticks([0, 25, 50])
        ax.grid(axis="y", alpha=0.3)

        if row == 0:
            ax.set_title(f"{attr_a} ∩ {attr_b}", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{TASK_LABELS[row]}", fontsize=11, fontweight="bold")

fig.suptitle(
    f"Per-task Top-{TOP_K} Dim Composition (Vision Encoder)\n"
    f"Random expected overlap: {RANDOM_VE:.1f}",
    fontsize=13, fontweight="bold", y=0.99
)
plt.tight_layout()
for ext in [".png", ".pdf"]:
    plt.savefig(f"analysis/within_task_dim_composition{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] analysis/within_task_dim_composition.png / .pdf")
