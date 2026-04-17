"""Plot cross-task probing 4x4 matrix for 3 models."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

with open("analysis/cross_task_probing_1500.json") as f:
    data = json.load(f)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_LABELS = ["Shape-Color\n(in-domain)", "Obj-Color", "Shape-Place", "Obj-Place\n(hardest OOD)"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
models = ["Vanilla", "Baseline", "Delta"]

for ax, model in zip(axes, models):
    if model not in data:
        continue
    matrix = np.array(data[model]["matrix"])
    diag = data[model]["diagonal_mean"]
    off_diag = data[model]["off_diagonal_mean"]
    gap = data[model]["transfer_gap"]

    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="RdYlGn", vmin=25, vmax=100,
        xticklabels=TASK_LABELS, yticklabels=TASK_LABELS,
        cbar_kws={"label": "Test Accuracy (%)"}, ax=ax,
        linewidths=0.5, linecolor='white', annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_xlabel("Test Task", fontsize=11, fontweight='bold')
    if ax == axes[0]:
        ax.set_ylabel("Train Task", fontsize=11, fontweight='bold')

    ax.set_title(
        f"{model}\nDiag={diag:.1f}% | Off-diag={off_diag:.1f}% | Gap={gap:.1f}%p",
        fontsize=12, fontweight='bold', pad=10
    )

fig.suptitle(
    "Cross-Task Direction Probing (Last Layer Answer Token, 1500-sample)\n"
    "Higher off-diagonal = identity-invariant direction representation",
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

for ext in [".png", ".pdf"]:
    sp = f"analysis/cross_task_probing_1500{ext}"
    plt.savefig(sp, dpi=150, bbox_inches="tight")
plt.close()
print(f"[SAVED] analysis/cross_task_probing_1500.png / .pdf")
