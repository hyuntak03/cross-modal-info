"""
Vision Encoder + After Projector 4x4 cross-task matrix 6개 시각화 (2 rows × 3 models).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FILES = {
    "Vanilla": "analysis/cross_task_probing_pipeline_1500_vanilla.json",
    "Baseline": "analysis/cross_task_probing_pipeline_1500_baseline.json",
    "Delta": "analysis/cross_task_probing_pipeline_1500_delta.json",
}

TASK_LABELS = ["Shape-Color", "Obj-Color", "Shape-Place", "Obj-Place"]
STAGES = [("vision_encoder", "Vision Encoder"), ("after_projector", "After Projector")]
MODELS = ["Vanilla", "Baseline", "Delta"]


def load_all():
    data = {}
    for m, fp in FILES.items():
        with open(fp) as f:
            d = json.load(f)
        data[m] = d[m]
    return data


def plot_matrices(data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    for row, (stage_key, stage_label) in enumerate(STAGES):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            d = data[model][stage_key]
            matrix = np.array(d["matrix"])
            diag = d["diag"]
            offdiag = d["offdiag"]
            gap = diag - offdiag

            sns.heatmap(
                matrix, annot=True, fmt=".1f", cmap="RdYlGn", vmin=25, vmax=100,
                xticklabels=TASK_LABELS, yticklabels=TASK_LABELS,
                cbar_kws={"label": "Test Acc (%)"}, ax=ax,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 11, "weight": "bold"},
            )
            ax.set_xlabel("Test Task", fontsize=10, fontweight="bold")
            ax.set_ylabel("Train Task", fontsize=10, fontweight="bold")
            ax.set_title(
                f"{stage_label} / {model}\n"
                f"Diag={diag:.1f}% | Off={offdiag:.1f}% | Gap={gap:.1f}%p",
                fontsize=11, fontweight="bold", pad=8
            )
            ax.tick_params(axis="x", rotation=30, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(
        "Cross-Task Direction Probing — Vision Encoder & After Projector\n"
        "Higher off-diagonal = identity-invariant direction representation",
        fontsize=14, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(f"analysis/cross_task_ve_ap_matrices{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] analysis/cross_task_ve_ap_matrices.png / .pdf")


if __name__ == "__main__":
    data = load_all()
    plot_matrices(data)
