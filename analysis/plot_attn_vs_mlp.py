"""
Attn vs MLP 결과 시각화.

3모델 × 2task, direction/identity probe: after_attn vs after_mlp layer-wise plot.

Usage:
    python analysis/plot_attn_vs_mlp.py
"""

import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "analysis/attn_vs_mlp_results"
OUTPUT_DIR = "analysis/attn_vs_mlp_results"

MODELS = ["vanilla", "4combo_v2_baseline", "4combo_v2_delta"]
MODEL_LABELS = ["Vanilla", "Baseline", "Delta"]
TASKS = ["shape_color", "obj_place"]
TASK_LABELS = ["Shape-Color (In-domain)", "Obj-Place (Hardest OOD)"]


def load_result(model, task):
    fp = os.path.join(RESULT_DIR, f"attn_vs_mlp_{model}_{task}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def plot_direction_identity():
    """2×3 grid: row=task, col=model. Each panel: direction(attn/mlp) + identity(attn/mlp)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)

    for ti, (task, tl) in enumerate(zip(TASKS, TASK_LABELS)):
        for mi, (model, ml) in enumerate(zip(MODELS, MODEL_LABELS)):
            ax = axes[ti, mi]
            d = load_result(model, task)
            if d is None:
                ax.set_title(f"{ml} / {tl}\n(no data)", fontsize=10)
                continue

            layers = d["layers"]

            # Direction: attn (solid) vs mlp (dashed)
            ax.plot(layers, d["after_attn_dir"], color="#e74c3c", linewidth=2,
                    marker="o", markersize=3, label="Attn → dir")
            ax.plot(layers, d["after_mlp_dir"], color="#e74c3c", linewidth=2,
                    linestyle="--", marker="s", markersize=3, alpha=0.6, label="MLP → dir")

            # Identity: attn (solid) vs mlp (dashed)
            ax.plot(layers, d["after_attn_id"], color="#3498db", linewidth=2,
                    marker="o", markersize=3, label="Attn → id")
            ax.plot(layers, d["after_mlp_id"], color="#3498db", linewidth=2,
                    linestyle="--", marker="s", markersize=3, alpha=0.6, label="MLP → id")

            ax.axhline(y=25, color="gray", linestyle=":", alpha=0.3)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Layer")
            if mi == 0:
                ax.set_ylabel("Probe Acc (%)")

            ax.set_title(f"{ml} — {tl}", fontsize=10, fontweight="bold")
            if ti == 0 and mi == 2:
                ax.legend(fontsize=7, loc="center right")

    fig.suptitle("Attention vs MLP Contribution: Direction & Identity Probing (Last Token)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    for ext in [".png", ".pdf"]:
        sp = os.path.join(OUTPUT_DIR, f"attn_vs_mlp_grid{ext}")
        plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {os.path.join(OUTPUT_DIR, 'attn_vs_mlp_grid.png')}")


def plot_delta_only():
    """Delta(MLP-Attn) for direction across models, side by side for 2 tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"vanilla": "#95a5a6", "4combo_v2_baseline": "#3498db", "4combo_v2_delta": "#e74c3c"}

    for ti, (task, tl) in enumerate(zip(TASKS, TASK_LABELS)):
        ax = axes[ti]
        for model, ml in zip(MODELS, MODEL_LABELS):
            d = load_result(model, task)
            if d is None:
                continue
            layers = d["layers"]
            delta = [m - a for a, m in zip(d["after_attn_dir"], d["after_mlp_dir"])]
            ax.plot(layers, delta, color=colors[model], linewidth=2,
                    marker="o", markersize=3, label=ml)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ(MLP - Attn) direction acc (%p)")
        ax.set_title(tl, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(-15, 15)

    fig.suptitle("MLP Contribution to Direction (Δ = after_mlp - after_attn)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    for ext in [".png", ".pdf"]:
        sp = os.path.join(OUTPUT_DIR, f"attn_vs_mlp_delta{ext}")
        plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {os.path.join(OUTPUT_DIR, 'attn_vs_mlp_delta.png')}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_direction_identity()
    plot_delta_only()
