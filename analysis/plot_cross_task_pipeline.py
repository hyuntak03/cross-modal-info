"""
Cross-task probing pipeline 시각화.
- Figure 1: Vision pipeline (VE, AP, vision_token per layer) — 3 models subplot
- Figure 2: Answer token per layer — 3 models subplot

Each: diagonal (in-task) vs off-diagonal (cross-task) curves + transfer gap
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = {
    "Vanilla": "analysis/cross_task_probing_pipeline_1500_vanilla.json",
    "Baseline": "analysis/cross_task_probing_pipeline_1500_baseline.json",
    "Delta": "analysis/cross_task_probing_pipeline_1500_delta.json",
}

COLORS = {"Vanilla": "#95a5a6", "Baseline": "#3498db", "Delta": "#e74c3c"}


def load_all():
    data = {}
    for m, fp in FILES.items():
        with open(fp) as f:
            d = json.load(f)
        data[m] = d[m]
    return data


def plot_vision_pipeline(data):
    """VE + AP (horizontal lines) + vision_token layers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, model in zip(axes, ["Vanilla", "Baseline", "Delta"]):
        d = data[model]
        color = COLORS[model]

        # Vision token layer curves
        vt_layers = sorted(int(l) for l in d["vision_token"].keys())
        diag = [d["vision_token"][str(l)]["diag"] for l in vt_layers]
        offdiag = [d["vision_token"][str(l)]["offdiag"] for l in vt_layers]

        ax.plot(vt_layers, diag, marker="o", markersize=4, linewidth=2, color=color,
                label="Vision Token — in-task (diag)")
        ax.plot(vt_layers, offdiag, marker="s", markersize=4, linewidth=2, color=color,
                linestyle="--", alpha=0.7,
                label="Vision Token — cross-task (off-diag)")

        # Vision encoder + after projector as horizontal lines
        if "vision_encoder" in d:
            ve_d = d["vision_encoder"]["diag"]
            ve_o = d["vision_encoder"]["offdiag"]
            ax.axhline(ve_d, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
                       label=f"VE diag={ve_d:.0f}%")
            ax.axhline(ve_o, color="green", linestyle=":", linewidth=1.0, alpha=0.4)
        if "after_projector" in d:
            ap_d = d["after_projector"]["diag"]
            ap_o = d["after_projector"]["offdiag"]
            ax.axhline(ap_d, color="purple", linestyle=":", linewidth=1.5, alpha=0.7,
                       label=f"AP diag={ap_d:.0f}%")
            ax.axhline(ap_o, color="purple", linestyle=":", linewidth=1.0, alpha=0.4)

        ax.axhline(25, color="black", linestyle=":", linewidth=0.8, alpha=0.3)  # chance
        ax.set_xlabel("LLM Layer", fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Direction Probe Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")
        ax.set_ylim(20, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Cross-Task Direction Probing — Vision Pipeline\n"
        "In-task (solid) vs Cross-task (dashed). Gap small = identity-invariant direction",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(f"analysis/cross_task_pipeline_vision{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] analysis/cross_task_pipeline_vision.png")


def plot_answer_token(data):
    """Answer token per layer, 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, model in zip(axes, ["Vanilla", "Baseline", "Delta"]):
        d = data[model]
        color = COLORS[model]

        at_layers = sorted(int(l) for l in d["answer_token"].keys())
        diag = [d["answer_token"][str(l)]["diag"] for l in at_layers]
        offdiag = [d["answer_token"][str(l)]["offdiag"] for l in at_layers]
        gap = [diag[i] - offdiag[i] for i in range(len(diag))]

        ax.plot(at_layers, diag, marker="o", markersize=4, linewidth=2, color=color,
                label="In-task (diag)")
        ax.plot(at_layers, offdiag, marker="s", markersize=4, linewidth=2, color=color,
                linestyle="--", alpha=0.7, label="Cross-task (off-diag)")

        # Transfer gap on secondary axis
        ax2 = ax.twinx()
        ax2.fill_between(at_layers, 0, gap, color=color, alpha=0.15, label="Transfer gap")
        ax2.set_ylim(0, 50)
        ax2.set_ylabel("Transfer Gap (%p)", fontsize=9, color="gray")
        ax2.tick_params(axis="y", colors="gray", labelsize=8)

        ax.axhline(25, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
        ax.set_xlabel("LLM Layer", fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Probe Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")
        ax.set_ylim(20, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Cross-Task Direction Probing — Answer Token (Last Token per Layer)\n"
        "Shaded area = transfer gap (smaller = more identity-invariant)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(f"analysis/cross_task_pipeline_answer{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] analysis/cross_task_pipeline_answer.png")


def plot_combined_models(data):
    """3모델 한 plot에 overlay. 2 panel: vision_token, answer_token."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for row, kind in enumerate(["vision_token", "answer_token"]):
        for col, metric in enumerate(["diag", "offdiag"]):
            ax = axes[row, col]
            for model in ["Vanilla", "Baseline", "Delta"]:
                d = data[model][kind]
                layers = sorted(int(l) for l in d.keys())
                vals = [d[str(l)][metric] for l in layers]
                ax.plot(layers, vals, marker="o", markersize=3, linewidth=2,
                        color=COLORS[model], label=model)
            ax.axhline(25, color="black", linestyle=":", alpha=0.3)
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"Probe Acc (%)")
            title_metric = "in-task (diagonal)" if metric == "diag" else "cross-task (off-diagonal)"
            ax.set_title(f"{kind.replace('_', ' ').title()} — {title_metric}",
                         fontsize=11, fontweight="bold")
            ax.set_ylim(20, 105)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

    fig.suptitle("Cross-Task Probing by Layer — Model Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(f"analysis/cross_task_pipeline_comparison{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] analysis/cross_task_pipeline_comparison.png")


if __name__ == "__main__":
    data = load_all()
    plot_vision_pipeline(data)
    plot_answer_token(data)
    plot_combined_models(data)
