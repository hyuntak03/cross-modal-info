"""
기존 output_4way_1500/{model}/answer_probe_results/{task}/linear_probe_accuracy.{pdf,png}
plot을 letter probe 선 추가된 버전으로 덮어쓰기.

기존: Train/Test direction acc per layer
신규: + Test letter acc per layer (binding gap 확인 가능)
"""

import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

with open("analysis/letter_vs_direction_probing.json") as f:
    letter_data = json.load(f)

MODEL_DIRS = {
    "Vanilla": "llava-video-7b",
    "Baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "llava-video-7b_lora_4combo_v2_delta",
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]


def update_plot(model_name, task, letter_results):
    """Load existing CSV + letter probe → overlay plot."""
    model_dir = MODEL_DIRS.get(model_name)
    if model_dir is None:
        return

    probe_dir = os.path.join(
        "output_4way_1500", model_dir, "answer_probe_results",
        f"vlm_direction_testbed_R2R_4way_1500_{task}"
    )
    csv_path = os.path.join(probe_dir, "linear_probe_results.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return

    # Load direction probe (train/test) from CSV
    layers = []
    train_accs = []
    test_accs = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            layers.append(int(float(row["layer"])))
            train_accs.append(float(row["train_acc"]))
            test_accs.append(float(row["test_acc"]))

    # Align letter probe with layer numbers
    layer_keys = sorted(letter_results.keys(), key=lambda k: int(k[1:]))
    letter_layer_map = {int(k[1:]): letter_results[k]["letter_acc"] for k in layer_keys}
    letter_accs = [letter_layer_map.get(l, None) for l in layers]

    # Plot
    sns.set(context="notebook")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(layers, test_accs, label="Direction — Test", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax.plot(layers, train_accs, label="Direction — Train", color="#5c95ff", linewidth=1.6, linestyle="--", alpha=0.6)

    valid_mask = [v is not None for v in letter_accs]
    valid_layers = [l for l, v in zip(layers, letter_accs) if v is not None]
    valid_letters = [v for v in letter_accs if v is not None]

    if valid_letters:
        ax.plot(valid_layers, valid_letters, label="Letter (MCQ A/B/C/D) — Test",
                color="#d62728", linewidth=2.2, marker="s", markersize=4)
        # binding gap shade
        valid_dir = [test_accs[layers.index(l)] for l in valid_layers]
        ax.fill_between(valid_layers, valid_dir, valid_letters,
                         alpha=0.15, color="purple", label="Binding gap")

    ax.axhline(y=25.0, color="gray", linestyle=":", alpha=0.6, label="Chance (25%)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Answer Token Probing per Layer — {model_name} / {task}")
    ax.set_xlim(0, max(layers))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    save_pdf = os.path.join(probe_dir, "linear_probe_accuracy.pdf")
    plt.savefig(save_pdf)
    plt.savefig(save_pdf.replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"[UPDATED] {save_pdf}")


def main():
    for model_name, results in letter_data.items():
        for task, task_results in results.items():
            update_plot(model_name, task, task_results)


if __name__ == "__main__":
    main()
