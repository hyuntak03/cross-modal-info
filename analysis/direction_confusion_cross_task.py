"""
Cross-task transfer 시 direction class 간 혼동 패턴 측정.
각 모델 × 각 stage에서:
  Source task probe → Target task features → 4x4 confusion matrix (GT direction × Pred direction)
  모든 (source != target) task pair에 대해 aggregate하여 평균.

Direction labels order: ['Down', 'Left', 'Right', 'Up'] (alphabetical)
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FEAT_ROOTS = {
    "Vanilla": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b",
    "Baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta",
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"
DIR_LABELS = ["Down", "Left", "Right", "Up"]  # alphabetical as in LabelEncoder

STAGES = [
    ("vision_encoder", None, "VE"),
    ("after_projector", None, "AP"),
    ("vision_token", 14, "vt_L14"),
    ("vision_token", 27, "vt_L27"),
    ("answer_token", 14, "at_L14"),
    ("answer_token", 27, "at_L27"),
]


def load_stage_feat(feat_root, task, stage, layer=None):
    if stage in ("vision_encoder", "after_projector"):
        d = os.path.join(feat_root, stage, TASK_FULL(task))
    else:
        d = os.path.join(feat_root, stage, TASK_FULL(task))
    if layer is None:
        feat = np.load(os.path.join(d, "features.npy")).astype(np.float32)
    else:
        feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
    labels = np.load(os.path.join(d, "labels.npy"))
    return feat, labels


def train_probe(X, y, nc=4, seed=42, epochs=50):
    device = torch.device("cuda")
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).long().to(device)
    mean = X_t.mean(0); std = X_t.std(0); std[std < 1e-8] = 1.0
    Xn = (X_t - mean) / std

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(X.shape[1], nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xn), device=device)
        for i in range(0, len(Xn), 256):
            b = idx[i:i+256]
            loss = crit(model(Xn[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    W = model.weight.detach()
    b_ = model.bias.detach()
    return W, b_, mean, std


def predict(W, b, mean, std, X):
    device = torch.device("cuda")
    X_t = torch.from_numpy(X).to(device)
    Xn = (X_t - mean) / std
    with torch.no_grad():
        pred = (Xn @ W.T + b).argmax(1).cpu().numpy()
    return pred


def confusion_matrix_normalized(y_true, y_pred, n_classes=4):
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for gt, pr in zip(y_true, y_pred):
        cm[gt, pr] += 1
    # row-normalize (GT 기준 prediction 분포)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm_norm * 100  # percent


def analyze_stage(feat_root, stage, layer, seed=42, test_ratio=0.3):
    """Source task → Target task (≠source) 모든 12쌍에 대해 confusion matrix 평균."""
    data = {}
    for task in TASKS:
        try:
            feat, labels = load_stage_feat(feat_root, task, stage, layer)
        except FileNotFoundError:
            return None
        np.random.seed(seed)
        idx = np.random.permutation(len(feat))
        n_test = int(len(feat) * test_ratio)
        tr, te = idx[n_test:], idx[:n_test]
        data[task] = {"Xtr": feat[tr], "ytr": labels[tr],
                      "Xte": feat[te], "yte": labels[te]}

    # Train probes
    probes = {}
    for task in TASKS:
        probes[task] = train_probe(data[task]["Xtr"], data[task]["ytr"])

    # Cross-task confusion
    cross_cms = []
    for src in TASKS:
        W, b, mean, std = probes[src]
        for tgt in TASKS:
            if src == tgt:
                continue
            pred = predict(W, b, mean, std, data[tgt]["Xte"])
            cm = confusion_matrix_normalized(data[tgt]["yte"], pred, 4)
            cross_cms.append(cm)

    mean_cm = np.mean(cross_cms, axis=0)
    return mean_cm.tolist()


def main():
    results = {}
    for model_name, feat_root in FEAT_ROOTS.items():
        if not os.path.exists(feat_root):
            continue
        print(f"\n{model_name}")
        model_results = {}
        for stage, layer, label in STAGES:
            cm = analyze_stage(feat_root, stage, layer)
            if cm is None:
                continue
            model_results[label] = cm
            acc = np.mean(np.diag(cm))
            print(f"  {label}: mean diagonal (correct) = {acc:.1f}%")
        results[model_name] = model_results

    with open("analysis/direction_confusion_cross_task.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] analysis/direction_confusion_cross_task.json")

    # Plot: 3 models × 6 stages grid
    n_models = len([m for m in results if results[m]])
    n_stages = len(STAGES)
    fig, axes = plt.subplots(n_models, n_stages, figsize=(22, 11))

    for row, (model, model_data) in enumerate(results.items()):
        for col, (stage, layer, label) in enumerate(STAGES):
            ax = axes[row, col] if n_models > 1 else axes[col]
            if label not in model_data:
                ax.axis('off')
                continue
            cm = np.array(model_data[label])
            diag_mean = np.diag(cm).mean()

            sns.heatmap(cm, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                        xticklabels=DIR_LABELS, yticklabels=DIR_LABELS,
                        cbar=False, ax=ax, linewidths=0.3, linecolor="white",
                        annot_kws={"size": 9, "weight": "bold"})
            ax.set_title(f"{model} / {label}\nCorrect={diag_mean:.1f}%",
                         fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel("GT Direction", fontsize=9, fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel("Predicted", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)

    fig.suptitle(
        "Cross-task Direction Confusion — Avg over all (source≠target) task pairs\n"
        "Row-normalized (each row sums to 100%). Diagonal = correct transfer.",
        fontsize=13, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(f"analysis/direction_confusion_cross_task{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] analysis/direction_confusion_cross_task.png / .pdf")


if __name__ == "__main__":
    main()
