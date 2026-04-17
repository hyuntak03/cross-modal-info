"""
Mechanism Analysis: Vision Token vs Answer Token Probe Gap 원인 분석.

Phase 1: Stored feature 기반 (모델 로딩 불필요)
  Exp A: Per-layer vision mean-pool probe trajectory (29 layers)
  Exp B: Per-layer answer token probe trajectory (29 layers, 비교용)
  Exp C: Dimension-wise Direction Selectivity (Fisher criterion)
  Exp D: Per-layer temporal delta probe trajectory

모든 연산 GPU, CPU 최소화.

Usage:
    CUDA_VISIBLE_DEVICES=2 python analysis/mechanism_analysis.py \
        --model llava-video-7b \
        --output_dir analysis/mechanism_results
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_LABELS = ["Shape-Color", "Obj-Color", "Shape-Place", "Obj-Place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
}

DIRECTION_LABELS = ["Down", "Left", "Right", "Up"]


# ============================================================
#  GPU Probe
# ============================================================

def train_probe_gpu(X, y, num_classes=4, seed=42, epochs=50, lr=1e-3,
                    weight_decay=1e-2, batch_size=64, test_ratio=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape

    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    del X

    mean = X_t.mean(dim=0); X_t -= mean
    std = X_t.std(dim=0); std[std < 1e-8] = 1.0; X_t /= std
    del mean, std

    n_test = max(1, int(N * test_ratio))
    n_train = N - n_test
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    X_train, y_train = X_t[perm[:n_train]], y_t[perm[:n_train]]
    X_test, y_test = X_t[perm[n_train:]], y_t[perm[n_train:]]
    del X_t, y_t

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            b = idx[i:i+batch_size]
            loss = criterion(model(X_train[b]), y_train[b])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item() * 100

    del model, optimizer, X_train, X_test, y_train, y_test
    torch.cuda.empty_cache()
    return test_acc


# ============================================================
#  Exp A: Per-layer vision mean-pool probe trajectory
# ============================================================

def exp_a_vision_meanpool_trajectory(feat_root, task, num_layers=29):
    """Vision token을 mean pool(3584 dim)로 29 layers probe."""
    task_full = TASK_FULL(task)
    vt_dir = os.path.join(feat_root, "vision_token", task_full)
    meta = np.load(os.path.join(vt_dir, "meta.npy"), allow_pickle=True).item()
    labels = np.load(os.path.join(vt_dir, "labels.npy"))

    nf = meta.get("num_frames", 8)
    tpf = meta.get("tokens_per_frame_post", meta.get("tokens_per_frame", 196))
    hd = meta.get("hidden_dim", 3584)
    nl = meta.get("num_layers", num_layers)

    results = []
    for l in range(nl):
        feat = np.load(os.path.join(vt_dir, f"features_layer_{l}.npy"), mmap_mode='r')
        feat = np.array(feat).reshape(-1, nf, tpf, hd)
        feat_mean = feat.mean(axis=(1, 2)).astype(np.float32)  # (N, hd)
        acc = train_probe_gpu(feat_mean, labels)
        results.append({"layer": l, "test_acc": acc})
        print(f"  Layer {l:2d}: vision_meanpool = {acc:.1f}%")
        del feat, feat_mean
    return results


# ============================================================
#  Exp B: Per-layer answer token probe trajectory
# ============================================================

def exp_b_answer_trajectory(feat_root, task, num_layers=29):
    """Answer token 29 layers probe."""
    task_full = TASK_FULL(task)
    at_dir = os.path.join(feat_root, "answer_token", task_full)
    meta = np.load(os.path.join(at_dir, "meta.npy"), allow_pickle=True).item()
    labels = np.load(os.path.join(at_dir, "labels.npy"))
    nl = meta.get("num_layers", num_layers)

    results = []
    for l in range(nl):
        feat = np.array(np.load(os.path.join(at_dir, f"features_layer_{l}.npy"), mmap_mode='r'))
        acc = train_probe_gpu(feat.astype(np.float32), labels)
        results.append({"layer": l, "test_acc": acc})
        print(f"  Layer {l:2d}: answer_token = {acc:.1f}%")
        del feat
    return results


# ============================================================
#  Exp C: Dimension-wise Fisher criterion
# ============================================================

def exp_c_fisher_criterion(feat_root, task, layer_idx=-1):
    """Vision mean-pool vs Answer token: dimension별 Fisher score 비교."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_full = TASK_FULL(task)

    # Vision mean-pool
    vt_dir = os.path.join(feat_root, "vision_token", task_full)
    meta = np.load(os.path.join(vt_dir, "meta.npy"), allow_pickle=True).item()
    labels = np.load(os.path.join(vt_dir, "labels.npy"))
    nf, tpf, hd = meta.get("num_frames", 8), meta.get("tokens_per_frame_post", 196), meta.get("hidden_dim", 3584)
    nl = meta.get("num_layers", 29)
    if layer_idx < 0: layer_idx = nl + layer_idx

    feat_v = np.array(np.load(os.path.join(vt_dir, f"features_layer_{layer_idx}.npy"), mmap_mode='r'))
    feat_v = feat_v.reshape(-1, nf, tpf, hd).mean(axis=(1, 2))  # (N, hd)

    # Answer token
    at_dir = os.path.join(feat_root, "answer_token", task_full)
    feat_a = np.array(np.load(os.path.join(at_dir, f"features_layer_{layer_idx}.npy"), mmap_mode='r'))

    def compute_fisher_gpu(X_np, y_np, num_classes=4):
        """GPU에서 Fisher criterion 계산: F(d) = var_between(d) / var_within(d)"""
        X = torch.from_numpy(X_np.astype(np.float32)).to(device)
        y = torch.from_numpy(y_np).long().to(device)
        global_mean = X.mean(dim=0)

        var_between = torch.zeros(X.shape[1], device=device)
        var_within = torch.zeros(X.shape[1], device=device)

        for c in range(num_classes):
            mask = (y == c)
            if mask.sum() == 0: continue
            Xc = X[mask]
            class_mean = Xc.mean(dim=0)
            var_between += mask.sum().float() * (class_mean - global_mean) ** 2
            var_within += (Xc - class_mean).pow(2).sum(dim=0)

        var_within = var_within / max(len(y) - num_classes, 1)
        var_between = var_between / max(num_classes - 1, 1)
        fisher = var_between / (var_within + 1e-8)
        return fisher.cpu().numpy()

    f_vision = compute_fisher_gpu(feat_v, labels)
    f_answer = compute_fisher_gpu(feat_a, labels)

    # Summary stats
    results = {
        "vision_meanpool_fisher_mean": float(f_vision.mean()),
        "vision_meanpool_fisher_max": float(f_vision.max()),
        "vision_meanpool_fisher_top50_mean": float(np.sort(f_vision)[-50:].mean()),
        "answer_token_fisher_mean": float(f_answer.mean()),
        "answer_token_fisher_max": float(f_answer.max()),
        "answer_token_fisher_top50_mean": float(np.sort(f_answer)[-50:].mean()),
        "fisher_ratio_mean": float(f_answer.mean() / max(f_vision.mean(), 1e-8)),
        "fisher_ratio_top50": float(np.sort(f_answer)[-50:].mean() / max(np.sort(f_vision)[-50:].mean(), 1e-8)),
    }

    print(f"  Vision mean-pool Fisher: mean={results['vision_meanpool_fisher_mean']:.4f}, "
          f"max={results['vision_meanpool_fisher_max']:.4f}, top50={results['vision_meanpool_fisher_top50_mean']:.4f}")
    print(f"  Answer token Fisher:     mean={results['answer_token_fisher_mean']:.4f}, "
          f"max={results['answer_token_fisher_max']:.4f}, top50={results['answer_token_fisher_top50_mean']:.4f}")
    print(f"  Ratio (answer/vision):   mean={results['fisher_ratio_mean']:.2f}x, top50={results['fisher_ratio_top50']:.2f}x")

    del feat_v, feat_a
    return results, f_vision, f_answer


# ============================================================
#  Exp D: Per-layer temporal delta probe trajectory
# ============================================================

def exp_d_temporal_delta_trajectory(feat_root, task, num_layers=29):
    """Temporal delta (frame[t+1] - frame[t]) mean pool로 29 layers probe."""
    task_full = TASK_FULL(task)
    vt_dir = os.path.join(feat_root, "vision_token", task_full)
    meta = np.load(os.path.join(vt_dir, "meta.npy"), allow_pickle=True).item()
    labels = np.load(os.path.join(vt_dir, "labels.npy"))

    nf = meta.get("num_frames", 8)
    tpf = meta.get("tokens_per_frame_post", 196)
    hd = meta.get("hidden_dim", 3584)
    nl = meta.get("num_layers", num_layers)

    results = []
    for l in range(nl):
        feat = np.load(os.path.join(vt_dir, f"features_layer_{l}.npy"), mmap_mode='r')
        feat = np.array(feat).reshape(-1, nf, tpf, hd)
        frame_means = feat.mean(axis=2)  # (N, nf, hd)
        deltas = frame_means[:, 1:, :] - frame_means[:, :-1, :]  # (N, nf-1, hd)
        delta_mean = deltas.mean(axis=1).astype(np.float32)  # (N, hd)
        acc = train_probe_gpu(delta_mean, labels)
        results.append({"layer": l, "test_acc": acc})
        print(f"  Layer {l:2d}: temporal_delta = {acc:.1f}%")
        del feat, frame_means, deltas, delta_mean
    return results


# ============================================================
#  Plotting
# ============================================================

def plot_trajectories(all_results, model_name, output_dir):
    """4 tasks × trajectory comparison plot."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    for ax, task, task_label in zip(axes, TASKS, TASK_LABELS):
        if task not in all_results:
            continue
        r = all_results[task]

        if "exp_a" in r:
            layers_a = [x["layer"] for x in r["exp_a"]]
            acc_a = [x["test_acc"] for x in r["exp_a"]]
            ax.plot(layers_a, acc_a, color="#2ecc71", linewidth=2, marker='o', markersize=3,
                    label="Vision Mean-Pool (3584d)")

        if "exp_d" in r:
            layers_d = [x["layer"] for x in r["exp_d"]]
            acc_d = [x["test_acc"] for x in r["exp_d"]]
            ax.plot(layers_d, acc_d, color="#e67e22", linewidth=2, marker='s', markersize=3,
                    label="Temporal Delta (3584d)")

        if "exp_b" in r:
            layers_b = [x["layer"] for x in r["exp_b"]]
            acc_b = [x["test_acc"] for x in r["exp_b"]]
            ax.plot(layers_b, acc_b, color="#e74c3c", linewidth=2, marker='^', markersize=3,
                    label="Answer Token (3584d)")

        ax.axhline(y=25, color="gray", linestyle=":", alpha=0.5, label="Chance (25%)")
        ax.set_title(task_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("LLM Layer", fontsize=11)
        ax.set_ylim(0, 100)
        if ax == axes[0]:
            ax.set_ylabel("Test Accuracy (%)", fontsize=11)

    axes[-1].legend(fontsize=8, loc="lower right")
    short = model_name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")
    fig.suptitle(f"Per-Layer Probe Trajectory: {short}\n"
                 f"(Vision Mean-Pool vs Temporal Delta vs Answer Token)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"trajectory_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-video-7b")
    parser.add_argument("--output_dir", type=str, default="analysis/mechanism_results")
    args = parser.parse_args()

    feat_root = FEAT_ROOTS.get(args.model)
    if feat_root is None or not os.path.exists(feat_root):
        print(f"[ERROR] Features not found: {feat_root}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for task in TASKS:
        print(f"\n{'#'*60}")
        print(f"  {args.model} / {task}")
        print(f"{'#'*60}")

        task_results = {}

        # Exp A: Vision mean-pool trajectory
        print("\n[Exp A] Vision Mean-Pool Trajectory")
        task_results["exp_a"] = exp_a_vision_meanpool_trajectory(feat_root, task)

        # Exp B: Answer token trajectory
        print("\n[Exp B] Answer Token Trajectory")
        task_results["exp_b"] = exp_b_answer_trajectory(feat_root, task)

        # Exp C: Fisher criterion (last layer)
        print("\n[Exp C] Fisher Criterion (Last Layer)")
        fisher_results, _, _ = exp_c_fisher_criterion(feat_root, task, layer_idx=-1)
        task_results["exp_c"] = fisher_results

        # Exp D: Temporal delta trajectory
        print("\n[Exp D] Temporal Delta Trajectory")
        task_results["exp_d"] = exp_d_temporal_delta_trajectory(feat_root, task)

        all_results[task] = task_results

    # Save JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    json_path = os.path.join(args.output_dir, f"mechanism_{args.model}.json")
    with open(json_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # Plot
    plot_trajectories(all_results, args.model, args.output_dir)


if __name__ == "__main__":
    main()
