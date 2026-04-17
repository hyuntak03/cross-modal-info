"""
Dimension Selection Analysis: 일부 차원만으로 linear probing 성능 향상 가능한가?

Vision encoder (1152d) / After projector (3584d)에서 mean-pool 후:
  (a) Top-k Fisher dims: direction-discriminative 차원만 선택
  (b) Top-k PCA components: 최대 분산 축으로 projection
  (c) Random k dims: 통제 조건
  k = [10, 25, 50, 100, 200, 500, 1000, full]

모든 연산 GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/dimension_selection.py \
        --model llava-video-7b \
        --output_dir analysis/dimension_selection_results
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
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
}

K_VALUES = [10, 25, 50, 100, 200, 500, 1000]


# ============================================================
#  GPU Probe
# ============================================================

def train_probe_gpu(X, y, num_classes=4, seed=42, epochs=50, lr=1e-3,
                    weight_decay=1e-2, batch_size=64, test_ratio=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32) if isinstance(X, np.ndarray) else X.to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device) if isinstance(y, np.ndarray) else y.long().to(device)

    mean = X_t.mean(dim=0); X_t -= mean
    std = X_t.std(dim=0); std[std < 1e-8] = 1.0; X_t /= std

    n_test = max(1, int(N * test_ratio))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    X_train, y_train = X_t[perm[:-n_test]], y_t[perm[:-n_test]]
    X_test, y_test = X_t[perm[-n_test:]], y_t[perm[-n_test:]]
    del X_t, y_t

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            b = idx[i:i+batch_size]
            loss = criterion(model(X_train[b]), y_train[b])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item() * 100
    del model, optimizer, X_train, X_test
    torch.cuda.empty_cache()
    return acc


# ============================================================
#  Feature Loading + Mean Pool
# ============================================================

def load_meanpooled(feat_root, task, stage):
    """stage별 mean-pooled feature 로드 → (N, hidden_dim)."""
    task_full = TASK_FULL(task)
    stage_dir = os.path.join(feat_root, stage, task_full)

    if stage == "vision_encoder":
        meta = np.load(os.path.join(stage_dir, "meta.npy"), allow_pickle=True).item()
        feat = np.array(np.load(os.path.join(stage_dir, "features.npy"), mmap_mode='r'))
        labels = np.load(os.path.join(stage_dir, "labels.npy"))
        nf = meta.get("num_frames", 8)
        tpf = meta.get("tokens_per_frame_pre", 729)
        hd = meta.get("vision_hidden_dim", 1152)
        feat = feat.reshape(-1, nf, tpf, hd).mean(axis=(1, 2)).astype(np.float32)
        return feat, labels, hd

    elif stage == "after_projector":
        meta = np.load(os.path.join(stage_dir, "meta.npy"), allow_pickle=True).item()
        feat = np.array(np.load(os.path.join(stage_dir, "features.npy"), mmap_mode='r'))
        labels = np.load(os.path.join(stage_dir, "labels.npy"))
        nf = meta.get("num_frames", 8)
        tpf = meta.get("tokens_per_frame_post", 196)
        hd = meta.get("hidden_dim", 3584)
        feat = feat.reshape(-1, nf, tpf, hd).mean(axis=(1, 2)).astype(np.float32)
        return feat, labels, hd

    elif stage == "vision_token":
        # Last layer, mean pool
        vt_dir = os.path.join(feat_root, "vision_token", task_full)
        meta = np.load(os.path.join(vt_dir, "meta.npy"), allow_pickle=True).item()
        labels = np.load(os.path.join(vt_dir, "labels.npy"))
        nl = meta.get("num_layers", 29)
        nf = meta.get("num_frames", 8)
        tpf = meta.get("tokens_per_frame_post", 196)
        hd = meta.get("hidden_dim", 3584)
        feat = np.array(np.load(os.path.join(vt_dir, f"features_layer_{nl-1}.npy"), mmap_mode='r'))
        feat = feat.reshape(-1, nf, tpf, hd).mean(axis=(1, 2)).astype(np.float32)
        return feat, labels, hd


# ============================================================
#  Dimension Selection Methods (all GPU)
# ============================================================

def compute_fisher_topk(X_np, y_np, k, num_classes=4):
    """Top-k Fisher criterion dimensions 선택 (GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(device, dtype=torch.float32)
    y = torch.from_numpy(y_np).long().to(device)
    global_mean = X.mean(dim=0)

    var_b = torch.zeros(X.shape[1], device=device)
    var_w = torch.zeros(X.shape[1], device=device)
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() == 0: continue
        Xc = X[mask]
        cm = Xc.mean(dim=0)
        var_b += mask.sum().float() * (cm - global_mean) ** 2
        var_w += (Xc - cm).pow(2).sum(dim=0)

    fisher = var_b / (var_w + 1e-8)
    topk_idx = fisher.argsort(descending=True)[:k]
    return X[:, topk_idx].cpu().numpy(), topk_idx.cpu().numpy(), fisher.cpu().numpy()


def compute_pca_topk(X_np, k):
    """Top-k PCA components (GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(device, dtype=torch.float32)
    X_centered = X - X.mean(dim=0)
    U, S, V = torch.pca_lowrank(X_centered, q=min(k + 10, X.shape[0], X.shape[1]))
    X_pca = (X_centered @ V[:, :k]).cpu().numpy()
    var_explained = (S[:k] ** 2 / (S ** 2).sum()).cpu().numpy()
    return X_pca, var_explained


def select_random_dims(X_np, k, seed=42):
    """Random k dimensions (control)."""
    np.random.seed(seed)
    D = X_np.shape[1]
    idx = np.random.choice(D, size=min(k, D), replace=False)
    return X_np[:, idx]


# ============================================================
#  Main Experiment
# ============================================================

def run_dimension_sweep(feat_root, model_name, task, stage, output_dir):
    """한 model/task/stage에 대해 k sweep."""
    print(f"\n  [{model_name}] {task} / {stage}")

    feat, labels, hd = load_meanpooled(feat_root, task, stage)
    print(f"    Feature shape: {feat.shape} (mean-pooled, {hd}d)")

    # Full dim baseline
    full_acc = train_probe_gpu(feat, labels)
    print(f"    Full ({hd}d): {full_acc:.1f}%")

    results = {"full_dim": hd, "full_acc": full_acc, "sweeps": {}}

    for k in K_VALUES:
        if k > hd:
            continue
        print(f"    k={k}:", end="")

        # Fisher top-k
        feat_fisher, _, _ = compute_fisher_topk(feat, labels, k)
        acc_fisher = train_probe_gpu(feat_fisher, labels)

        # PCA top-k
        feat_pca, var_exp = compute_pca_topk(feat, k)
        acc_pca = train_probe_gpu(feat_pca, labels)

        # Random top-k
        feat_rand = select_random_dims(feat, k)
        acc_rand = train_probe_gpu(feat_rand, labels)

        results["sweeps"][k] = {
            "fisher": acc_fisher, "pca": acc_pca, "random": acc_rand,
            "pca_var_explained": float(var_exp.sum()),
        }
        print(f" Fisher={acc_fisher:.1f}%, PCA={acc_pca:.1f}%, Random={acc_rand:.1f}%"
              f" (PCA var={var_exp.sum()*100:.1f}%)")

    del feat
    return results


def plot_sweep(all_results, model_name, output_dir):
    """4 tasks × stage sweep plot."""
    stages = ["vision_encoder", "after_projector", "vision_token"]
    stage_labels = ["Vision Encoder (1152d)", "After Projector (3584d)", "Vision Token Last (3584d)"]
    stage_colors = {"fisher": "#e74c3c", "pca": "#3498db", "random": "#95a5a6"}

    for stage, stage_label in zip(stages, stage_labels):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)

        for ax, task, tl in zip(axes, TASKS, TASK_LABELS):
            key = f"{task}_{stage}"
            if key not in all_results:
                ax.set_title(f"{tl}\n(no data)")
                continue

            r = all_results[key]
            ks = sorted([int(k) for k in r["sweeps"].keys()])
            full_dim = r["full_dim"]

            for method, color, marker in [("fisher", "#e74c3c", "o"), ("pca", "#3498db", "s"), ("random", "#95a5a6", "^")]:
                accs = [r["sweeps"][str(k) if str(k) in r["sweeps"] else k][method] for k in ks]
                ax.plot(ks, accs, color=color, marker=marker, markersize=4, linewidth=1.5, label=method.capitalize())

            ax.axhline(y=r["full_acc"], color="black", linestyle="--", alpha=0.5,
                        label=f"Full ({full_dim}d) = {r['full_acc']:.1f}%")
            ax.axhline(y=25, color="gray", linestyle=":", alpha=0.3)

            ax.set_title(tl, fontsize=11, fontweight="bold")
            ax.set_xlabel("k (num dims)")
            ax.set_xscale("log")
            ax.set_ylim(0, 100)
            if ax == axes[0]:
                ax.set_ylabel("Test Acc (%)")

        axes[-1].legend(fontsize=7, loc="lower right")
        short = model_name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")
        fig.suptitle(f"Dimension Selection: {short} — {stage_label}",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"dim_sweep_{model_name}_{stage}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-video-7b")
    parser.add_argument("--output_dir", type=str, default="analysis/dimension_selection_results")
    args = parser.parse_args()

    feat_root = FEAT_ROOTS.get(args.model)
    if not feat_root or not os.path.exists(feat_root):
        print(f"[ERROR] Not found: {feat_root}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    stages = ["vision_encoder", "after_projector", "vision_token"]

    for task in TASKS:
        for stage in stages:
            stage_dir = os.path.join(feat_root, stage if stage != "vision_token" else "vision_token",
                                      TASK_FULL(task))
            if not os.path.exists(stage_dir):
                print(f"  [SKIP] {task}/{stage}: not found")
                continue

            r = run_dimension_sweep(feat_root, args.model, task, stage, args.output_dir)
            all_results[f"{task}_{stage}"] = r

    # Save JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    json_path = os.path.join(args.output_dir, f"dim_selection_{args.model}.json")
    with open(json_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {json_path}")

    plot_sweep(all_results, args.model, args.output_dir)


if __name__ == "__main__":
    main()
