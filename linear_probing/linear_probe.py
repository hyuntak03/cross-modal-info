"""
통합 Linear Probing + 시각화 스크립트.

3단계 feature를 자동 탐색하여 probing하고 통합 시각화:
  - Vision Encoder (pre-projector): 수평선
  - After Projector: 수평선
  - LLM Per-Layer Vision Token: 선 그래프

x축: LLM Layer, y축: Test Accuracy (%)

Usage:
    python linear_probing/linear_probe.py \
        --feat_base_dir linear_probe_features/llava-video-7b \
        --task vlm_direction_testbed_R2R_shape_color \
        --output_dir output/llava-video-7b/linear_probe_results/vlm_direction_testbed_R2R_shape_color
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#  GPU-only Linear Probe (numpy CPU 연산 제거)
# ============================================================

def train_linear_probe_gpu(X, y, num_classes, args, desc=""):
    """전체 파이프라인 GPU: load → standardize → split → train → eval."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape

    # numpy → GPU tensor (float16 → float32 변환도 GPU에서)
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    del X  # numpy 즉시 해제

    # GPU standardize
    mean = X_t.mean(dim=0)
    X_t -= mean
    std = X_t.std(dim=0)
    std[std < 1e-8] = 1.0
    X_t /= std
    del mean, std

    # GPU train/test split (sklearn 대신 index 기반)
    n_test = max(1, int(N * args.test_ratio))
    n_train = N - n_test

    generator = torch.Generator(device='cpu')
    generator.manual_seed(args.seed)
    perm = torch.randperm(N, generator=generator)
    train_idx = perm[:n_train].to(device)
    test_idx = perm[n_train:].to(device)

    X_train = X_t[train_idx]
    y_train = y_t[train_idx]
    X_test = X_t[test_idx]
    y_test = y_t[test_idx]
    del X_t, y_t, train_idx, test_idx

    # Model — seed 고정 for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    bs = args.batch_size
    n_batches = (n_train + bs - 1) // bs

    pbar = tqdm(range(args.epochs), desc=f"    Training {desc}", leave=False)
    for epoch in pbar:
        # Shuffle train indices each epoch
        idx = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_idx = idx[i * bs: (i + 1) * bs]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    del model, optimizer, X_train, X_test, y_train, y_test
    torch.cuda.empty_cache()

    return train_acc * 100, test_acc * 100


# ============================================================
#  메인 probing
# ============================================================

def run_probe(args):
    base = args.feat_base_dir
    task = args.task

    results = []
    num_classes = None
    model_name = None

    # --- 1. Vision Encoder ---
    ve_dir = os.path.join(base, "vision_encoder", task)
    ve_feat_path = os.path.join(ve_dir, "features.npy")
    if os.path.exists(ve_feat_path):
        print("\n[Stage] Vision Encoder (pre-projector)")
        X = np.load(ve_feat_path, mmap_mode='r')
        labels = np.load(os.path.join(ve_dir, "labels.npy"))
        meta = np.load(os.path.join(ve_dir, "meta.npy"), allow_pickle=True).item()
        num_classes = meta["num_classes"]
        model_name = meta.get("model_name", "unknown")
        print(f"  Feature dim: {X.shape[1]}, samples: {X.shape[0]}")
        # mmap → numpy copy (GPU로 보내기 위해)
        X_copy = np.array(X)
        del X
        train_acc, test_acc = train_linear_probe_gpu(X_copy, labels, num_classes, args, "VE")
        results.append({"stage": "vision_encoder", "layer": -2, "train_acc": train_acc, "test_acc": test_acc})
        print(f"  Vision Encoder: train={train_acc:.1f}% test={test_acc:.1f}%")

    # --- 2. After Projector ---
    ap_dir = os.path.join(base, "after_projector", task)
    ap_feat_path = os.path.join(ap_dir, "features.npy")
    if os.path.exists(ap_feat_path):
        print("\n[Stage] After Projector")
        X = np.load(ap_feat_path, mmap_mode='r')
        labels = np.load(os.path.join(ap_dir, "labels.npy"))
        meta = np.load(os.path.join(ap_dir, "meta.npy"), allow_pickle=True).item()
        num_classes = num_classes or meta["num_classes"]
        model_name = model_name or meta.get("model_name", "unknown")
        print(f"  Feature dim: {X.shape[1]}, samples: {X.shape[0]}")
        X_copy = np.array(X)
        del X
        train_acc, test_acc = train_linear_probe_gpu(X_copy, labels, num_classes, args, "AP")
        results.append({"stage": "after_projector", "layer": -1, "train_acc": train_acc, "test_acc": test_acc})
        print(f"  After Projector: train={train_acc:.1f}% test={test_acc:.1f}%")

    # --- 3. After Gate (channel_gate 적용 후) ---
    ag_dir = os.path.join(base, "after_gate", task)
    ag_feat_path = os.path.join(ag_dir, "features.npy")
    if os.path.exists(ag_feat_path):
        print("\n[Stage] After Gate (channel_gate)")
        X = np.load(ag_feat_path, mmap_mode='r')
        labels = np.load(os.path.join(ag_dir, "labels.npy"))
        meta = np.load(os.path.join(ag_dir, "meta.npy"), allow_pickle=True).item()
        num_classes = num_classes or meta["num_classes"]
        model_name = model_name or meta.get("model_name", "unknown")
        print(f"  Feature dim: {X.shape[1]}, samples: {X.shape[0]}")
        X_copy = np.array(X)
        del X
        train_acc, test_acc = train_linear_probe_gpu(X_copy, labels, num_classes, args, "AG")
        results.append({"stage": "after_gate", "layer": -0.5, "train_acc": train_acc, "test_acc": test_acc})
        print(f"  After Gate: train={train_acc:.1f}% test={test_acc:.1f}%")

    # --- 4. LLM Per-Layer Vision Token ---
    vt_dir = os.path.join(base, "vision_token", task)
    vt_meta_path = os.path.join(vt_dir, "meta.npy")
    if os.path.exists(vt_meta_path):
        print("\n[Stage] LLM Per-Layer Vision Token")
        meta = np.load(vt_meta_path, allow_pickle=True).item()
        labels = np.load(os.path.join(vt_dir, "labels.npy"))
        num_layers = meta["num_layers"]
        num_classes = num_classes or meta["num_classes"]
        model_name = model_name or meta.get("model_name", "unknown")
        print(f"  {num_layers} layers, {meta['num_samples']} samples, {num_classes} classes")

        pbar = tqdm(range(num_layers), desc="  Probing layers")
        for layer_idx in pbar:
            feat_path = os.path.join(vt_dir, f"features_layer_{layer_idx}.npy")
            X = np.array(np.load(feat_path, mmap_mode='r'))
            train_acc, test_acc = train_linear_probe_gpu(X, labels, num_classes, args)
            results.append({"stage": "vision_token", "layer": layer_idx, "train_acc": train_acc, "test_acc": test_acc})
            pbar.set_postfix_str(f"L{layer_idx}: test={test_acc:.1f}%")

    if not results:
        print("[ERROR] No features found. Check feat_base_dir and task.")
        return

    # --- 결과 저장 ---
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "linear_probe_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # --- 통합 시각화 ---
    plot_unified(df, args.output_dir, num_classes, model_name, task)


# ============================================================
#  통합 시각화
# ============================================================

def plot_unified(df, output_dir, num_classes, model_name, task):
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 5))

    # LLM per-layer line
    vt = df[df["stage"] == "vision_token"].sort_values("layer")
    if len(vt):
        ax.plot(vt["layer"], vt["test_acc"], label="LLM Vision Token",
                color="#5c95ff", linewidth=2, marker='o', markersize=3, zorder=3)

    # Vision Encoder horizontal line
    ve = df[df["stage"] == "vision_encoder"]
    if len(ve):
        ve_acc = ve["test_acc"].values[0]
        ax.axhline(y=ve_acc, color="#ff6b35", linestyle="--", linewidth=2, alpha=0.8,
                    label=f"Vision Encoder ({ve_acc:.1f}%)", zorder=2)

    # After Projector horizontal line
    ap = df[df["stage"] == "after_projector"]
    if len(ap):
        ap_acc = ap["test_acc"].values[0]
        ax.axhline(y=ap_acc, color="#2ecc71", linestyle="--", linewidth=2, alpha=0.8,
                    label=f"After Projector ({ap_acc:.1f}%)", zorder=2)

    # After Gate horizontal line
    ag = df[df["stage"] == "after_gate"]
    if len(ag):
        ag_acc = ag["test_acc"].values[0]
        ax.axhline(y=ag_acc, color="#e040fb", linestyle="--", linewidth=2, alpha=0.8,
                    label=f"After Gate ({ag_acc:.1f}%)", zorder=2)

    # Chance level
    chance = 100.0 / num_classes
    ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.5,
                label=f"Chance ({chance:.1f}%)")

    if len(vt):
        ax.set_xlim(0, vt["layer"].max())

    ax.set_ylim(0, 100)
    ax.set_xlabel("LLM Layer", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Linear Probing: Vision Information Flow\n{model_name} / {task}", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "linear_probe_unified.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"[SAVED] {save_path}")


# ============================================================
#  메인
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified linear probing + visualization")
    parser.add_argument("--feat_base_dir", type=str, required=True,
                        help="e.g., linear_probe_features/llava-video-7b")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    run_probe(args)
