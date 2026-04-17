"""
Vision Token vs Answer Token Linear Probing 심층 분석.

핵심 의문: Vision token probe acc는 낮은데 answer token probe는 높은 이유는?

실험:
  Exp1. Pooling Strategy — 차원 불균형이 원인인가?
    (a) All tokens flattened (5.6M dim, 현재 방식)
    (b) Mean pool over all tokens → (3584 dim)
    (c) Per-frame mean pool → (8 × 3584 = 28672 dim)
    (d) Temporal delta mean pool → (7 × 3584 = 25088 dim)

  Exp2. Per-Frame Probing — 시간 정보가 핵심인가?
    (a) 단일 프레임 (frame 0, 1, ...) → direction 정보 있나?
    (b) Temporal delta (frame[t+1] - frame[t]) → 프레임 간 차이에 있나?

  Exp3. Cross-Task Generalization — 진짜 direction인가?
    (a) Train: shape_color → Test: obj_place (같은 direction, 다른 identity)
    (b) Answer token에서도 동일 cross-task probe

  Exp4. Label Shuffle Control — data leakage 없는지 확인
    (a) Random label → chance 나와야 정상

  Exp5. MLP Probe — 비선형 분리가 필요한 정보인가?
    (a) Vision token mean-pooled + 2-layer MLP
    (b) Linear vs MLP 비교

모든 probe는 GPU (torch), CPU 최소화.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/probe_analysis.py \
        --model llava-video-7b \
        --task shape_color \
        --experiments all
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
    "llava-video-7b_lora_syn_v4_channel_gate": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_channel_gate",
    "llava-video-7b_lora_syn_v4_dual_delta": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_dual_delta",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
    "llava-video-7b_lora_4combo_v2_delta": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_delta",
}


# ============================================================
#  GPU Probe (from linear_probe.py)
# ============================================================

def train_probe_gpu(X, y, num_classes, test_ratio=0.3, seed=42, epochs=50,
                    lr=1e-3, weight_decay=1e-2, batch_size=64, mlp_hidden=0):
    """GPU-only probe. mlp_hidden>0이면 MLP, 아니면 Linear."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape

    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    del X

    # GPU standardize
    mean = X_t.mean(dim=0)
    X_t -= mean
    std = X_t.std(dim=0)
    std[std < 1e-8] = 1.0
    X_t /= std
    del mean, std

    # Split
    n_test = max(1, int(N * test_ratio))
    n_train = N - n_test
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_train].to(device)
    test_idx = perm[n_train:].to(device)

    X_train, y_train = X_t[train_idx], y_t[train_idx]
    X_test, y_test = X_t[test_idx], y_t[test_idx]
    del X_t, y_t

    # Model
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if mlp_hidden > 0:
        model = nn.Sequential(
            nn.Linear(D, mlp_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden, num_classes)
        ).to(device)
    else:
        model = nn.Linear(D, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    bs = batch_size
    for epoch in range(epochs):
        idx = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            batch_idx = idx[i:i+bs]
            logits = model(X_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item() * 100
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item() * 100

    del model, optimizer, X_train, X_test, y_train, y_test
    torch.cuda.empty_cache()
    return train_acc, test_acc


# ============================================================
#  Feature Loading
# ============================================================

def load_vision_features(feat_root, task, layer_idx=-1):
    """Vision token features 로드 + reshape 정보."""
    task_full = TASK_FULL(task)
    vt_dir = os.path.join(feat_root, "vision_token", task_full)
    meta = np.load(os.path.join(vt_dir, "meta.npy"), allow_pickle=True).item()

    if layer_idx < 0:
        layer_idx = meta["num_layers"] + layer_idx

    feat = np.load(os.path.join(vt_dir, f"features_layer_{layer_idx}.npy"), mmap_mode='r')
    labels = np.load(os.path.join(vt_dir, "labels.npy"))

    num_frames = meta.get("num_frames", 8)
    tokens_per_frame = meta.get("tokens_per_frame_post", meta.get("tokens_per_frame", 196))
    hidden_dim = meta.get("hidden_dim", 3584)

    return np.array(feat), labels, num_frames, tokens_per_frame, hidden_dim, meta


def load_answer_features(feat_root, task, layer_idx=-1):
    """Answer token features 로드."""
    task_full = TASK_FULL(task)
    at_dir = os.path.join(feat_root, "answer_token", task_full)
    meta = np.load(os.path.join(at_dir, "meta.npy"), allow_pickle=True).item()

    if layer_idx < 0:
        layer_idx = meta["num_layers"] + layer_idx

    feat = np.load(os.path.join(at_dir, f"features_layer_{layer_idx}.npy"), mmap_mode='r')
    labels = np.load(os.path.join(at_dir, "labels.npy"))
    return np.array(feat), labels, meta


# ============================================================
#  Exp1: Pooling Strategy Comparison
# ============================================================

def exp1_pooling(feat_root, task, layer_idx=-1, num_classes=4):
    """Vision token을 다양한 pooling 전략으로 probe → 차원 불균형 효과 검증."""
    print(f"\n{'='*60}")
    print(f"  Exp1: Pooling Strategy — {task} (layer {layer_idx})")
    print(f"{'='*60}")

    feat, labels, nf, tpf, hd, meta = load_vision_features(feat_root, task, layer_idx)
    N = feat.shape[0]

    results = {}

    # (a) Flattened (current)
    print(f"  (a) Flattened: {feat.shape}")
    _, test_acc = train_probe_gpu(feat.astype(np.float32), labels, num_classes)
    results['flattened'] = test_acc
    print(f"      → test_acc = {test_acc:.1f}%")

    # Reshape: (N, nf * tpf * hd) → (N, nf, tpf, hd)
    feat_4d = feat.reshape(N, nf, tpf, hd)

    # (b) Mean pool all tokens → (N, hd)
    feat_mean = feat_4d.mean(axis=(1, 2)).astype(np.float32)
    print(f"  (b) Mean pool (all tokens): {feat_mean.shape}")
    _, test_acc = train_probe_gpu(feat_mean, labels, num_classes)
    results['mean_all'] = test_acc
    print(f"      → test_acc = {test_acc:.1f}%")
    del feat_mean

    # (c) Per-frame mean → (N, nf * hd)
    feat_frame_mean = feat_4d.mean(axis=2).reshape(N, -1).astype(np.float32)
    print(f"  (c) Per-frame mean: {feat_frame_mean.shape}")
    _, test_acc = train_probe_gpu(feat_frame_mean, labels, num_classes)
    results['per_frame_mean'] = test_acc
    print(f"      → test_acc = {test_acc:.1f}%")
    del feat_frame_mean

    # (d) Temporal delta mean → (N, (nf-1) * hd)
    frame_means = feat_4d.mean(axis=2)  # (N, nf, hd)
    deltas = frame_means[:, 1:, :] - frame_means[:, :-1, :]  # (N, nf-1, hd)
    feat_delta = deltas.reshape(N, -1).astype(np.float32)
    print(f"  (d) Temporal delta mean: {feat_delta.shape}")
    _, test_acc = train_probe_gpu(feat_delta, labels, num_classes)
    results['temporal_delta'] = test_acc
    print(f"      → test_acc = {test_acc:.1f}%")
    del feat_delta, deltas, frame_means

    # (e) Answer token reference
    ans_feat, ans_labels, _ = load_answer_features(feat_root, task, layer_idx)
    print(f"  (e) Answer token: {ans_feat.shape}")
    _, test_acc = train_probe_gpu(ans_feat.astype(np.float32), ans_labels, num_classes)
    results['answer_token'] = test_acc
    print(f"      → test_acc = {test_acc:.1f}%")

    del feat, feat_4d
    return results


# ============================================================
#  Exp2: Per-Frame Probing
# ============================================================

def exp2_per_frame(feat_root, task, layer_idx=-1, num_classes=4):
    """프레임별 개별 probe → 시간 정보 분석."""
    print(f"\n{'='*60}")
    print(f"  Exp2: Per-Frame Probing — {task} (layer {layer_idx})")
    print(f"{'='*60}")

    feat, labels, nf, tpf, hd, meta = load_vision_features(feat_root, task, layer_idx)
    N = feat.shape[0]
    feat_4d = feat.reshape(N, nf, tpf, hd)

    results = {}

    # Per-frame mean pool → (N, hd) per frame
    for f in range(nf):
        feat_f = feat_4d[:, f, :, :].mean(axis=1).astype(np.float32)  # (N, hd)
        _, test_acc = train_probe_gpu(feat_f, labels, num_classes)
        results[f'frame_{f}'] = test_acc
        print(f"  Frame {f}: {test_acc:.1f}%")

    # Temporal delta per adjacent pair
    frame_means = feat_4d.mean(axis=2)  # (N, nf, hd)
    for t in range(nf - 1):
        delta = (frame_means[:, t+1, :] - frame_means[:, t, :]).astype(np.float32)
        _, test_acc = train_probe_gpu(delta, labels, num_classes)
        results[f'delta_{t}_{t+1}'] = test_acc
        print(f"  Delta {t}→{t+1}: {test_acc:.1f}%")

    del feat, feat_4d, frame_means
    return results


# ============================================================
#  Exp3: Cross-Task Generalization
# ============================================================

def exp3_cross_task(feat_root, num_classes=4, layer_idx=-1):
    """Train on task A, test on task B → direction generalization 검증."""
    print(f"\n{'='*60}")
    print(f"  Exp3: Cross-Task Generalization (layer {layer_idx})")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all tasks' answer token features
    all_feats = {}
    all_labels = {}
    for task in TASKS:
        feat, labels, _ = load_answer_features(feat_root, task, layer_idx)
        all_feats[task] = torch.from_numpy(feat.astype(np.float32)).to(device)
        all_labels[task] = torch.from_numpy(labels).long().to(device)

    results = {}

    for train_task in TASKS:
        X_train = all_feats[train_task]
        y_train = all_labels[train_task]

        # Standardize based on train
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0)
        std[std < 1e-8] = 1.0
        X_train_norm = (X_train - mean) / std

        # Train probe
        D = X_train.shape[1]
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = nn.Linear(D, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(50):
            logits = model(X_train_norm)
            loss = criterion(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test on all tasks
        model.eval()
        row = {}
        with torch.no_grad():
            for test_task in TASKS:
                X_test = (all_feats[test_task] - mean) / std
                acc = (model(X_test).argmax(1) == all_labels[test_task]).float().mean().item() * 100
                row[test_task] = acc
        results[train_task] = row
        del model, optimizer

        # Print row
        vals = " | ".join(f"{results[train_task][t]:5.1f}" for t in TASKS)
        print(f"  Train: {train_task:<15} → Test: {vals}")

    torch.cuda.empty_cache()
    return results


# ============================================================
#  Exp4: Label Shuffle Control
# ============================================================

def exp4_shuffle(feat_root, task, layer_idx=-1, num_classes=4, n_repeats=5):
    """Labels를 random shuffle → chance인지 확인."""
    print(f"\n{'='*60}")
    print(f"  Exp4: Label Shuffle Control — {task}")
    print(f"{'='*60}")

    ans_feat, ans_labels, _ = load_answer_features(feat_root, task, layer_idx)

    # Normal
    _, real_acc = train_probe_gpu(ans_feat.astype(np.float32), ans_labels, num_classes)
    print(f"  Real labels:     {real_acc:.1f}%")

    # Shuffled (multiple trials)
    shuffle_accs = []
    for i in range(n_repeats):
        shuffled = ans_labels.copy()
        np.random.seed(i + 100)
        np.random.shuffle(shuffled)
        _, sacc = train_probe_gpu(ans_feat.astype(np.float32), shuffled, num_classes, seed=i+100)
        shuffle_accs.append(sacc)

    mean_shuffle = np.mean(shuffle_accs)
    std_shuffle = np.std(shuffle_accs)
    print(f"  Shuffled labels: {mean_shuffle:.1f}% ± {std_shuffle:.1f}% (chance={100/num_classes:.1f}%)")

    return {"real": real_acc, "shuffled_mean": mean_shuffle, "shuffled_std": std_shuffle}


# ============================================================
#  Exp5: MLP Probe
# ============================================================

def exp5_mlp(feat_root, task, layer_idx=-1, num_classes=4):
    """Linear vs MLP probe on mean-pooled vision tokens."""
    print(f"\n{'='*60}")
    print(f"  Exp5: Linear vs MLP — {task} (layer {layer_idx})")
    print(f"{'='*60}")

    feat, labels, nf, tpf, hd, meta = load_vision_features(feat_root, task, layer_idx)
    N = feat.shape[0]
    feat_4d = feat.reshape(N, nf, tpf, hd)
    feat_mean = feat_4d.mean(axis=(1, 2)).astype(np.float32)  # (N, hd)

    # Linear
    _, linear_acc = train_probe_gpu(feat_mean, labels, num_classes, mlp_hidden=0)
    print(f"  Linear (mean pool): {linear_acc:.1f}%")

    # MLP 256
    _, mlp_acc = train_probe_gpu(feat_mean.copy(), labels, num_classes, mlp_hidden=256)
    print(f"  MLP-256 (mean pool): {mlp_acc:.1f}%")

    # MLP 512
    _, mlp512_acc = train_probe_gpu(feat_mean.copy(), labels, num_classes, mlp_hidden=512)
    print(f"  MLP-512 (mean pool): {mlp512_acc:.1f}%")

    del feat, feat_4d, feat_mean
    return {"linear": linear_acc, "mlp_256": mlp_acc, "mlp_512": mlp512_acc}


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Vision vs Answer Token Probe Deep Analysis")
    parser.add_argument("--model", type=str, default="llava-video-7b")
    parser.add_argument("--task", type=str, default="all",
                        help="Task name or 'all'")
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated: exp1,exp2,exp3,exp4,exp5 or 'all'")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="analysis/results")
    args = parser.parse_args()

    feat_root = FEAT_ROOTS.get(args.model)
    if feat_root is None or not os.path.exists(feat_root):
        print(f"[ERROR] Feature root not found: {feat_root}")
        return

    tasks = TASKS if args.task == "all" else [args.task]
    exps = ["exp1", "exp2", "exp3", "exp4", "exp5"] if args.experiments == "all" else args.experiments.split(",")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {args.model} / TASK: {task}")
        print(f"{'#'*60}")

        task_results = {}

        if "exp1" in exps:
            task_results["exp1_pooling"] = exp1_pooling(feat_root, task, args.layer)

        if "exp2" in exps:
            task_results["exp2_per_frame"] = exp2_per_frame(feat_root, task, args.layer)

        if "exp4" in exps:
            task_results["exp4_shuffle"] = exp4_shuffle(feat_root, task, args.layer)

        if "exp5" in exps:
            task_results["exp5_mlp"] = exp5_mlp(feat_root, task, args.layer)

        all_results[task] = task_results

    # Exp3 is cross-task (runs once, not per task)
    if "exp3" in exps:
        all_results["exp3_cross_task"] = exp3_cross_task(feat_root, layer_idx=args.layer)

    # Save
    save_path = os.path.join(args.output_dir, f"probe_analysis_{args.model}.json")
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(save_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {save_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.model}")
    print(f"{'='*60}")
    for task in tasks:
        if "exp1_pooling" in all_results.get(task, {}):
            r = all_results[task]["exp1_pooling"]
            print(f"\n  {task}:")
            print(f"    Flattened (5.6M):    {r.get('flattened', 'N/A'):.1f}%")
            print(f"    Mean pool (3584):    {r.get('mean_all', 'N/A'):.1f}%")
            print(f"    Per-frame (28672):   {r.get('per_frame_mean', 'N/A'):.1f}%")
            print(f"    Temporal delta:      {r.get('temporal_delta', 'N/A'):.1f}%")
            print(f"    Answer token (3584): {r.get('answer_token', 'N/A'):.1f}%")


if __name__ == "__main__":
    main()
