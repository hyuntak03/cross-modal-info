"""
Vision Encoder / Projector Feature에 대한 Linear / MLP Probing.

extract_features.py로 저장한 pre_projector, post_projector features를
각각 probing하여 비교.

Usage (linear):
    python linear_probing_before_llm/linear_probe.py \
        --feature_dir output/before_llm_features/llava-video-7b/identity_testbed_realobj_realbg \
        --output_dir output/before_llm_results/llava-video-7b/identity_testbed_realobj_realbg

Usage (MLP):
    python linear_probing_before_llm/linear_probe.py \
        --feature_dir output/before_llm_features/llava-video-7b/identity_testbed_realobj_realbg \
        --output_dir output/before_llm_results_mlp/llava-video-7b/identity_testbed_realobj_realbg \
        --probe_type mlp --mlp_hidden 256 --mlp_dropout 0.3
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _timer(msg):
    """간단한 타이머 컨텍스트매니저."""
    class Timer:
        def __enter__(self):
            self.t = time.time()
            print(f"  [{msg}] ...", end="", flush=True)
            return self
        def __exit__(self, *args):
            print(f" done ({time.time() - self.t:.1f}s)")
    return Timer()


def _build_probe(probe_type, feat_dim, num_classes, mlp_hidden=256, mlp_dropout=0.3):
    """Linear 또는 MLP probe 생성."""
    if probe_type == "linear":
        return nn.Linear(feat_dim, num_classes)
    elif probe_type == "mlp":
        return nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, num_classes),
        )
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")


def train_probe(X_train, y_train, X_test, y_test, num_classes, args):
    """Mini-batch GPU 전송: feature는 RAM, batch만 GPU로."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = X_train.shape[1]

    # CPU에 텐서 유지, DataLoader가 batch만 꺼냄
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).long()
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = _build_probe(
        args.probe_type, feat_dim, num_classes,
        mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    probe_label = f"  Training ({args.probe_type})"
    if args.probe_type == "mlp":
        n_params = sum(p.numel() for p in model.parameters())
        probe_label += f" [hidden={args.mlp_hidden}, dropout={args.mlp_dropout}, params={n_params:,}]"

    # Train — batch만 GPU
    model.train()
    pbar = tqdm(range(args.epochs), desc=probe_label)
    for epoch in pbar:
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{epoch_loss / len(train_loader):.4f}")

    # Eval — batch 단위
    model.eval()
    train_correct, train_total = 0, 0
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for xb, yb in tqdm(train_loader, desc="  Eval (train)"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            train_correct += (model(xb).argmax(1) == yb).sum().item()
            train_total += yb.size(0)
        for xb, yb in tqdm(test_loader, desc="  Eval (test)"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            test_correct += (model(xb).argmax(1) == yb).sum().item()
            test_total += yb.size(0)

    return train_correct / train_total, test_correct / test_total


def _load_features(feature_dir, stage, num_ranks=None):
    """단일 파일 또는 rank별 분할 파일을 로드. pre-allocate로 메모리 효율적."""
    single_path = os.path.join(feature_dir, f"features_{stage}.npy")
    if os.path.exists(single_path):
        print(f"  Loading {single_path}...")
        X = np.load(single_path)
        if X.dtype != np.float32:
            print(f"  Converting {X.dtype} → float32...")
            X = X.astype(np.float32)
        return X

    # rank별 파일 로드
    if num_ranks is None:
        num_ranks = 0
        while os.path.exists(os.path.join(feature_dir, f"features_{stage}_rank{num_ranks}.npy")):
            num_ranks += 1

    if num_ranks == 0:
        return None

    # 먼저 전체 크기 파악 (header만 읽기)
    rank_paths = []
    rank_sizes = []
    feat_dim = None
    dtype = None
    for rank in range(num_ranks):
        path = os.path.join(feature_dir, f"features_{stage}_rank{rank}.npy")
        if os.path.exists(path):
            header = np.load(path, mmap_mode='r')
            rank_paths.append(path)
            rank_sizes.append(header.shape[0])
            if feat_dim is None:
                feat_dim = header.shape[1]
                dtype = header.dtype
            del header

    total = sum(rank_sizes)
    print(f"  Pre-allocating {total} x {feat_dim} (float32, converting from {dtype})...")
    X = np.empty((total, feat_dim), dtype=np.float32)

    offset = 0
    for path, size in tqdm(zip(rank_paths, rank_sizes), total=len(rank_paths), desc=f"  Loading {stage} ranks"):
        chunk = np.load(path, mmap_mode='r')
        X[offset:offset + size] = chunk  # float16→float32 자동 변환
        offset += size
        del chunk

    return X


def _load_labels(feature_dir, num_ranks=None):
    """단일 파일 또는 rank별 분할 라벨 로드."""
    single_path = os.path.join(feature_dir, "labels.npy")
    if os.path.exists(single_path):
        return np.load(single_path)

    if num_ranks is None:
        num_ranks = 0
        while os.path.exists(os.path.join(feature_dir, f"labels_rank{num_ranks}.npy")):
            num_ranks += 1

    chunks = []
    for rank in range(num_ranks):
        path = os.path.join(feature_dir, f"labels_rank{rank}.npy")
        if os.path.exists(path):
            chunks.append(np.load(path))
    return np.concatenate(chunks, axis=0) if chunks else None


def _standardize_chunked(X, chunk_size=500):
    """chunk 단위 standardization — 거대 임시 배열 생성 방지."""
    n, d = X.shape
    mean = X.mean(axis=0)
    X -= mean

    # chunk 단위로 variance 계산 (임시 배열 = chunk_size x d만 사용)
    var_acc = np.zeros(d, dtype=np.float64)
    for i in tqdm(range(0, n, chunk_size), desc="  Computing std (chunked)"):
        chunk = X[i:i+chunk_size]
        var_acc += (chunk.astype(np.float64) ** 2).sum(axis=0)
    std = np.sqrt(var_acc / n).astype(np.float32)
    std[std < 1e-8] = 1.0

    # in-place divide
    for i in tqdm(range(0, n, chunk_size), desc="  Standardizing (chunked)"):
        X[i:i+chunk_size] /= std


def run_probe(args):
    meta = np.load(os.path.join(args.feature_dir, "meta.npy"), allow_pickle=True).item()
    num_ranks = meta.get("num_ranks", None)
    labels = _load_labels(args.feature_dir, num_ranks)

    num_classes = meta["num_classes"]
    model_name = meta["model_name"]
    task = meta["task"]

    print(f"[INFO] Model: {model_name}, Task: {task}")
    print(f"[INFO] {meta['num_samples']} samples, {num_classes} classes")
    print(f"[INFO] Label distribution: {np.bincount(labels, minlength=num_classes)}")
    print(f"[INFO] Probe: {args.probe_type}" + (f" (hidden={args.mlp_hidden}, dropout={args.mlp_dropout})" if args.probe_type == "mlp" else ""))
    print(f"[INFO] epochs: {args.epochs}, lr: {args.lr}, wd: {args.weight_decay}")

    stages = ["pre_projector", "post_projector"]
    results = []

    for stage in stages:
        print(f"\n{'='*60}")
        print(f"  {stage}")
        print(f"{'='*60}")
        X = _load_features(args.feature_dir, stage, num_ranks)
        if X is None:
            print(f"  [WARN] no features found, skipping")
            continue

        feat_dim = X.shape[1]
        print(f"  Feature dim: {feat_dim}  dtype: {X.dtype}  size: {X.nbytes / 1e9:.1f}GB")

        # standardize (chunk 단위 — 거대 임시 배열 방지)
        _standardize_chunked(X)

        with _timer("Train/test split"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=args.test_ratio,
                random_state=args.seed, stratify=labels,
            )
        del X  # split 후 원본 즉시 해제

        print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

        train_acc, test_acc = train_probe(
            X_train, y_train, X_test, y_test, num_classes, args
        )
        del X_train, X_test  # 학습 후 해제

        result = {
            "stage": stage,
            "probe_type": args.probe_type,
            "feat_dim": feat_dim,
            "train_acc": train_acc * 100,
            "test_acc": test_acc * 100,
        }
        if args.probe_type == "mlp":
            result["mlp_hidden"] = args.mlp_hidden
            result["mlp_dropout"] = args.mlp_dropout
        results.append(result)
        print(f"  >>> train={train_acc*100:.1f}%  test={test_acc*100:.1f}%")

    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "linear_probe_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Plot
    if len(results) >= 2:
        plot_results(df, args.output_dir, num_classes, model_name, task, args.probe_type)


def plot_results(df, output_dir, num_classes, model_name, task, probe_type):
    sns.set(context="notebook")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))

    x = range(len(df))
    ax.bar([i - 0.15 for i in x], df["train_acc"], width=0.3, label="Train Acc", color="#f20089", alpha=0.6)
    ax.bar([i + 0.15 for i in x], df["test_acc"], width=0.3, label="Test Acc", color="#5c95ff")

    chance = 100.0 / num_classes
    ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.5, label=f"Chance ({chance:.1f}%)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["stage"], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Before-LLM {probe_type.upper()} Probing\n{model_name} / {task}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{probe_type}_probe_results.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"[SAVED] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear/MLP probing on before-LLM features")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/before_llm_results")
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)

    # Probe type
    parser.add_argument("--probe_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.3)

    args = parser.parse_args()
    run_probe(args)
