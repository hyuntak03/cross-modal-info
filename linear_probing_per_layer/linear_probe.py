"""
Layer별 Linear Probing 스크립트 (PyTorch GPU).

extract_vision_features.py로 저장한 features를 로드하여
각 layer마다 nn.Linear로 linear probing 수행.

Usage:
    python linear_probing_per_layer/linear_probe.py \
        --feature_dir linear_probe_features/MODEL_NAME/TASK \
        --output_dir output/MODEL_NAME/linear_probe_results/TASK \
        --test_ratio 0.2
"""

import os
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


def train_linear_probe(X_train, y_train, X_test, y_test, num_classes, args):
    """nn.Linear + CrossEntropyLoss로 linear probing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = X_train.shape[1]

    # Tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    model = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(args.epochs):
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).argmax(dim=1)
        train_acc = (train_preds == y_train_t).float().mean().item()

        test_preds = model(X_test_t).argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()

    return train_acc, test_acc


def run_probe(args):
    # 메타 정보 로드
    meta = np.load(os.path.join(args.feature_dir, "meta.npy"), allow_pickle=True).item()
    labels = np.load(os.path.join(args.feature_dir, "labels.npy"))

    num_layers = meta["num_layers"]
    num_classes = meta["num_classes"]
    label_list = meta["label_list"]
    model_name = meta["model_name"]
    task = meta["task"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Model: {model_name}, Task: {task}")
    print(f"[INFO] {meta['num_samples']} samples, {num_layers} layers, {num_classes} classes")
    print(f"[INFO] Label distribution: {np.bincount(labels, minlength=num_classes)}")
    print(f"[INFO] Device: {device}, epochs: {args.epochs}, lr: {args.lr}, wd: {args.weight_decay}")

    results = []

    pbar = tqdm(range(num_layers), desc="Linear probing")
    for layer_idx in pbar:
        pbar.set_description(f"Layer {layer_idx}: loading")
        feat_path = os.path.join(args.feature_dir, f"features_layer_{layer_idx}.npy")
        X = np.load(feat_path).astype(np.float32)

        # Standardize (numpy)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=args.test_ratio,
            random_state=args.seed, stratify=labels,
        )

        # Linear probe on GPU
        pbar.set_description(f"Layer {layer_idx}: training")
        train_acc, test_acc = train_linear_probe(
            X_train, y_train, X_test, y_test, num_classes, args
        )

        result = {
            "layer": layer_idx,
            "train_acc": train_acc * 100,
            "test_acc": test_acc * 100,
        }
        results.append(result)
        pbar.set_postfix_str(f"train={train_acc*100:.1f}% test={test_acc*100:.1f}%")

    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "linear_probe_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    # Plot
    plot_results(df, args.output_dir, num_layers, num_classes, model_name, task)


def plot_results(df, output_dir, num_layers, num_classes, model_name, task):
    """Layer별 linear probing accuracy plot."""
    sns.set(context="notebook")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["layer"], df["test_acc"], label="Test Acc", color="#5c95ff", linewidth=2)
    ax.plot(df["layer"], df["train_acc"], label="Train Acc", color="#f20089", linewidth=2, linestyle="--", alpha=0.6)

    # chance level
    chance = 100.0 / num_classes
    ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.5, label=f"Chance ({chance:.1f}%)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Linear Probing per Layer\n{model_name} / {task}")
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "linear_probe_accuracy.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150)
    plt.close()
    print(f"[SAVED] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probing per layer (PyTorch GPU)")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="e.g., output/MODEL_NAME/linear_probe_results/TASK")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    run_probe(args)
