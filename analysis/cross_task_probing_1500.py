"""
Cross-task direction probing (1500 samples).

Train probe on task A's answer token (last layer), test on task B.
4x4 matrix per model. Off-diagonal이 높으면 direction이 identity-invariant.

기존 linear_probing_1500 feature 활용 (direction label은 candidates 개수와 무관).
"""

import os, json
import numpy as np
import torch
import torch.nn as nn

FEAT_ROOTS = {
    "Vanilla": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b",
    "Baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta",
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
LAST_LAYER = 28  # 0~28, 마지막


def load_features(feat_root, task, layer=LAST_LAYER):
    d = os.path.join(feat_root, "answer_token", f"vlm_direction_testbed_R2R_4way_1500_{task}")
    feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
    labels = np.load(os.path.join(d, "labels.npy"))
    return feat, labels


def train_probe(X_train, y_train, nc, seed=42, epochs=50, lr=1e-3, weight_decay=1e-2):
    device = torch.device("cuda")
    X = torch.from_numpy(X_train).to(device)
    y = torch.from_numpy(y_train).long().to(device)
    # Standardize
    mean = X.mean(0); std = X.std(0); std[std < 1e-8] = 1.0
    X = (X - mean) / std

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(X.shape[1], nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 128):
            b = idx[i:i+128]
            loss = crit(model(X[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, mean, std


def eval_probe(model, mean, std, X_test, y_test):
    device = torch.device("cuda")
    X = torch.from_numpy(X_test).to(device)
    y = torch.from_numpy(y_test).long().to(device)
    X = (X - mean) / std
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item() * 100
    return acc


def split_train_test(feat, labels, test_ratio=0.3, seed=42):
    """각 class에서 test_ratio만큼 test로 분리."""
    np.random.seed(seed)
    n = len(feat)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    return idx[n_test:], idx[:n_test]


def main():
    results = {}

    for model_name, feat_root in FEAT_ROOTS.items():
        if not os.path.exists(feat_root):
            print(f"[SKIP] {model_name}: {feat_root} 없음")
            continue

        print(f"\n{'='*70}")
        print(f"  {model_name}")
        print(f"{'='*70}")

        # 모든 task의 feature 로드
        data = {}
        for task in TASKS:
            try:
                feat, labels = load_features(feat_root, task)
                train_idx, test_idx = split_train_test(feat, labels)
                data[task] = {
                    "feat_train": feat[train_idx], "labels_train": labels[train_idx],
                    "feat_test": feat[test_idx], "labels_test": labels[test_idx],
                }
            except Exception as e:
                print(f"  [SKIP] {task}: {e}")

        # Cross-task matrix
        matrix = np.zeros((len(TASKS), len(TASKS)))
        for i, train_task in enumerate(TASKS):
            if train_task not in data: continue
            probe, mean, std = train_probe(
                data[train_task]["feat_train"],
                data[train_task]["labels_train"],
                4
            )
            for j, test_task in enumerate(TASKS):
                if test_task not in data: continue
                acc = eval_probe(probe, mean, std,
                                 data[test_task]["feat_test"],
                                 data[test_task]["labels_test"])
                matrix[i, j] = acc

        # Print
        print(f"\n  Train→Test (rows=train, cols=test):")
        header = "           " + "".join(f"{t:>13}" for t in TASKS)
        print(header)
        for i, train_task in enumerate(TASKS):
            row = f"  {train_task:>9} "
            for j in range(len(TASKS)):
                val = matrix[i, j]
                marker = "*" if i == j else " "
                row += f"{val:>11.1f}%{marker}"
            print(row)

        # Diagonal mean vs off-diagonal mean
        diag = np.diag(matrix).mean()
        off_diag = (matrix.sum() - np.trace(matrix)) / (len(TASKS)**2 - len(TASKS))
        print(f"\n  Diagonal (in-task):     {diag:.1f}%")
        print(f"  Off-diagonal (cross):   {off_diag:.1f}%")
        print(f"  Transfer gap:           {diag - off_diag:.1f}%p")

        results[model_name] = {
            "matrix": matrix.tolist(),
            "tasks": TASKS,
            "diagonal_mean": diag,
            "off_diagonal_mean": off_diag,
            "transfer_gap": diag - off_diag,
        }

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/cross_task_probing_1500.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] analysis/cross_task_probing_1500.json")


if __name__ == "__main__":
    main()
