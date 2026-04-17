"""
전체 pipeline × 모든 layer에서 Cross-Task Direction Probing.

Stages:
  - vision_encoder
  - after_projector
  - vision_token layer 0..N
  - answer_token layer 0..N

각 stage별로 4×4 cross-task matrix (train_task → test_task) 생성.
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

FEAT_ROOTS = {
    "Vanilla": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b",
    "Baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta",
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"


def load_stage_feat(feat_root, task, stage, layer=None):
    """Load features for a given stage/layer. Returns (feat, labels)."""
    if stage in ("vision_encoder", "after_projector"):
        d = os.path.join(feat_root, stage, TASK_FULL(task))
        feat = np.load(os.path.join(d, "features.npy")).astype(np.float32)
        labels = np.load(os.path.join(d, "labels.npy"))
    elif stage == "vision_token":
        d = os.path.join(feat_root, "vision_token", TASK_FULL(task))
        feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
        labels = np.load(os.path.join(d, "labels.npy"))
    elif stage == "answer_token":
        d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
        feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
        labels = np.load(os.path.join(d, "labels.npy"))
    else:
        raise ValueError(stage)
    return feat, labels


def split_train_test(n, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    return idx[n_test:], idx[:n_test]


def train_probe(X_train, y_train, nc=4, seed=42, epochs=50, lr=1e-3, weight_decay=1e-2, batch_size=256):
    device = torch.device("cuda")
    X = torch.from_numpy(X_train).to(device)
    y = torch.from_numpy(y_train).long().to(device)
    mean = X.mean(0); std = X.std(0); std[std < 1e-8] = 1.0
    X = (X - mean) / std

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(X.shape[1], nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), device=device)
        for i in range(0, len(X), batch_size):
            b = idx[i:i+batch_size]
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


def cross_task_matrix(feat_root, stage, layer=None):
    """Return 4×4 matrix of test accuracy."""
    # Load 4 tasks
    data = {}
    for task in TASKS:
        try:
            feat, labels = load_stage_feat(feat_root, task, stage, layer)
            tr, te = split_train_test(len(feat))
            data[task] = {
                "Xtr": feat[tr], "ytr": labels[tr],
                "Xte": feat[te], "yte": labels[te],
            }
        except FileNotFoundError:
            return None

    matrix = np.zeros((len(TASKS), len(TASKS)))
    for i, train_task in enumerate(TASKS):
        probe, m, s = train_probe(data[train_task]["Xtr"], data[train_task]["ytr"])
        for j, test_task in enumerate(TASKS):
            matrix[i, j] = eval_probe(probe, m, s, data[test_task]["Xte"], data[test_task]["yte"])
        del probe
    torch.cuda.empty_cache()
    return matrix


def get_num_layers(feat_root, stage):
    """Detect how many layer files exist for vision_token/answer_token."""
    d = os.path.join(feat_root, stage, TASK_FULL(TASKS[0]))
    if not os.path.exists(d): return 0
    files = [f for f in os.listdir(d) if f.startswith("features_layer_")]
    layers = [int(f.split("_")[-1].replace(".npy", "")) for f in files]
    return max(layers) + 1 if layers else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="analysis/cross_task_probing_pipeline_1500.json")
    parser.add_argument("--models", default="all",
                        help="comma-separated model names (Vanilla,Baseline,Delta) or 'all'")
    args = parser.parse_args()

    target_models = list(FEAT_ROOTS.keys()) if args.models == "all" else args.models.split(",")

    results = {}

    for model_name in target_models:
        feat_root = FEAT_ROOTS.get(model_name)
        if feat_root is None:
            print(f"[SKIP] Unknown model: {model_name}")
            continue
        if not os.path.exists(feat_root):
            print(f"[SKIP] {model_name}: not found")
            continue

        print(f"\n{'='*70}\n  {model_name}\n{'='*70}")
        model_results = {}

        # 1. vision_encoder
        print("  [vision_encoder]")
        m = cross_task_matrix(feat_root, "vision_encoder")
        if m is not None:
            model_results["vision_encoder"] = {"matrix": m.tolist(),
                                                "diag": np.diag(m).mean(),
                                                "offdiag": (m.sum() - np.trace(m)) / 12}
            print(f"    diag={model_results['vision_encoder']['diag']:.1f}%, off={model_results['vision_encoder']['offdiag']:.1f}%")

        # 2. after_projector
        print("  [after_projector]")
        m = cross_task_matrix(feat_root, "after_projector")
        if m is not None:
            model_results["after_projector"] = {"matrix": m.tolist(),
                                                 "diag": np.diag(m).mean(),
                                                 "offdiag": (m.sum() - np.trace(m)) / 12}
            print(f"    diag={model_results['after_projector']['diag']:.1f}%, off={model_results['after_projector']['offdiag']:.1f}%")

        # 3. vision_token per layer
        vt_nl = get_num_layers(feat_root, "vision_token")
        print(f"  [vision_token] {vt_nl} layers")
        model_results["vision_token"] = {}
        for l in tqdm(range(vt_nl), desc=f"    vt layers"):
            m = cross_task_matrix(feat_root, "vision_token", l)
            if m is not None:
                model_results["vision_token"][str(l)] = {
                    "matrix": m.tolist(),
                    "diag": np.diag(m).mean(),
                    "offdiag": (m.sum() - np.trace(m)) / 12,
                }

        # 4. answer_token per layer
        at_nl = get_num_layers(feat_root, "answer_token")
        print(f"  [answer_token] {at_nl} layers")
        model_results["answer_token"] = {}
        for l in tqdm(range(at_nl), desc=f"    at layers"):
            m = cross_task_matrix(feat_root, "answer_token", l)
            if m is not None:
                model_results["answer_token"][str(l)] = {
                    "matrix": m.tolist(),
                    "diag": np.diag(m).mean(),
                    "offdiag": (m.sum() - np.trace(m)) / 12,
                }

        results[model_name] = model_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
