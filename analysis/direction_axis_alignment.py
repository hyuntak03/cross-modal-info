"""
Cross-task direction axis alignment analysis.

Method 1: 각 task에서 학습된 probe의 direction axis (Up-Down, Left-Right)
          끼리 cosine similarity 측정. 축이 같은지 vs 다른지 판정.

Method 2: Cross-task probe evaluation with/without target-feature z-norm
          축은 같지만 scale만 다른 경우를 rule-out.

Stages:
  vision_encoder, after_projector, vision_token (key layers), answer_token (key layers)
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
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"

# Direction label order (확인: ['Down', 'Left', 'Right', 'Up'])
#   Down=0, Left=1, Right=2, Up=3
DIR_IDX = {"Down": 0, "Left": 1, "Right": 2, "Up": 3}

STAGES = [
    ("vision_encoder", None),
    ("after_projector", None),
    ("vision_token", 0),
    ("vision_token", 7),
    ("vision_token", 14),
    ("vision_token", 21),
    ("vision_token", 27),
    ("answer_token", 7),
    ("answer_token", 14),
    ("answer_token", 21),
    ("answer_token", 27),
]


def load_stage_feat(feat_root, task, stage, layer=None):
    if stage in ("vision_encoder", "after_projector"):
        d = os.path.join(feat_root, stage, TASK_FULL(task))
        feat = np.load(os.path.join(d, "features.npy")).astype(np.float32)
        labels = np.load(os.path.join(d, "labels.npy"))
    else:
        d = os.path.join(feat_root, stage, TASK_FULL(task))
        feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
        labels = np.load(os.path.join(d, "labels.npy"))
    return feat, labels


def train_probe_return_weights(X, y, nc=4, seed=42, epochs=50, lr=1e-3, weight_decay=1e-2):
    """Train linear probe, return (W, b, mean, std) for weight analysis."""
    device = torch.device("cuda")
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).long().to(device)
    mean = X_t.mean(0); std = X_t.std(0); std[std < 1e-8] = 1.0
    Xn = (X_t - mean) / std

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(X.shape[1], nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xn), device=device)
        for i in range(0, len(Xn), 256):
            b = idx[i:i+256]
            loss = crit(model(Xn[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    W = model.weight.detach().cpu().numpy()  # (4, D)
    mean_np = mean.cpu().numpy()
    std_np = std.cpu().numpy()
    return W, mean_np, std_np


def extract_axes(W):
    """Extract Up-Down and Left-Right axes from probe weight."""
    v_UD = W[DIR_IDX["Up"]] - W[DIR_IDX["Down"]]
    v_LR = W[DIR_IDX["Left"]] - W[DIR_IDX["Right"]]
    v_UD /= np.linalg.norm(v_UD) + 1e-8
    v_LR /= np.linalg.norm(v_LR) + 1e-8
    return v_UD, v_LR


def cross_task_cosim(axes_by_task):
    """Return 4x4 cos similarity matrix (averaged over UD and LR)."""
    n = len(TASKS)
    mat_ud = np.zeros((n, n))
    mat_lr = np.zeros((n, n))
    for i, t1 in enumerate(TASKS):
        for j, t2 in enumerate(TASKS):
            v1_ud, v1_lr = axes_by_task[t1]
            v2_ud, v2_lr = axes_by_task[t2]
            mat_ud[i, j] = float(v1_ud @ v2_ud)
            mat_lr[i, j] = float(v1_lr @ v2_lr)
    return mat_ud, mat_lr


def cross_task_eval_rescale(probes_by_task, feat_by_task, labels_by_task):
    """
    Method 2: cross-task probe eval with/without z-norm on target features.
    Returns two matrices: without rescale, with rescale.
    """
    n = len(TASKS)
    mat_orig = np.zeros((n, n))  # normalize with SOURCE task's mean/std
    mat_rescale = np.zeros((n, n))  # normalize with TARGET task's mean/std

    device = torch.device("cuda")

    for i, train_task in enumerate(TASKS):
        W, mean_src, std_src = probes_by_task[train_task]
        W_t = torch.from_numpy(W).to(device)

        for j, test_task in enumerate(TASKS):
            X = feat_by_task[test_task]
            y = labels_by_task[test_task]

            # original (source normalization)
            Xn_src = (X - mean_src) / std_src
            Xn_src_t = torch.from_numpy(Xn_src).to(device)
            y_t = torch.from_numpy(y).long().to(device)
            with torch.no_grad():
                pred = (Xn_src_t @ W_t.T).argmax(1)
                acc_orig = (pred == y_t).float().mean().item() * 100

            # rescaled (target normalization)
            mean_tgt = X.mean(0); std_tgt = X.std(0); std_tgt[std_tgt < 1e-8] = 1.0
            Xn_tgt = (X - mean_tgt) / std_tgt
            Xn_tgt_t = torch.from_numpy(Xn_tgt).to(device)
            with torch.no_grad():
                pred = (Xn_tgt_t @ W_t.T).argmax(1)
                acc_rescale = (pred == y_t).float().mean().item() * 100

            mat_orig[i, j] = acc_orig
            mat_rescale[i, j] = acc_rescale

    return mat_orig, mat_rescale


def analyze_stage(feat_root, stage, layer=None, test_ratio=0.3, seed=42):
    """Run method 1 + method 2 for one stage."""
    # Load all 4 tasks
    feat_by_task = {}
    labels_by_task = {}
    probes_by_task = {}
    axes_by_task = {}

    for task in TASKS:
        try:
            feat, labels = load_stage_feat(feat_root, task, stage, layer)
        except FileNotFoundError:
            return None

        # Train/test split
        np.random.seed(seed)
        idx = np.random.permutation(len(feat))
        n_test = int(len(feat) * test_ratio)
        tr, te = idx[n_test:], idx[:n_test]

        W, mean, std = train_probe_return_weights(feat[tr], labels[tr])
        v_UD, v_LR = extract_axes(W)

        probes_by_task[task] = (W, mean, std)
        axes_by_task[task] = (v_UD, v_LR)
        feat_by_task[task] = feat[te]
        labels_by_task[task] = labels[te]

    mat_ud, mat_lr = cross_task_cosim(axes_by_task)
    mat_orig, mat_rescale = cross_task_eval_rescale(probes_by_task, feat_by_task, labels_by_task)

    return {
        "cosim_ud": mat_ud.tolist(),
        "cosim_lr": mat_lr.tolist(),
        "cross_task_orig": mat_orig.tolist(),
        "cross_task_rescale": mat_rescale.tolist(),
    }


def summarize_cosim(mat_ud, mat_lr):
    """Off-diagonal mean of UD and LR cosim."""
    n = len(TASKS)
    off_mask = ~np.eye(n, dtype=bool)
    ud_mean = np.array(mat_ud)[off_mask].mean()
    lr_mean = np.array(mat_lr)[off_mask].mean()
    return ud_mean, lr_mean


def main():
    results = {}
    for model_name, feat_root in FEAT_ROOTS.items():
        if not os.path.exists(feat_root):
            print(f"[SKIP] {model_name}")
            continue
        print(f"\n{'='*70}\n  {model_name}\n{'='*70}")
        model_results = {}
        for stage, layer in STAGES:
            key = stage if layer is None else f"{stage}_L{layer}"
            r = analyze_stage(feat_root, stage, layer)
            if r is None:
                print(f"  [SKIP] {key}")
                continue
            model_results[key] = r

            # Summary
            ud, lr = summarize_cosim(r["cosim_ud"], r["cosim_lr"])
            orig = np.array(r["cross_task_orig"])
            rescale = np.array(r["cross_task_rescale"])
            off_mask = ~np.eye(4, dtype=bool)
            orig_off = orig[off_mask].mean()
            rescale_off = rescale[off_mask].mean()
            diag_mean = np.diag(orig).mean()

            print(f"  {key:>20} | "
                  f"cos_UD={ud:+.3f} cos_LR={lr:+.3f} | "
                  f"diag={diag_mean:.1f}% off_orig={orig_off:.1f}% off_rescale={rescale_off:.1f}% "
                  f"(Δ={rescale_off - orig_off:+.1f})")

        results[model_name] = model_results

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/direction_axis_alignment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] analysis/direction_axis_alignment.json")


if __name__ == "__main__":
    main()
