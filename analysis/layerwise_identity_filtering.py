"""
Layer-by-layer Identity Filtering 분석.

핵심 질문: "어떤 layer에서, 어떤 mechanism으로 identity가 사라지고 direction이 출현하는가?"

실험:
  Exp1. Layer-wise Direction vs Identity Probe (Answer Token)
    - 매 layer에서 direction probe + identity probe
    - "identity가 급감하는 layer" = filtering layer 특정

  Exp2. Layer-wise Fisher Discriminant Ratio (Answer Token)
    - 매 layer에서 direction-selective vs identity-selective dimension 비율
    - "어느 layer에서 dimension-level 전환이 일어나는지"

  Exp3. Direction-selective Dimension 추적
    - Last layer에서 direction-selective top-100 dimensions 특정
    - 이 dimensions이 각 layer에서 어떤 값을 갖는지 추적
    - "direction feature가 어느 layer에서 활성화되는지"

  Exp4. Identity-selective Dimension 추적
    - Last layer에서 identity-selective top-100 dimensions 특정
    - 이 dimensions이 각 layer에서 어떻게 억제되는지

  Exp5. Dimension Cluster Analysis
    - Direction-selective dimensions이 특정 head output에 집중되는지
    - Qwen2-7B: 28 heads × 128 dim = 3584 (W_o 거치지만 상관 분석 가능)

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/layerwise_identity_filtering.py \
        --model llava-video-7b_lora_4combo_v2_baseline --task shape_color
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
}

IDENTITY_ATTRS = {
    "shape_color": "shape",
    "obj_color": "obj_class",
    "shape_place": "place_class",
    "obj_place": "obj_class",
}


def gpu_probe(X, y, num_classes, seed=42, epochs=50, lr=1e-3, wd=1e-2, bs=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    mean = X_t.mean(0); X_t -= mean
    std = X_t.std(0); std[std < 1e-8] = 1.0; X_t /= std
    n_test = max(1, int(N * 0.3))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    Xtr, ytr = X_t[perm[:-n_test]], y_t[perm[:-n_test]]
    Xte, yte = X_t[perm[-n_test:]], y_t[perm[-n_test:]]
    del X_t, y_t
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            b = idx[i:i+bs]
            loss = crit(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
    del model, opt, Xtr, Xte, ytr, yte; torch.cuda.empty_cache()
    return acc


def compute_fdr_gpu(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y).long().to(device)
    classes = torch.unique(y_t)
    grand_mean = X_t.mean(0)
    between = torch.zeros(X_t.shape[1], device=device)
    within = torch.zeros(X_t.shape[1], device=device)
    for c in classes:
        Xc = X_t[y_t == c]
        cm = Xc.mean(0)
        between += Xc.shape[0] * (cm - grand_mean) ** 2
        within += ((Xc - cm) ** 2).sum(0)
    within = within.clamp(min=1e-10)
    fdr = (between / within).cpu().numpy()
    del X_t, y_t; torch.cuda.empty_cache()
    return fdr


def load_metadata(task):
    with open(os.path.join(META_ROOT, f"{task}_metadata.json")) as f:
        return json.load(f)


def get_labels(metadata, qids, attr):
    meta_by_id = {m['id']: m for m in metadata}
    le = LabelEncoder()
    raw = [str(meta_by_id[int(str(q).split('_')[0])][attr]) for q in qids]
    return le.fit_transform(raw), len(le.classes_)


def load_answer_layer(feat_root, task, layer_idx):
    d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
    feat = np.array(np.load(os.path.join(d, f"features_layer_{layer_idx}.npy"), mmap_mode='r'))
    qids = np.load(os.path.join(d, "qids.npy"))
    meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
    return feat, qids, meta


# ============================================================
#  Exp1: Layer-wise Direction vs Identity Probe
# ============================================================

def exp1_layerwise_probe(feat_root, task):
    print(f"\n{'='*60}")
    print(f"  Exp1: Layer-wise Probe — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    id_attr = IDENTITY_ATTRS[task]
    _, _, meta = load_answer_layer(feat_root, task, 0)
    num_layers = meta["num_layers"]

    results = {"direction": [], "identity": [], "layers": []}

    for l in range(num_layers):
        feat, qids, _ = load_answer_layer(feat_root, task, l)
        feat = feat.astype(np.float32)
        dir_labels, dir_nc = get_labels(metadata, qids, "direction")
        id_labels, id_nc = get_labels(metadata, qids, id_attr)

        dir_acc = gpu_probe(feat.copy(), dir_labels, dir_nc)
        id_acc = gpu_probe(feat.copy(), id_labels, id_nc)

        results["direction"].append(dir_acc)
        results["identity"].append(id_acc)
        results["layers"].append(l)
        print(f"  Layer {l:2d}: direction={dir_acc:5.1f}%  {id_attr}={id_acc:5.1f}%")

    return results


# ============================================================
#  Exp2: Layer-wise FDR
# ============================================================

def exp2_layerwise_fdr(feat_root, task):
    print(f"\n{'='*60}")
    print(f"  Exp2: Layer-wise FDR — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    id_attr = IDENTITY_ATTRS[task]
    _, _, meta = load_answer_layer(feat_root, task, 0)
    num_layers = meta["num_layers"]

    results = {"dir_selective_pct": [], "fdr_ratio": [], "mean_fdr_dir": [], "mean_fdr_id": [], "layers": []}

    for l in range(num_layers):
        feat, qids, _ = load_answer_layer(feat_root, task, l)
        feat = feat.astype(np.float32)
        dir_labels, _ = get_labels(metadata, qids, "direction")
        id_labels, _ = get_labels(metadata, qids, id_attr)

        fdr_dir = compute_fdr_gpu(feat, dir_labels)
        fdr_id = compute_fdr_gpu(feat, id_labels)

        dir_sel = (fdr_dir > fdr_id).sum() / len(fdr_dir) * 100
        ratio = fdr_dir.mean() / max(fdr_id.mean(), 1e-10)

        results["dir_selective_pct"].append(float(dir_sel))
        results["fdr_ratio"].append(float(ratio))
        results["mean_fdr_dir"].append(float(fdr_dir.mean()))
        results["mean_fdr_id"].append(float(fdr_id.mean()))
        results["layers"].append(l)
        print(f"  Layer {l:2d}: dir-selective={dir_sel:5.1f}%  FDR ratio={ratio:.2f}x")

    return results


# ============================================================
#  Exp3 & 4: Direction/Identity Dimension Tracking
# ============================================================

def exp3_dimension_tracking(feat_root, task, top_k=100):
    """Last layer의 direction-selective/identity-selective top-k dims을 찾고, 각 layer에서 추적."""
    print(f"\n{'='*60}")
    print(f"  Exp3: Dimension Tracking — {task} (top-{top_k})")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    id_attr = IDENTITY_ATTRS[task]
    _, _, meta = load_answer_layer(feat_root, task, 0)
    num_layers = meta["num_layers"]
    last_layer = num_layers - 1

    # Last layer에서 direction-selective / identity-selective dims 특정
    feat_last, qids, _ = load_answer_layer(feat_root, task, last_layer)
    feat_last = feat_last.astype(np.float32)
    dir_labels, _ = get_labels(metadata, qids, "direction")
    id_labels, _ = get_labels(metadata, qids, id_attr)

    fdr_dir = compute_fdr_gpu(feat_last, dir_labels)
    fdr_id = compute_fdr_gpu(feat_last, id_labels)

    # Top-k direction-selective dims (highest fdr_dir / fdr_id ratio)
    dir_ratio = fdr_dir / np.maximum(fdr_id, 1e-10)
    dir_top_dims = np.argsort(dir_ratio)[-top_k:]

    # Top-k identity-selective dims (highest fdr_id / fdr_dir ratio)
    id_ratio = fdr_id / np.maximum(fdr_dir, 1e-10)
    id_top_dims = np.argsort(id_ratio)[-top_k:]

    print(f"  Direction top-{top_k} dims: mean FDR ratio = {dir_ratio[dir_top_dims].mean():.2f}")
    print(f"  Identity top-{top_k} dims: mean FDR ratio = {id_ratio[id_top_dims].mean():.2f}")

    # 각 layer에서 이 dims의 FDR 추적
    results = {
        "dir_top_dims": dir_top_dims.tolist(),
        "id_top_dims": id_top_dims.tolist(),
        "layers": [],
        "dir_dims_fdr_dir": [],  # direction-selective dims의 direction FDR (layer별)
        "dir_dims_fdr_id": [],   # direction-selective dims의 identity FDR
        "id_dims_fdr_dir": [],   # identity-selective dims의 direction FDR
        "id_dims_fdr_id": [],    # identity-selective dims의 identity FDR
        "dir_dims_probe_dir": [],  # direction-selective dims만으로 direction probe
        "id_dims_probe_dir": [],   # identity-selective dims만으로 direction probe
    }

    for l in range(num_layers):
        feat, qids, _ = load_answer_layer(feat_root, task, l)
        feat = feat.astype(np.float32)
        dir_labels, dir_nc = get_labels(metadata, qids, "direction")
        id_labels, _ = get_labels(metadata, qids, id_attr)

        fdr_d = compute_fdr_gpu(feat, dir_labels)
        fdr_i = compute_fdr_gpu(feat, id_labels)

        results["layers"].append(l)
        results["dir_dims_fdr_dir"].append(float(fdr_d[dir_top_dims].mean()))
        results["dir_dims_fdr_id"].append(float(fdr_i[dir_top_dims].mean()))
        results["id_dims_fdr_dir"].append(float(fdr_d[id_top_dims].mean()))
        results["id_dims_fdr_id"].append(float(fdr_i[id_top_dims].mean()))

        # Probe with only top-k dims
        dir_acc = gpu_probe(feat[:, dir_top_dims].copy(), dir_labels, dir_nc)
        id_acc = gpu_probe(feat[:, id_top_dims].copy(), dir_labels, dir_nc)
        results["dir_dims_probe_dir"].append(dir_acc)
        results["id_dims_probe_dir"].append(id_acc)

        print(f"  Layer {l:2d}: dir-dims→dir={dir_acc:5.1f}% | id-dims→dir={id_acc:5.1f}% | "
              f"dir-FDR={fdr_d[dir_top_dims].mean():.3f} | id-FDR={fdr_i[dir_top_dims].mean():.3f}")

    return results


# ============================================================
#  Exp5: Dimension-Head Correspondence
# ============================================================

def exp5_head_analysis(feat_root, task, top_k=100):
    """Direction-selective dims가 특정 attention head output에 집중되는지."""
    print(f"\n{'='*60}")
    print(f"  Exp5: Head-Dimension Correspondence — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    id_attr = IDENTITY_ATTRS[task]
    _, _, meta = load_answer_layer(feat_root, task, 0)
    last_layer = meta["num_layers"] - 1
    hidden_dim = 3584
    n_heads = 28
    head_dim = hidden_dim // n_heads  # 128

    feat, qids, _ = load_answer_layer(feat_root, task, last_layer)
    feat = feat.astype(np.float32)
    dir_labels, _ = get_labels(metadata, qids, "direction")
    id_labels, _ = get_labels(metadata, qids, id_attr)

    fdr_dir = compute_fdr_gpu(feat, dir_labels)
    fdr_id = compute_fdr_gpu(feat, id_labels)

    dir_ratio = fdr_dir / np.maximum(fdr_id, 1e-10)
    dir_top_dims = np.argsort(dir_ratio)[-top_k:]

    # 어떤 head segment에 direction-selective dims이 집중되는지
    # dim d → head d // head_dim (approximate, W_o 거치면 섞이지만 참고용)
    head_counts = np.zeros(n_heads)
    for d in dir_top_dims:
        head_counts[d // head_dim] += 1

    # Per-head segment FDR
    head_dir_fdr = []
    head_id_fdr = []
    for h in range(n_heads):
        dims = list(range(h * head_dim, (h + 1) * head_dim))
        head_dir_fdr.append(float(fdr_dir[dims].mean()))
        head_id_fdr.append(float(fdr_id[dims].mean()))

    results = {
        "dir_top_dims_per_head": head_counts.tolist(),
        "head_dir_fdr": head_dir_fdr,
        "head_id_fdr": head_id_fdr,
    }

    print(f"  Direction top-{top_k} dims distribution across heads:")
    for h in range(n_heads):
        bar = "█" * int(head_counts[h])
        ratio = head_dir_fdr[h] / max(head_id_fdr[h], 1e-10)
        print(f"    Head {h:2d}: {head_counts[h]:3.0f} dims {bar:<10} FDR ratio={ratio:.2f}x")

    return results


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="analysis/filtering_results")
    args = parser.parse_args()

    feat_root = FEAT_ROOTS.get(args.model)
    if not feat_root or not os.path.exists(feat_root):
        print(f"[ERROR] Not found: {feat_root}")
        return

    tasks = TASKS if args.task == "all" else [args.task]
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  {args.model} / {task}")
        print(f"{'#'*60}")

        task_results = {}
        task_results["exp1_probe"] = exp1_layerwise_probe(feat_root, task)
        task_results["exp2_fdr"] = exp2_layerwise_fdr(feat_root, task)
        task_results["exp3_dim_tracking"] = exp3_dimension_tracking(feat_root, task)
        task_results["exp5_head"] = exp5_head_analysis(feat_root, task)
        all_results[task] = task_results

    save_path = os.path.join(args.output_dir, f"filtering_{args.model}.json")
    def convert(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        return o
    with open(save_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {save_path}")


if __name__ == "__main__":
    main()
