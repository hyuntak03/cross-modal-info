"""
Identity vs Direction Information 분석.

각 stage에서 direction과 identity 중 어떤 정보가 더 잘 encode되는지 비교.

실험:
  Exp1. Direction vs Identity Probe Accuracy
    - 같은 feature로 direction(4-class) vs identity(shape/color/obj/place) 분류
    - Stage별: vision_encoder, after_projector, vision_token(mean-pool), answer_token
    - Identity가 강한 task일수록 identity probe > direction probe → entanglement 증거

  Exp2. Fisher Discriminant Ratio (FDR)
    - 각 dimension에서 direction FDR vs identity FDR
    - Direction-selective vs identity-selective dimension 비율
    - Stage별 비교 → LLM이 어떤 stage에서 disentangle하는지

  Exp3. Variance Explained
    - PCA top components가 direction을 설명하는 비율 vs identity
    - CCA (Canonical Correlation) 로 representation-label alignment

  Exp4. Mutual Information Proxy
    - KNN-based MI estimator: I(representation; direction) vs I(representation; identity)
    - Stage별, model별 비교

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/identity_vs_direction.py \
        --model llava-video-7b --output_dir analysis/identity_results
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

# Task별 primary identity attribute
IDENTITY_ATTRS = {
    "shape_color": ["shape", "color"],
    "obj_color": ["obj_class"],
    "shape_place": ["shape", "place_class"],
    "obj_place": ["obj_class", "place_class"],
}


# ============================================================
#  GPU Probe
# ============================================================

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


# ============================================================
#  Feature & Metadata Loading
# ============================================================

def load_metadata(task):
    with open(os.path.join(META_ROOT, f"{task}_metadata.json")) as f:
        return json.load(f)


def get_labels(metadata, qids, attr):
    """qids와 metadata를 매칭하여 attribute label 생성."""
    meta_by_id = {m['id']: m for m in metadata}
    le = LabelEncoder()
    raw = []
    for qid in qids:
        sid = int(str(qid).split('_')[0])
        raw.append(str(meta_by_id[sid][attr]))
    encoded = le.fit_transform(raw)
    return encoded, len(le.classes_), le.classes_


def load_stage_features(feat_root, task, stage, layer_idx=-1):
    """stage별 feature 로드. vision_token은 mean pool."""
    task_full = TASK_FULL(task)

    if stage == "vision_encoder":
        d = os.path.join(feat_root, "vision_encoder", task_full)
        feat = np.array(np.load(os.path.join(d, "features.npy"), mmap_mode='r'))
        qids = np.load(os.path.join(d, "qids.npy")) if os.path.exists(os.path.join(d, "qids.npy")) else None
        # vision_encoder에 qids 없으면 vision_token에서 가져옴
        if qids is None:
            qids = np.load(os.path.join(feat_root, "vision_token", task_full, "qids.npy"))
        return feat, qids

    elif stage == "after_projector":
        d = os.path.join(feat_root, "after_projector", task_full)
        feat = np.array(np.load(os.path.join(d, "features.npy"), mmap_mode='r'))
        qids = np.load(os.path.join(d, "qids.npy")) if os.path.exists(os.path.join(d, "qids.npy")) else None
        if qids is None:
            qids = np.load(os.path.join(feat_root, "vision_token", task_full, "qids.npy"))
        return feat, qids

    elif stage == "vision_token_meanpool":
        d = os.path.join(feat_root, "vision_token", task_full)
        meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
        nl = meta["num_layers"]
        li = nl + layer_idx if layer_idx < 0 else layer_idx
        feat = np.array(np.load(os.path.join(d, f"features_layer_{li}.npy"), mmap_mode='r'))
        qids = np.load(os.path.join(d, "qids.npy"))
        # Mean pool: reshape → mean
        nf = meta.get("num_frames", 8)
        tpf = meta.get("tokens_per_frame_post", meta.get("tokens_per_frame", 196))
        hd = meta.get("hidden_dim", 3584)
        feat = feat.reshape(feat.shape[0], nf, tpf, hd).mean(axis=(1, 2))
        return feat, qids

    elif stage == "answer_token":
        d = os.path.join(feat_root, "answer_token", task_full)
        meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
        nl = meta["num_layers"]
        li = nl + layer_idx if layer_idx < 0 else layer_idx
        feat = np.array(np.load(os.path.join(d, f"features_layer_{li}.npy"), mmap_mode='r'))
        qids = np.load(os.path.join(d, "qids.npy"))
        return feat, qids


# ============================================================
#  Exp1: Direction vs Identity Probe
# ============================================================

def exp1_direction_vs_identity(feat_root, task, layer_idx=-1):
    """같은 feature로 direction vs identity probe 비교."""
    print(f"\n{'='*60}")
    print(f"  Exp1: Direction vs Identity Probe — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    identity_attrs = IDENTITY_ATTRS[task]
    stages = ["vision_encoder", "after_projector", "vision_token_meanpool", "answer_token"]

    results = {}
    for stage in stages:
        feat_dir = os.path.join(feat_root, stage.replace("_meanpool", "").replace("answer_token", "answer_token"),
                                TASK_FULL(task))
        # Check existence
        try:
            feat, qids = load_stage_features(feat_root, task, stage, layer_idx)
        except Exception as e:
            print(f"  [{stage}] SKIP: {e}")
            continue

        feat = feat.astype(np.float32)
        print(f"\n  [{stage}] dim={feat.shape[1]}")

        # Direction probe
        dir_labels, dir_nc, _ = get_labels(metadata, qids, "direction")
        dir_acc = gpu_probe(feat.copy(), dir_labels, dir_nc)
        results[f"{stage}_direction"] = dir_acc
        print(f"    Direction ({dir_nc}-class): {dir_acc:.1f}%")

        # Identity probes
        for attr in identity_attrs:
            id_labels, id_nc, classes = get_labels(metadata, qids, attr)
            # 클래스가 너무 많으면 (>20) top-10으로 필터
            if id_nc > 20:
                top_classes = [c for c, _ in Counter(id_labels).most_common(10)]
                mask = np.isin(id_labels, top_classes)
                le2 = LabelEncoder()
                id_labels_filtered = le2.fit_transform(id_labels[mask])
                id_nc_filtered = len(le2.classes_)
                id_acc = gpu_probe(feat[mask].copy(), id_labels_filtered, id_nc_filtered)
                print(f"    {attr} (top-10 of {id_nc}): {id_acc:.1f}%")
            else:
                id_acc = gpu_probe(feat.copy(), id_labels, id_nc)
                print(f"    {attr} ({id_nc}-class): {id_acc:.1f}%")
            results[f"{stage}_{attr}"] = id_acc

    return results


# ============================================================
#  Exp2: Fisher Discriminant Ratio per Dimension
# ============================================================

def compute_fdr(X, y):
    """Per-dimension Fisher Discriminant Ratio (GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y).long().to(device)

    classes = torch.unique(y_t)
    grand_mean = X_t.mean(0)

    between_var = torch.zeros(X_t.shape[1], device=device)
    within_var = torch.zeros(X_t.shape[1], device=device)

    for c in classes:
        mask = y_t == c
        Xc = X_t[mask]
        nc = Xc.shape[0]
        class_mean = Xc.mean(0)
        between_var += nc * (class_mean - grand_mean) ** 2
        within_var += ((Xc - class_mean) ** 2).sum(0)

    within_var = within_var.clamp(min=1e-10)
    fdr = (between_var / within_var).cpu().numpy()

    del X_t, y_t; torch.cuda.empty_cache()
    return fdr


def exp2_fisher(feat_root, task, layer_idx=-1):
    """Direction FDR vs Identity FDR per dimension."""
    print(f"\n{'='*60}")
    print(f"  Exp2: Fisher Discriminant Ratio — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    identity_attrs = IDENTITY_ATTRS[task]
    primary_identity = identity_attrs[0]

    stages = ["vision_encoder", "after_projector", "vision_token_meanpool", "answer_token"]
    results = {}

    for stage in stages:
        try:
            feat, qids = load_stage_features(feat_root, task, stage, layer_idx)
        except Exception:
            continue

        feat = feat.astype(np.float32)
        dir_labels, _, _ = get_labels(metadata, qids, "direction")
        id_labels, _, _ = get_labels(metadata, qids, primary_identity)

        fdr_dir = compute_fdr(feat, dir_labels)
        fdr_id = compute_fdr(feat, id_labels)

        # Statistics
        dir_selective = (fdr_dir > fdr_id).sum()
        id_selective = (fdr_id > fdr_dir).sum()
        total = len(fdr_dir)

        results[stage] = {
            "dir_selective_pct": float(dir_selective / total * 100),
            "id_selective_pct": float(id_selective / total * 100),
            "mean_fdr_direction": float(fdr_dir.mean()),
            "mean_fdr_identity": float(fdr_id.mean()),
            "fdr_ratio": float(fdr_dir.mean() / max(fdr_id.mean(), 1e-10)),
        }

        print(f"  [{stage}]")
        print(f"    Direction-selective dims: {dir_selective}/{total} ({dir_selective/total*100:.1f}%)")
        print(f"    Identity-selective dims:  {id_selective}/{total} ({id_selective/total*100:.1f}%)")
        print(f"    Mean FDR direction: {fdr_dir.mean():.4f}")
        print(f"    Mean FDR {primary_identity}: {fdr_id.mean():.4f}")
        print(f"    Ratio (dir/id): {fdr_dir.mean()/max(fdr_id.mean(),1e-10):.2f}x")

    return results


# ============================================================
#  Exp3: Direction vs Identity Variance in Top PCA Components
# ============================================================

def exp3_pca_variance(feat_root, task, layer_idx=-1, n_components=20):
    """PCA top components가 direction vs identity 중 뭘 더 설명하는지."""
    print(f"\n{'='*60}")
    print(f"  Exp3: PCA Variance Analysis — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    primary_identity = IDENTITY_ATTRS[task][0]

    results = {}
    for stage in ["vision_token_meanpool", "answer_token"]:
        try:
            feat, qids = load_stage_features(feat_root, task, stage, layer_idx)
        except Exception:
            continue

        feat = feat.astype(np.float32)
        dir_labels, _, _ = get_labels(metadata, qids, "direction")
        id_labels, _, _ = get_labels(metadata, qids, primary_identity)

        # GPU PCA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(feat).to(device)
        X = X - X.mean(0)
        U, S, V = torch.pca_lowrank(X, q=n_components)
        X_pca = (X @ V).cpu().numpy()  # (N, n_components)
        var_exp = (S ** 2 / (S ** 2).sum()).cpu().numpy()
        del X; torch.cuda.empty_cache()

        # Per-component: probe direction vs identity
        dir_accs = []
        id_accs = []
        for pc in range(min(n_components, 10)):
            # Single PC → 1D probe
            X_1d = X_pca[:, pc:pc+1]
            da = gpu_probe(X_1d.copy(), dir_labels, 4, epochs=100)
            ia = gpu_probe(X_1d.copy(), id_labels, len(set(id_labels)), epochs=100)
            dir_accs.append(da)
            id_accs.append(ia)

        results[stage] = {
            "var_explained": var_exp[:10].tolist(),
            "direction_acc_per_pc": dir_accs,
            "identity_acc_per_pc": id_accs,
        }

        print(f"  [{stage}]")
        for pc in range(min(5, len(dir_accs))):
            print(f"    PC{pc} (var={var_exp[pc]*100:.1f}%): direction={dir_accs[pc]:.1f}%, {primary_identity}={id_accs[pc]:.1f}%")

    return results


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-video-7b")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="analysis/identity_results")
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
        print(f"  MODEL: {args.model} / TASK: {task}")
        print(f"{'#'*60}")

        task_results = {}
        task_results["exp1"] = exp1_direction_vs_identity(feat_root, task, args.layer)
        task_results["exp2"] = exp2_fisher(feat_root, task, args.layer)
        task_results["exp3"] = exp3_pca_variance(feat_root, task, args.layer)
        all_results[task] = task_results

    # Save
    save_path = os.path.join(args.output_dir, f"identity_analysis_{args.model}.json")
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
