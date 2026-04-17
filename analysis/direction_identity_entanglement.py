"""
Direction-Identity Entanglement 분석.

각 task에서 vision token (after projector + LLM layer별)의:
  (1) Direction probe accuracy
  (2) Identity probe accuracy (task에 따라 shape/obj_class/place_class)
  (3) Fisher top-k dims for direction vs identity → overlap 측정
  (4) Direction-Identity FDR ratio

Easy task (shape_color): direction/identity가 분리된 subspace → low overlap
Hard task (obj_place): direction/identity가 entangled → high overlap

+ Last token에서도 동일 측정 → fine-tuning이 entanglement를 어떻게 해소하는지

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/direction_identity_entanglement.py \
        --model llava-video-7b_lora_4combo_v2_baseline

    # 전체
    CUDA_VISIBLE_DEVICES=0 python analysis/direction_identity_entanglement.py --model all
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"

# 각 task에서 probing할 identity attribute
# shape_color: shape이 identity, color는 background
# obj_color: obj_class가 identity
# shape_place: shape + place_class 둘 다
# obj_place: obj_class + place_class 둘 다
IDENTITY_ATTRS = {
    "shape_color": ["shape"],
    "obj_color": ["obj_class"],
    "shape_place": ["shape", "place_class"],
    "obj_place": ["obj_class", "place_class"],
}

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
    "llava-video-7b_lora_4combo_v2_delta": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_delta",
}

META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"

ALL_MODELS = ["llava-video-7b", "llava-video-7b_lora_4combo_v2_baseline", "llava-video-7b_lora_4combo_v2_delta"]

# Probe할 layer indices
KEY_LAYERS = [0, 7, 14, 18, 21, 24, 27]


def model_short(name):
    return name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")


# ============================================================
#  GPU Probe + Fisher
# ============================================================

def gpu_probe(X, y, nc, seed=42, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    m = X_t.mean(0); X_t -= m; s = X_t.std(0); s[s < 1e-8] = 1; X_t /= s
    nt = max(1, int(N * 0.3))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    p = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    Xtr, ytr = X_t[p[:-nt]], y_t[p[:-nt]]
    Xte, yte = X_t[p[-nt:]], y_t[p[-nt:]]
    del X_t, y_t
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(D, nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 64):
            b = idx[i:i + 64]; loss = crit(model(Xtr[b]), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
    del model, opt; torch.cuda.empty_cache()
    return acc


def compute_fisher_topk(X_np, y_np, k=100, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(device, dtype=torch.float32)
    y = torch.from_numpy(y_np).long().to(device)
    gm = X.mean(0)
    var_b = torch.zeros(X.shape[1], device=device)
    var_w = torch.zeros(X.shape[1], device=device)
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() == 0: continue
        Xc = X[mask]; cm = Xc.mean(0)
        var_b += mask.sum().float() * (cm - gm) ** 2
        var_w += (Xc - cm).pow(2).sum(dim=0)
    fisher = var_b / (var_w + 1e-8)
    fdr = fisher.mean().item()
    topk_idx = fisher.argsort(descending=True)[:k].cpu().numpy()
    return topk_idx, fdr


def dims_overlap(idx_a, idx_b):
    """Two sets of Fisher top-k indices → overlap metrics."""
    sa, sb = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(sa & sb)
    k = max(len(sa), 1)
    return inter, inter / k  # count, ratio


# ============================================================
#  Feature Loading
# ============================================================

def load_metadata(task):
    with open(os.path.join(META_ROOT, f"{task}_metadata.json")) as f:
        return json.load(f)


def get_labels(metadata, qids, attr):
    mb = {m['id']: m for m in metadata}
    le = LabelEncoder()
    raw = [str(mb[int(str(q).split('_')[0])][attr]) for q in qids]
    return le.fit_transform(raw), len(le.classes_)


def load_vision_meanpooled(feat_root, task, stage, layer=None):
    """Vision token features → mean-pooled (N, D)."""
    task_full = TASK_FULL(task)

    if stage == "after_projector":
        d = os.path.join(feat_root, "after_projector", task_full)
        meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
        feat = np.array(np.load(os.path.join(d, "features.npy"), mmap_mode='r'))
        labels = np.load(os.path.join(d, "labels.npy"))
        qids = np.load(os.path.join(d, "qids.npy"))
        nf = meta.get("num_frames", 8)
        tpf = meta.get("tokens_per_frame_post", 196)
        hd = meta.get("hidden_dim", 3584)
        feat = feat.reshape(-1, nf, tpf, hd).mean(axis=(1, 2)).astype(np.float32)
        return feat, qids

    elif stage == "vision_token":
        d = os.path.join(feat_root, "vision_token", task_full)
        meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
        qids = np.load(os.path.join(d, "qids.npy"))
        nf = meta.get("num_frames", 8)
        tpf = meta.get("tokens_per_frame_post", 196)
        hd = meta.get("hidden_dim", 3584)
        feat = np.array(np.load(os.path.join(d, f"features_layer_{layer}.npy"), mmap_mode='r'))
        feat = feat.reshape(-1, nf, tpf, hd).mean(axis=(1, 2)).astype(np.float32)
        return feat, qids

    elif stage == "answer_token":
        d = os.path.join(feat_root, "answer_token", task_full)
        qids = np.load(os.path.join(d, "qids.npy"))
        feat = np.array(np.load(os.path.join(d, f"features_layer_{layer}.npy"), mmap_mode='r'))
        return feat.astype(np.float32), qids


# ============================================================
#  Main Analysis
# ============================================================

def analyze_entanglement(feat_root, model_name, output_dir):
    """모든 task에 대해 direction-identity entanglement 측정."""

    FISHER_K = 100
    results = {}

    for task in TASKS:
        print(f"\n{'='*70}")
        print(f"  {model_name} / {task}")
        print(f"{'='*70}")

        metadata = load_metadata(task)
        id_attrs = IDENTITY_ATTRS[task]

        task_results = {"stages": {}}

        # Stages to analyze
        stages = [("after_projector", None)] + \
                 [("vision_token", l) for l in KEY_LAYERS] + \
                 [("answer_token", l) for l in KEY_LAYERS]

        print(f"\n  {'Stage':>20} {'Dir probe':>10}", end="")
        for attr in id_attrs:
            print(f" {attr[:8]+' probe':>12} {attr[:8]+' overlap':>14}", end="")
        print(f" {'Dir FDR':>9}")
        print("  " + "-" * (35 + 26 * len(id_attrs) + 10))

        for stage, layer in stages:
            stage_key = f"{stage}" if layer is None else f"{stage}_layer_{layer}"

            try:
                feat, qids = load_vision_meanpooled(feat_root, task, stage, layer)
            except Exception as e:
                continue

            # Direction labels
            dir_labels, dir_nc = get_labels(metadata, qids, "direction")
            dir_topk, dir_fdr = compute_fisher_topk(feat, dir_labels, FISHER_K, dir_nc)
            dir_acc = gpu_probe(feat.copy(), dir_labels, dir_nc)

            stage_data = {
                "dir_acc": dir_acc, "dir_fdr": dir_fdr,
                "identities": {}
            }

            label = stage_key.replace("vision_token_layer_", "vt_L").replace("answer_token_layer_", "at_L").replace("after_projector", "proj")
            print(f"  {label:>20} {dir_acc:>9.1f}%", end="")

            for attr in id_attrs:
                id_labels, id_nc = get_labels(metadata, qids, attr)
                id_topk, id_fdr = compute_fisher_topk(feat, id_labels, FISHER_K, id_nc)
                id_acc = gpu_probe(feat.copy(), id_labels, id_nc)
                overlap_count, overlap_ratio = dims_overlap(dir_topk, id_topk)

                stage_data["identities"][attr] = {
                    "id_acc": id_acc, "id_fdr": id_fdr,
                    "overlap_count": int(overlap_count),
                    "overlap_ratio": overlap_ratio,
                }

                print(f" {id_acc:>11.1f}% {overlap_count:>4}/{FISHER_K} ({overlap_ratio:>5.1%})", end="")

            print(f" {dir_fdr:>9.4f}")
            task_results["stages"][stage_key] = stage_data
            del feat

        results[task] = task_results

    # Save
    os.makedirs(output_dir, exist_ok=True)
    short = model_short(model_name)
    sp = os.path.join(output_dir, f"entanglement_{short}.json")
    with open(sp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {sp}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--output_dir", default="analysis/entanglement_results")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    for model_name in models:
        feat_root = FEAT_ROOTS.get(model_name)
        if not feat_root or not os.path.exists(feat_root):
            print(f"[SKIP] {model_name}: feature root not found")
            continue
        analyze_entanglement(feat_root, model_name, args.output_dir)


if __name__ == "__main__":
    main()
