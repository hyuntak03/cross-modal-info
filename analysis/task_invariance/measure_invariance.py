"""
Last token direction encoding의 task-invariance 직접 측정.

Metrics per (model, layer):
  1. Prototype cos: cos(h_avg(A,d), h_avg(B,d))    — 같은 direction, 다른 task의 절대 위치
  2. Δ-axis cos:    cos(Δ(A,d), Δ(B,d))            — direction axis 자체 정렬도 (within-task mean 빼고)
  3. Within-task per-sample alignment std (axis 주변 흩어짐)
  4. Direction separation (within task): cos(Δ(d_i), Δ(d_j)) for i≠j

Features: /data3/.../linear_probing_1500/{model}/answer_token/{task}/features_layer_{L}.npy
Labels: 0=Down, 1=Left, 2=Right, 3=Up
"""
import os, json, numpy as np
from itertools import combinations

ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500"
MODELS = {
    "vanilla":  "llava-video-7b",
    "baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "delta":    "llava-video-7b_lora_4combo_v2_delta",
}
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
LAYERS = [0, 3, 7, 10, 14, 18, 21, 24, 27]
LABEL_NAMES = ["Down", "Left", "Right", "Up"]
OUT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"


def task_path(model, task):
    return f"{ROOT}/{MODELS[model]}/answer_token/vlm_direction_testbed_R2R_4way_1500_{task}"


def load(model, task, L):
    p = task_path(model, task)
    h = np.load(f"{p}/features_layer_{L}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{p}/labels.npy")
    return h, y


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cos_mat(A, B):
    """Row-wise cosine: returns (n,) where result[i] = cos(A[i], B[i])."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(axis=1)


def per_dir_stats(h, y):
    """Returns: prototype (4,D), grand (D,), Delta (4,D), per_sample_align (n,)."""
    grand = h.mean(0)
    proto = np.stack([h[y == d].mean(0) for d in range(4)])
    Delta = proto - grand[None, :]
    Delta_hat = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)
    centered = h - grand[None, :]
    centered_hat = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)
    align = (centered_hat * Delta_hat[y]).sum(axis=1)
    return proto, grand, Delta, align


def analyze(model, L):
    res = {}
    cache = {}
    for task in TASKS:
        h, y = load(model, task, L)
        proto, grand, Delta, align = per_dir_stats(h, y)
        cache[task] = dict(proto=proto, grand=grand, Delta=Delta, align=align,
                            n=len(y), y=y)
        res[task] = {
            "n": int(len(y)),
            "align_mean": float(align.mean()),
            "align_std":  float(align.std()),
            "delta_norm_mean": float(np.linalg.norm(Delta, axis=1).mean()),
            "grand_norm": float(np.linalg.norm(grand)),
            "within_task_dir_cos": {},
        }
        for i, j in combinations(range(4), 2):
            res[task]["within_task_dir_cos"][f"{LABEL_NAMES[i]}-{LABEL_NAMES[j]}"] = \
                cos(Delta[i], Delta[j])

    cross = {"prototype_cos": {}, "delta_cos": {}}
    for tA, tB in combinations(TASKS, 2):
        proto_cos = [cos(cache[tA]["proto"][d], cache[tB]["proto"][d]) for d in range(4)]
        delta_cos = [cos(cache[tA]["Delta"][d], cache[tB]["Delta"][d]) for d in range(4)]
        cross["prototype_cos"][f"{tA}__{tB}"] = {
            "per_dir": {LABEL_NAMES[d]: proto_cos[d] for d in range(4)},
            "mean": float(np.mean(proto_cos)),
        }
        cross["delta_cos"][f"{tA}__{tB}"] = {
            "per_dir": {LABEL_NAMES[d]: delta_cos[d] for d in range(4)},
            "mean": float(np.mean(delta_cos)),
        }
    res["_cross_task"] = cross
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    full = {}
    for model in MODELS:
        full[model] = {}
        for L in LAYERS:
            print(f"[{model} L{L}] ...", flush=True)
            full[model][f"L{L}"] = analyze(model, L)

    out_path = f"{OUT}/invariance_metrics.json"
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"[SAVED] {out_path}")

    # Compact summary table
    print("\n" + "=" * 100)
    print("CROSS-TASK Δ-axis cosine (mean over 4 directions, mean over 6 task-pairs)")
    print("=" * 100)
    print(f"{'Layer':>6} | {'Vanilla':>10} | {'Baseline':>10} | {'Delta':>10}")
    for L in LAYERS:
        row = f"  L{L:<3} |"
        for model in ["vanilla", "baseline", "delta"]:
            d = full[model][f"L{L}"]["_cross_task"]["delta_cos"]
            avg = float(np.mean([v["mean"] for v in d.values()]))
            row += f" {avg:>9.3f}  |"
        print(row)

    print("\n" + "=" * 100)
    print("WITHIN-TASK per-sample alignment STD (mean over 4 tasks)")
    print("=" * 100)
    print(f"{'Layer':>6} | {'Vanilla':>10} | {'Baseline':>10} | {'Delta':>10}")
    for L in LAYERS:
        row = f"  L{L:<3} |"
        for model in ["vanilla", "baseline", "delta"]:
            stds = [full[model][f"L{L}"][t]["align_std"] for t in TASKS]
            row += f" {np.mean(stds):>9.3f}  |"
        print(row)

    print("\n" + "=" * 100)
    print("CROSS-TASK prototype cosine (h_avg(d) absolute position)")
    print("=" * 100)
    print(f"{'Layer':>6} | {'Vanilla':>10} | {'Baseline':>10} | {'Delta':>10}")
    for L in LAYERS:
        row = f"  L{L:<3} |"
        for model in ["vanilla", "baseline", "delta"]:
            d = full[model][f"L{L}"]["_cross_task"]["prototype_cos"]
            avg = float(np.mean([v["mean"] for v in d.values()]))
            row += f" {avg:>9.3f}  |"
        print(row)


if __name__ == "__main__":
    main()
