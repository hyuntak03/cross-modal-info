"""
Exp 2 analysis — Per-sample variance + alignment correlation with MCQ outcome.

Input: per_sample/{model}_{task}.npz (hiddens, directions, correct, ...)

Computations:
  1. Δ(d)[L] = h_avg(d)[L] - h_avg(all)[L]  — per-task pure direction vector
  2. For each sample i with direction d:
       proj_i = (h_i - h_avg(all)) · Δ(d) / ||Δ(d)||
       align_i = cos(h_i - h_avg(all), Δ(d))
       dev_i = ||h_i - h_avg(d)||
  3. Aggregate per (model, task, layer):
       mean/std of align, proj, dev  for direction pool
       correct vs wrong subgroups: does alignment separate them?

Output:
  variance_summary.json with tables
  variance_plot.png with in-domain vs OOD distributions
"""

import os, json
import numpy as np

PER_SAMPLE_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/per_sample"
OUT_JSON = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/variance_summary.json"
OUT_PLOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/variance_plot"

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
MODELS = ["vanilla", "baseline"]
DIRS = ["up", "right", "down", "left"]


def load(model, task):
    path = os.path.join(PER_SAMPLE_ROOT, f"{model}_{task}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def analyze_one(data):
    """Return dict[layer] of stats."""
    hiddens = data["hiddens"].astype(np.float32)      # (n, n_layers, D)
    directions = data["directions"]                    # (n,)
    correct = data["correct"]                           # (n,)
    layers = data["layers"]

    res = {}
    for li, L in enumerate(layers):
        H = hiddens[:, li, :]                           # (n, D)
        h_all = H.mean(0)                               # (D,)

        # Direction averages + deltas
        h_avg_dir = {d: H[directions == d].mean(0) for d in DIRS}
        delta = {d: h_avg_dir[d] - h_all for d in DIRS}

        # Per-sample stats
        per_dir = {d: {"align_mean": 0, "align_std": 0, "align_corr": 0, "align_wrong": 0,
                        "dev_mean": 0, "dev_std": 0, "proj_mean": 0}
                   for d in DIRS}
        n_dir = {d: 0 for d in DIRS}
        for d in DIRS:
            mask = directions == d
            H_d = H[mask]
            c_d = correct[mask]
            n_dir[d] = len(H_d)
            if len(H_d) == 0:
                continue
            # alignment = cos(h_i - h_all, Δ(d))
            centered = H_d - h_all[None, :]
            delta_d = delta[d]
            align = (centered @ delta_d) / (np.linalg.norm(centered, axis=1) * np.linalg.norm(delta_d) + 1e-9)
            # projection magnitude onto Δ(d)
            proj = centered @ (delta_d / (np.linalg.norm(delta_d) + 1e-9))
            # deviation from direction mean
            dev = np.linalg.norm(H_d - h_avg_dir[d][None, :], axis=1)

            per_dir[d]["align_mean"] = float(align.mean())
            per_dir[d]["align_std"] = float(align.std())
            per_dir[d]["align_corr"] = float(align[c_d == 1].mean()) if (c_d == 1).any() else 0.0
            per_dir[d]["align_wrong"] = float(align[c_d == 0].mean()) if (c_d == 0).any() else 0.0
            per_dir[d]["dev_mean"] = float(dev.mean())
            per_dir[d]["dev_std"] = float(dev.std())
            per_dir[d]["proj_mean"] = float(proj.mean())

        # Also overall (all directions pooled)
        align_all = []
        proj_all = []
        dev_all = []
        for d in DIRS:
            mask = directions == d
            H_d = H[mask]
            if len(H_d) == 0: continue
            centered = H_d - h_all[None, :]
            delta_d = delta[d]
            align = (centered @ delta_d) / (np.linalg.norm(centered, axis=1) * np.linalg.norm(delta_d) + 1e-9)
            proj = centered @ (delta_d / (np.linalg.norm(delta_d) + 1e-9))
            dev = np.linalg.norm(H_d - h_avg_dir[d][None, :], axis=1)
            align_all.append(align); proj_all.append(proj); dev_all.append(dev)
        align_all = np.concatenate(align_all) if align_all else np.array([])
        proj_all = np.concatenate(proj_all) if proj_all else np.array([])
        dev_all = np.concatenate(dev_all) if dev_all else np.array([])

        res[int(L)] = {
            "per_dir": per_dir,
            "overall": {
                "align_mean": float(align_all.mean()) if len(align_all) else 0,
                "align_std": float(align_all.std()) if len(align_all) else 0,
                "proj_mean": float(proj_all.mean()) if len(proj_all) else 0,
                "dev_mean": float(dev_all.mean()) if len(dev_all) else 0,
                "dev_std": float(dev_all.std()) if len(dev_all) else 0,
                "acc": float((correct == 1).mean() * 100) if len(correct) else 0,
            }
        }
    return res


def main():
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for task in TASKS:
            data = load(model, task)
            if data is None:
                print(f"[SKIP] {model} {task} — no file")
                continue
            summary[model][task] = analyze_one(data)
            o = summary[model][task][21]["overall"]
            print(f"[{model}/{task}] L21: acc={o['acc']:.1f}%  align mean={o['align_mean']:.3f} std={o['align_std']:.3f}  dev mean={o['dev_mean']:.2f}")

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] {OUT_JSON}")

    # Print side-by-side comparison
    print("\n" + "=" * 120)
    print("  Per-sample stats at L21 — (align: avg cosine, dev: deviation from direction mean)")
    print("=" * 120)
    print(f"{'':>10s}", end="")
    for task in TASKS:
        print(f"  {task:>18s}", end="")
    print()
    for model in MODELS:
        print(f"  {model:>8s}:")
        for metric_name, key in [("align_mean", "align_mean"), ("align_std", "align_std"),
                                    ("align_corr", "per_dir"), ("acc", "acc"), ("dev_mean", "dev_mean")]:
            print(f"  {metric_name:>8s}", end="")
            for task in TASKS:
                if task not in summary[model]:
                    print(f"  {'--':>18s}", end=""); continue
                L21 = summary[model][task][21]
                if key == "per_dir":
                    # avg across directions of align_corr - align_wrong
                    diffs = []
                    for d in DIRS:
                        v = L21["per_dir"][d]
                        diffs.append(v["align_corr"] - v["align_wrong"])
                    val = np.mean(diffs)
                    print(f"  {val:>18.4f}", end="")
                else:
                    v = L21["overall"].get(key, 0)
                    print(f"  {v:>18.3f}", end="")
            print()


if __name__ == "__main__":
    main()
