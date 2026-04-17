"""Exp 1 analysis — aggregate counterfactual steering results."""
import glob, json, os
import numpy as np

ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/steering_results"
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
MODELS = ["vanilla", "baseline"]

by_model = {m: {} for m in MODELS}
for f in sorted(glob.glob(os.path.join(ROOT, "*.json"))):
    r = json.load(open(f))
    # skip cross-delta files for main table
    if r.get("delta_source_task") and r["delta_source_task"] != r["task"]:
        continue
    by_model[r["model"]][r["task"]] = r

print("=" * 100)
print("  Counterfactual Steering (Kang Alg.2) — Flip to opposite direction")
print("  orig_acc: no-steering baseline | flip_rate: steered answer = opposite dir letter")
print("=" * 100)
for model in MODELS:
    print(f"\n  {model}:")
    print(f"  {'task':>14s}  {'orig_acc':>9s}  {'flip→target':>12s}  {'unchanged':>10s}  {'→other':>9s}")
    for task in TASKS:
        r = by_model[model].get(task)
        if r is None:
            print(f"  {task:>14s}  (missing)")
            continue
        print(f"  {task:>14s}  {r['orig_acc']:>8.1f}%  {r['flip_to_target_rate']:>11.1f}%  "
              f"{r['flip_unchanged_rate']:>9.1f}%  {r['flip_to_other_rate']:>8.1f}%")

print("\n" + "=" * 100)
print("  Per-direction flip rate (baseline)")
print("=" * 100)
for task in TASKS:
    r = by_model["baseline"].get(task)
    if r is None: continue
    print(f"\n  [{task}]")
    for d, stats in r["per_direction"].items():
        n = stats["n"]
        if n == 0: continue
        flip = stats["flip_to_target"] / n * 100
        unch = stats["flip_unchanged"] / n * 100
        othr = stats["flip_to_other"] / n * 100
        orig = stats["orig_correct"] / n * 100
        print(f"    {d:>6s}  n={n}  orig={orig:5.1f}%  flip→target={flip:5.1f}%  unchanged={unch:5.1f}%  other={othr:5.1f}%")

# Save summary
with open(os.path.join(ROOT, "_summary.json"), "w") as f:
    out = {}
    for model in MODELS:
        out[model] = {}
        for task in TASKS:
            r = by_model[model].get(task)
            if r is None: continue
            out[model][task] = {
                "orig_acc": r["orig_acc"],
                "flip_rate": r["flip_to_target_rate"],
                "unchanged": r["flip_unchanged_rate"],
                "other": r["flip_to_other_rate"],
                "per_dir": r["per_direction"],
            }
    json.dump(out, f, indent=2)
print(f"\n[SAVED] {os.path.join(ROOT, '_summary.json')}")
