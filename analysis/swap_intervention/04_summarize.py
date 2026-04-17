"""Aggregate swap_results/*.json → summary table."""
import glob, json, os

ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/swap_results"
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*.json"))):
    r = json.load(open(f))
    rows.append(r)

by_model = {"vanilla": {}, "baseline": {}}
for r in rows:
    key = (r["source_task"], r["target_task"])
    by_model[r["model"]][key] = r

for model in ["vanilla", "baseline"]:
    print(f"\n{'='*110}")
    print(f"  {model.upper()}  —  no_swap | swap | Δ    (in-domain = diagonal)")
    print(f"{'='*110}")
    print(f"{'src\\tgt':>14s}", end="")
    for t in TASKS:
        print(f"  {t:>22s}", end="")
    print()
    for src in TASKS:
        print(f"{src:>14s}", end="")
        for tgt in TASKS:
            r = by_model[model].get((src, tgt))
            if r is None:
                print(f"  {'MISSING':>22s}", end="")
                continue
            ns = r["no_swap"]["acc"]
            sw = r["swap"]["acc"]
            d = sw - ns
            tag = "*" if src == tgt else " "
            print(f"  {tag}{ns:5.1f}|{sw:5.1f}|{d:+5.1f}    ", end="")
        print()

print(f"\n{'='*110}\n  AGGREGATE (mean across direction, model-level)\n{'='*110}")
for model in ["vanilla", "baseline"]:
    diag_ns, diag_sw, off_ns, off_sw = [], [], [], []
    for (src, tgt), r in by_model[model].items():
        if src == tgt:
            diag_ns.append(r["no_swap"]["acc"])
            diag_sw.append(r["swap"]["acc"])
        else:
            off_ns.append(r["no_swap"]["acc"])
            off_sw.append(r["swap"]["acc"])
    import statistics
    def m(x): return statistics.mean(x) if x else 0
    print(f"\n  {model}:")
    print(f"    In-domain  (diag, 4 cells):  no_swap={m(diag_ns):5.1f}  swap={m(diag_sw):5.1f}  Δ={m(diag_sw)-m(diag_ns):+5.1f}")
    print(f"    Cross-dom  (off, 12 cells):  no_swap={m(off_ns):5.1f}  swap={m(off_sw):5.1f}  Δ={m(off_sw)-m(off_ns):+5.1f}")

# Save JSON
agg = {}
for model in ["vanilla", "baseline"]:
    agg[model] = {}
    for src in TASKS:
        for tgt in TASKS:
            r = by_model[model].get((src, tgt))
            if r is not None:
                agg[model][f"{src}->{tgt}"] = {
                    "no_swap_acc": r["no_swap"]["acc"],
                    "swap_acc": r["swap"]["acc"],
                    "delta": r["swap"]["acc"] - r["no_swap"]["acc"],
                    "pair_type": r["pair_type"],
                }
with open(os.path.join(ROOT, "_summary.json"), "w") as f:
    json.dump(agg, f, indent=2)
print(f"\n[SAVED] {ROOT}/_summary.json")
