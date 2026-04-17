"""
Combined plots for Exp 0/1/2 results.

Figure 1: Readout alignment trajectory per (model, task, layer)
Figure 2: Per-sample variance vs MCQ outcome (Exp 2)
Figure 3: Counterfactual steering flip rates (Exp 1)
"""

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention"
PER_SAMPLE = os.path.join(ROOT, "per_sample")
VAR_JSON = os.path.join(ROOT, "variance_summary.json")
READOUT_JSON = os.path.join(ROOT, "readout_alignment.json")
STEER_ROOT = os.path.join(ROOT, "steering_results")

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_LABELS = ["Shape-Color\n(in-domain)", "Obj-Color\n(mid OOD)",
               "Shape-Place\n(hard OOD)", "Obj-Place\n(hardest OOD)"]
MODELS = ["vanilla", "baseline"]
MODEL_COLORS = {"vanilla": "#888", "baseline": "#1f77b4"}
DIRS = ["up", "right", "down", "left"]

# ---------- Figure 1: Readout alignment ----------
if os.path.exists(READOUT_JSON):
    with open(READOUT_JSON) as f:
        rd = json.load(f)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("cos_h_UD_letter", "h_UD vs w[A]-w[C] (letter readout, UD)"),
        ("cos_h_LR_letter", "h_LR vs w[B]-w[D] (letter readout, LR)"),
        ("cos_h_UD_word",   "h_UD vs w[Up]-w[Down] (word readout, UD)"),
        ("cos_h_LR_word",   "h_LR vs w[Right]-w[Left] (word readout, LR)"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        for model in MODELS:
            if model not in rd: continue
            for ti, task in enumerate(TASKS):
                if task not in rd[model]["per_task"]: continue
                per = rd[model]["per_task"][task]
                layers = sorted(int(L) for L in per.keys())
                vals = [per[str(L) if str(L) in per else L][key] if (str(L) in per or L in per) else 0 for L in layers]
                # fix: keys might be strings or ints depending on loader
                vals = []
                for L in layers:
                    v = per.get(L) or per.get(str(L))
                    vals.append(v[key] if v else 0)
                ls = "-" if model == "baseline" else "--"
                ax.plot(layers, vals, ls, marker="o", markersize=4,
                        label=f"{model} {task}", alpha=0.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Exp 0: Hidden direction axis vs lm_head readout axis (Eq.8 alignment)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig1_readout_alignment.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("[saved] fig1_readout_alignment.png")

# ---------- Figure 2: per-sample variance ----------
if os.path.exists(VAR_JSON):
    with open(VAR_JSON) as f:
        vr = json.load(f)

    # Plot alignment distribution by correctness per task (baseline L21)
    fig, axes = plt.subplots(2, len(TASKS), figsize=(16, 8), sharey="row")
    # Row 0: align_corr vs align_wrong bars at L21
    for ci, task in enumerate(TASKS):
        ax = axes[0, ci]
        if "baseline" not in vr or task not in vr["baseline"]:
            ax.axis("off"); continue
        L21 = vr["baseline"][task].get("21") or vr["baseline"][task].get(21)
        if L21 is None: ax.axis("off"); continue
        d_names, corr_vals, wrong_vals = [], [], []
        for d in DIRS:
            pd = L21["per_dir"][d]
            d_names.append(d)
            corr_vals.append(pd["align_corr"])
            wrong_vals.append(pd["align_wrong"])
        x = np.arange(len(d_names))
        ax.bar(x - 0.2, corr_vals, 0.4, label="MCQ correct", color="#2ca02c")
        ax.bar(x + 0.2, wrong_vals, 0.4, label="MCQ wrong", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(d_names)
        ax.set_title(f"{task}\nL21 alignment: correct vs wrong", fontsize=10)
        if ci == 0:
            ax.set_ylabel("mean cos(h-mean, Δ(dir))")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)

    # Row 1: dev_mean across layers, per task, comparing models
    for ci, task in enumerate(TASKS):
        ax = axes[1, ci]
        for model in MODELS:
            if model not in vr or task not in vr[model]: continue
            per = vr[model][task]
            layers = sorted(int(L) for L in per.keys())
            vals = []
            for L in layers:
                v = per.get(L) or per.get(str(L))
                vals.append(v["overall"]["dev_mean"] if v else 0)
            ax.plot(layers, vals, marker="o", markersize=4, label=model,
                    color=MODEL_COLORS[model])
        ax.set_title(f"{task}\nmean ‖h - h_avg(dir)‖", fontsize=10)
        ax.set_xlabel("Layer")
        if ci == 0:
            ax.set_ylabel("mean deviation from direction prototype")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Exp 2: Per-sample alignment and variance analysis", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig2_variance_analysis.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("[saved] fig2_variance_analysis.png")

# ---------- Figure 3: steering flip rates ----------
steer_files = glob.glob(os.path.join(STEER_ROOT, "*.json"))
steer_files = [f for f in steer_files if "_summary" not in f and "_from-" not in f]
if steer_files:
    by_model = {m: {} for m in MODELS}
    for f in steer_files:
        r = json.load(open(f))
        by_model[r["model"]][r["task"]] = r

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    width = 0.35
    x = np.arange(len(TASKS))
    for i, model in enumerate(MODELS):
        vals_orig = []
        vals_flip = []
        for task in TASKS:
            r = by_model[model].get(task)
            vals_orig.append(r["orig_acc"] if r else 0)
            vals_flip.append(r["flip_to_target_rate"] if r else 0)
        off = (i - 0.5) * width
        ax.bar(x + off - width/4, vals_orig, width/2, label=f"{model} orig_acc",
               color=MODEL_COLORS[model], alpha=0.5)
        ax.bar(x + off + width/4, vals_flip, width/2, label=f"{model} flip_rate",
               color=MODEL_COLORS[model], hatch="//", edgecolor="black")
    ax.axhline(25, color="k", ls=":", alpha=0.5, label="chance (25%)")
    ax.set_xticks(x); ax.set_xticklabels(TASK_LABELS)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Exp 1: Counterfactual steering flip rate\n"
                 "(orig_acc: no steering, flip_rate: steered→opposite direction letter)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig3_steering.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("[saved] fig3_steering.png")
