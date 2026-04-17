"""
Direction axis alignment 시각화:
- Layer별 cos similarity (UD, LR) 변화
- Rescale Δ (scale 기여도) 변화
- 3모델 비교
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis/direction_axis_alignment.json") as f:
    data = json.load(f)

MODELS = ["Vanilla", "Baseline", "Delta"]
COLORS = {"Vanilla": "#95a5a6", "Baseline": "#3498db", "Delta": "#e74c3c"}

# Stage order for x-axis
STAGES_ORDER = [
    "vision_encoder", "after_projector",
    "vision_token_L0", "vision_token_L7", "vision_token_L14", "vision_token_L21", "vision_token_L27",
    "answer_token_L7", "answer_token_L14", "answer_token_L21", "answer_token_L27",
]
STAGE_LABELS = [
    "VE", "AP",
    "vt_L0", "vt_L7", "vt_L14", "vt_L21", "vt_L27",
    "at_L7", "at_L14", "at_L21", "at_L27",
]

def off_diag_mean(mat_list):
    m = np.array(mat_list)
    n = m.shape[0]
    off = ~np.eye(n, dtype=bool)
    return m[off].mean()


fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (1) cos_UD per stage
ax = axes[0, 0]
for model in MODELS:
    d = data[model]
    vals = []
    for s in STAGES_ORDER:
        if s in d:
            vals.append(off_diag_mean(d[s]["cosim_ud"]))
        else:
            vals.append(np.nan)
    ax.plot(range(len(STAGES_ORDER)), vals, marker="o", linewidth=2, color=COLORS[model], label=model)
ax.set_xticks(range(len(STAGES_ORDER)))
ax.set_xticklabels(STAGE_LABELS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Off-diagonal cos similarity (UD axis)", fontsize=11, fontweight="bold")
ax.set_title("Direction Axis Alignment — Up-Down axis", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="black", linestyle=":", alpha=0.3)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# (2) cos_LR per stage
ax = axes[0, 1]
for model in MODELS:
    d = data[model]
    vals = []
    for s in STAGES_ORDER:
        if s in d:
            vals.append(off_diag_mean(d[s]["cosim_lr"]))
        else:
            vals.append(np.nan)
    ax.plot(range(len(STAGES_ORDER)), vals, marker="o", linewidth=2, color=COLORS[model], label=model)
ax.set_xticks(range(len(STAGES_ORDER)))
ax.set_xticklabels(STAGE_LABELS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Off-diagonal cos similarity (LR axis)", fontsize=11, fontweight="bold")
ax.set_title("Direction Axis Alignment — Left-Right axis", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="black", linestyle=":", alpha=0.3)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# (3) Cross-task acc (off-diag) original vs rescale
ax = axes[1, 0]
for model in MODELS:
    d = data[model]
    orig_vals, rescale_vals = [], []
    for s in STAGES_ORDER:
        if s in d:
            orig_vals.append(off_diag_mean(d[s]["cross_task_orig"]))
            rescale_vals.append(off_diag_mean(d[s]["cross_task_rescale"]))
        else:
            orig_vals.append(np.nan); rescale_vals.append(np.nan)
    ax.plot(range(len(STAGES_ORDER)), orig_vals, marker="o", linewidth=2,
            color=COLORS[model], label=f"{model} (orig)")
    ax.plot(range(len(STAGES_ORDER)), rescale_vals, marker="s", linewidth=2,
            color=COLORS[model], linestyle="--", alpha=0.6, label=f"{model} (rescale)")
ax.set_xticks(range(len(STAGES_ORDER)))
ax.set_xticklabels(STAGE_LABELS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Off-diagonal accuracy (%)", fontsize=11, fontweight="bold")
ax.set_title("Cross-task Transfer: Original vs Target-rescaled", fontsize=12, fontweight="bold")
ax.set_ylim(20, 100)
ax.axhline(25, color="black", linestyle=":", alpha=0.3)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)

# (4) Rescale delta (scale contribution)
ax = axes[1, 1]
for model in MODELS:
    d = data[model]
    deltas = []
    for s in STAGES_ORDER:
        if s in d:
            deltas.append(off_diag_mean(d[s]["cross_task_rescale"]) - off_diag_mean(d[s]["cross_task_orig"]))
        else:
            deltas.append(np.nan)
    ax.plot(range(len(STAGES_ORDER)), deltas, marker="o", linewidth=2, color=COLORS[model], label=model)
ax.set_xticks(range(len(STAGES_ORDER)))
ax.set_xticklabels(STAGE_LABELS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Rescale Δ (%p)", fontsize=11, fontweight="bold")
ax.set_title("Scale Contribution to Cross-task Gap\n(Higher Δ = scale was more of an issue)",
             fontsize=12, fontweight="bold")
ax.axhline(0, color="black", linestyle="-", alpha=0.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

fig.suptitle(
    "Direction Axis Alignment Analysis — Cross-task probe weight cos similarity & rescale effect",
    fontsize=13, fontweight="bold", y=1.00
)
plt.tight_layout()
for ext in [".png", ".pdf"]:
    plt.savefig(f"analysis/direction_axis_alignment{ext}", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] analysis/direction_axis_alignment.png / .pdf")
