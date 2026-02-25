import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# BASE_DIR = "/data/hyuntak/project/2026/vlm_direction/cross-modal-information-flow-in-MLLM/output/information_flow/LLaVA_NeXT_Video_7B/100/val"
BASE_DIR = "../output/information_flow/LLaVA_NeXT_Video_7B/existence/val"



folders = [
    "Image___Last",
    "Image___Question",
    "Question___Last",
    "Last___Last",
]

labels = [
    "Video → Last",
    "Video → Question",
    "Question → Last",
    "Last → Last",
]

# ── 데이터 로드 & 합치기 ──
all_dfs = []
for folder, label in zip(folders, labels):
    folder_path = os.path.join(BASE_DIR, folder)
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"[WARN] No CSV found in {folder_path}")
        continue
    for f in csv_files:
        df = pd.read_csv(f)
        df["knockout_pair"] = label
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# gt_answer만 사용 (predicted_answer 섞이면 그래프 복잡해짐)
if "trace_target" in data.columns:
    data = data[data["trace_target"] == "gt_answer"]

data = data[data["is_correct"] == True]

num_layers = int(data["layer"].max()) + 1

# ── 스타일 ──
sns.set(context="notebook",
        rc={"font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12})
sns.set_theme(style='whitegrid')

palette = sns.color_palette("Set1", n_colors=len(labels))

plt.figure(figsize=(8, 5))
ax = sns.lineplot(data=data, x="layer", y="relative diff first",
                  hue="knockout_pair",
                  hue_order=labels,
                  style="knockout_pair",
                  style_order=labels,
                  dashes=True,
                  palette=palette, linewidth=1.5)

ax.set_xlabel("Layer")
ax.set_ylabel("% change in prediction probability")
ax.set_xlim(0, num_layers - 0.5)
ax.set_ylim(-100, 50)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.7)
ax.set_title("Attention Knockout — LLaVA-NeXT-Video-7B", fontweight='bold')
plt.legend(title='Knockout pair', fontsize=11, handlelength=2.5)
plt.tight_layout()

save_path = os.path.join(BASE_DIR, "combined_single_plot.pdf")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")