import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ============ 여기만 수정하면 됨 ============
BASE_DIR = "./output/information_flow/LLaVA_NeXT_Video_7B/100/val"
OUTPUT_PDF = "combined_correct_only.pdf"
# ==========================================

sns.set(context="notebook",
        rc={"font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 10.0})
sns.set_theme(style='whitegrid')
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[5:6] + palette_[7:] + palette_[0:2] + palette_[6:7]

# block_all_layers 제외
folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f)) and "block_all_layers" not in f
])

print(f"Found {len(folders)} folders:")
for f in folders:
    print(f"  - {f}")

# 모든 폴더의 정답 데이터를 하나의 DataFrame으로 합침
all_data = []
for folder in folders:
    folder_path = os.path.join(BASE_DIR, folder)
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv") and not f.startswith("layer_accuracy_") and not f.startswith("summary_")
    ]
    if not csv_files:
        print(f"  [SKIP] {folder}")
        continue

    csv_path = os.path.join(folder_path, csv_files[0])
    df = pd.read_csv(csv_path, dtype={"question_id": str}, encoding="utf-8")

    # gt_answer row만 + 정답만
    df_gt = df[df["trace_target"] == "gt_answer"].copy()
    df_correct = df_gt[
        df_gt.apply(
            lambda row: str(row["origin_predicted_answer"]).strip().lower() == str(row["goden answer"]).strip().lower(),
            axis=1
        )
    ]

    if len(df_correct) == 0:
        print(f"  [SKIP] {folder} — no correct samples")
        continue

    # block_desc를 보기 좋게 변환
    label = folder.replace("___", "→").replace(",", " + ")
    df_correct["block_type"] = label
    all_data.append(df_correct)

if not all_data:
    print("No correct samples found!")
    exit()

combined = pd.concat(all_data, ignore_index=True)

# plot
plt.figure(figsize=(6, 4))
ax = sns.lineplot(
    data=combined,
    x="layer",
    y="relative diff first",
    hue="block_type",
    style="block_type",
    dashes=True,
    palette=palette,
    linewidth=1,
)
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("% change in prediction probability")
ax.set_xlim(0, combined["layer"].max() + 0.5)
plt.legend(title="blocked positions", fontsize=7, handlelength=2, handletextpad=0.5)
plt.subplots_adjust(left=0.2, bottom=0.2)

out_path = os.path.join(BASE_DIR, OUTPUT_PDF)
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved: {out_path}")