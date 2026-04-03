"""
block_all_layers가 아닌 폴더 내 CSV 파일들을 읽어서
knockout 전/후 raw probability를 layer별로 plot하는 스크립트.

Usage:
    python plot_prob.py --result_dir <결과 디렉토리 경로>

Example:
    python plot_prob.py --result_dir ./output/information_flow/LLaVA_NeXT_Video_7B/direction/0216_results
"""

import os
import argparse
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plot_raw_probs(data, save_file, x="layer", layers=0, title=""):
    """
    knockout 전/후 확률값(raw probability)을 layer별로 plot.
    - base_score_first: knockout 전 (수평 점선)
    - new_score_first: knockout 후 (layer별 실선)
    trace_target이 gt_answer / predicted_answer 두 종류 있으면 각각 색 구분.
    """
    sns.set(context="notebook",
            rc={"font.size": 14,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 14.0,
                "ytick.labelsize": 14.0,
                "legend.fontsize": 10.0})
    sns.set_theme(style='whitegrid')

    has_trace_target = "trace_target" in data.columns and data["trace_target"].nunique() > 1

    trace_palette = {"gt_answer": "#2ecc71", "predicted_answer": "#e74c3c"}

    plt.figure(figsize=(4, 4))

    if has_trace_target:
        for trace_target, group in data.groupby("trace_target"):
            color = trace_palette.get(trace_target, "#333333")
            label_prefix = "GT" if trace_target == "gt_answer" else "Predicted"

            # knockout 후 확률: layer별 lineplot
            ax = sns.lineplot(data=group, x=x, y="new_score_first",
                              color=color, linewidth=1,
                              label=f"{label_prefix} (knocked)")

            # knockout 전 확률: 수평 baseline (샘플 평균)
            base_mean = group["base_score_first"].mean()
            ax.axhline(y=base_mean, color=color, linewidth=1,
                       linestyle='--', alpha=0.7,
                       label=f"{label_prefix} baseline ({base_mean:.4f})")
    else:
        color = "#2ecc71"
        ax = sns.lineplot(data=data, x=x, y="new_score_first",
                          color=color, linewidth=1,
                          label="knocked")
        base_mean = data["base_score_first"].mean()
        ax.axhline(y=base_mean, color=color, linewidth=1,
                   linestyle='--', alpha=0.7,
                   label=f"baseline ({base_mean:.4f})")

    ax.set_xlabel("layer")
    ax.set_ylabel("prediction probability")
    if layers > 0:
        ax.set_xlim(0, layers + 0.5)
    if title:
        ax.set_title(title, fontsize=10)
    plt.legend(fontsize=6, handlelength=2, handletextpad=0.3)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(save_file)
    plt.close()
    print(f"  Saved: {save_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="결과 폴더 경로 (e.g. output/information_flow/.../0216_results)")
    parser.add_argument("--layers", type=int, default=32,
                        help="모델의 총 layer 수 (x축 범위, default=32)")
    args = parser.parse_args()

    result_dir = args.result_dir
    layers = args.layers

    # block_all_layers 제외한 하위 폴더만 탐색
    subdirs = sorted([
        d for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d)) and "block_all_layers" not in d
    ])

    if not subdirs:
        print(f"[WARN] No non-block_all_layers directories found in {result_dir}")
        return

    print(f"[INFO] Found {len(subdirs)} directories (excluding block_all_layers):")
    for d in subdirs:
        print(f"  - {d}")
    print()

    for subdir in subdirs:
        subdir_path = os.path.join(result_dir, subdir)
        csv_files = [
            f for f in glob.glob(os.path.join(subdir_path, "*.csv"))
            if not os.path.basename(f).startswith("layer_accuracy_") and not os.path.basename(f).startswith("summary_")
        ]

        if not csv_files:
            print(f"[SKIP] No CSV in {subdir}")
            continue

        for csv_file in csv_files:
            print(f"[Processing] {csv_file}")
            df = pd.read_csv(csv_file)

            # layer 컬럼이 숫자인지 확인
            if "layer" not in df.columns:
                print(f"  [SKIP] No 'layer' column")
                continue

            df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
            df = df.dropna(subset=["layer"])
            df["layer"] = df["layer"].astype(int)

            base_name = os.path.splitext(csv_file)[0]
            block_desc = subdir.replace("___", "->").replace("_", " ").strip()

            # 전체
            save_all = f"{base_name}_raw_probs_all.pdf"
            generate_plot_raw_probs(df, save_all, x="layer", layers=layers,
                                    title=f"Raw Probs - All\n{block_desc}")

            # 정답만
            tmp_correct = df[df["is_correct"] == True]
            if len(tmp_correct) > 0:
                save_correct = f"{base_name}_raw_probs_correct.pdf"
                generate_plot_raw_probs(tmp_correct, save_correct, x="layer", layers=layers,
                                        title=f"Raw Probs - Correct\n{block_desc}")

            # 오답만
            tmp_incorrect = df[df["is_correct"] == False]
            if len(tmp_incorrect) > 0:
                save_incorrect = f"{base_name}_raw_probs_incorrect.pdf"
                generate_plot_raw_probs(tmp_incorrect, save_incorrect, x="layer", layers=layers,
                                        title=f"Raw Probs - Incorrect\n{block_desc}")

    print("\n[DONE] All plots generated.")


if __name__ == "__main__":
    main()