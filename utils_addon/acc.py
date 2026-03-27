import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

def compute_layer_acc(csv_path):
    df = pd.read_csv(csv_path, dtype={"question_id": str})
    
    # gt_answer row만 사용 (predicted_answer row는 중복이니까 제외)
    df_gt = df[df["trace_target"] == "gt_answer"].copy()
    
    # knocked_predicted_answer가 goden answer와 일치하는지 판별
    df_gt["knocked_correct"] = df_gt.apply(
        lambda row: str(row["knocked_predicted_answer"]).strip().lower() == str(row["goden answer"]).strip().lower(),
        axis=1
    )
    
    # 원래 정답 여부도 기록
    df_gt["origin_correct"] = df_gt["is_correct"]
    
    # layer별 accuracy
    layer_acc = df_gt.groupby("layer").agg(
        total=("knocked_correct", "count"),
        knocked_correct_count=("knocked_correct", "sum"),
        origin_correct_count=("origin_correct", "sum"),
    ).reset_index()
    
    layer_acc["knocked_acc"] = layer_acc["knocked_correct_count"] / layer_acc["total"] * 100
    layer_acc["origin_acc"] = layer_acc["origin_correct_count"] / layer_acc["total"] * 100
    
    return layer_acc


def process_folder(folder_path):
    # CSV 파일 찾기
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and not f.startswith("layer_accuracy_")]
    if not csv_files:
        print(f"  [SKIP] No CSV found in {folder_path}")
        return
    
    csv_path = os.path.join(folder_path, csv_files[0])
    folder_name = os.path.basename(folder_path)
    
    print(f"\n{'='*60}")
    print(f"  Folder: {folder_name}")
    print(f"  CSV: {csv_files[0]}")
    print(f"{'='*60}")
    
    layer_acc = compute_layer_acc(csv_path)
    
    origin_acc = layer_acc["origin_acc"].iloc[0]  # 모든 layer에서 동일
    print(f"  Original ACC (no knockout): {origin_acc:.2f}%")
    print(f"  {'Layer':>6} | {'Knocked ACC':>12} | {'Δ ACC':>8} | {'Total':>6}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")
    
    for _, row in layer_acc.iterrows():
        delta = row["knocked_acc"] - origin_acc
        print(f"  {int(row['layer']):>6} | {row['knocked_acc']:>11.2f}% | {delta:>+7.2f}% | {int(row['total']):>6}")
    
    # CSV로 저장
    out_csv = os.path.join(folder_path, f"layer_accuracy_{folder_name}.csv")
    layer_acc.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n  Saved: {out_csv}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_acc["layer"], layer_acc["knocked_acc"], 
            color="#2ecc71", linewidth=1.5, label="Knocked ACC")
    ax.axhline(y=origin_acc, color="#e74c3c", linestyle="--", 
               linewidth=1, label=f"Original ACC ({origin_acc:.1f}%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Layer-wise Accuracy after Knockout\n{folder_name}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, layer_acc["layer"].max() + 0.5)
    plt.tight_layout()
    
    out_pdf = os.path.join(folder_path, f"layer_accuracy_{folder_name}.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"  Plot: {out_pdf}")

BASE_DIR = "..output/information_flow/LLaVA_NeXT_Video_7B/shape/val"

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_layer_acc.py <base_dir>")
        print("Example: python analyze_layer_acc.py output/information_flow/LLaVA_NeXT_Video_7B/direction/val")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)
    
    # block_all_layers 폴더 제외
    folders = sorted([
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and "block_all_layers" not in f
    ])
    
    print(f"Base dir: {base_dir}")
    print(f"Found {len(folders)} folders (excluding block_all_layers):")
    for f in folders:
        print(f"  - {f}")
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        process_folder(folder_path)
    
    # 전체 요약 테이블
    print(f"\n\n{'='*80}")
    print("  SUMMARY: Layer-wise ACC across all knockout types")
    print(f"{'='*80}")
    
    all_results = {}
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.startswith("layer_accuracy_") and f.endswith(".csv")]
        if csv_files:
            df = pd.read_csv(os.path.join(folder_path, csv_files[0]), encoding="utf-8")
            all_results[folder] = df.set_index("layer")["knocked_acc"]
    
    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = os.path.join(base_dir, "summary_layer_accuracy.csv")
        summary.to_csv(summary_path)
        print(f"\n  Summary saved: {summary_path}")
        print(summary.to_string())


if __name__ == "__main__":
    main()

# python acc.py ../output/information_flow/LLaVA_NeXT_Video_7B/existence/val
# ./output/information_flow/llava_v1_6_vicuna_7b/position/val