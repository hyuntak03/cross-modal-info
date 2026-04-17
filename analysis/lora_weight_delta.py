"""
LoRA Weight Delta Analysis: 어떤 layer/projection이 가장 많이 바뀌었는지 분석.

모델 2개 로드 → 각 layer의 q/k/v/o/gate/up/down_proj weight 비교.
변화가 큰 layer = direction 정보 처리에 핵심 layer.

Usage:
    CUDA_VISIBLE_DEVICES=3 python analysis/lora_weight_delta.py \
        --output_dir analysis/mechanism_results
"""

import os
import sys
import json
import argparse

import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_args_str):
    """LLaVA 모델 로드."""
    os.environ["PYTHONPATH"] = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT"
    sys.path.insert(0, "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT")
    from core.model_loader import parse_model_args, load_model_from_args
    model_args_dict = parse_model_args(model_args_str)
    _, model, _, _, model_name, _ = load_model_from_args(model_args_dict)
    model.eval()
    return model, model_name


def compute_weight_delta(model_vanilla, model_finetuned):
    """Layer별, projection별 weight 변화량 계산 (GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_v = model_vanilla.model.layers
    layers_f = model_finetuned.model.layers
    n_layers = len(layers_v)

    projections = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    results = []

    for l in range(n_layers):
        layer_result = {"layer": l}
        for proj_name in projections:
            parts = proj_name.split(".")
            w_v = getattr(getattr(layers_v[l], parts[0]), parts[1]).weight.to(device).float()
            w_f = getattr(getattr(layers_f[l], parts[0]), parts[1]).weight.to(device).float()

            # Frobenius norm of delta / original
            delta_norm = (w_f - w_v).norm().item()
            orig_norm = w_v.norm().item()
            rel_change = delta_norm / max(orig_norm, 1e-8)

            short_name = proj_name.split(".")[-1]
            layer_result[f"{short_name}_delta"] = delta_norm
            layer_result[f"{short_name}_rel"] = rel_change

            del w_v, w_f

        results.append(layer_result)
        print(f"  Layer {l:2d}: " + " | ".join(
            f"{p.split('.')[-1]}={layer_result[p.split('.')[-1]+'_rel']:.4f}"
            for p in projections[:4]))  # show attention projs

    torch.cuda.empty_cache()
    return results


def plot_weight_delta(results, output_dir):
    """Layer별 weight 변화량 heatmap."""
    sns.set_theme(style="whitegrid", context="notebook")

    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    n_layers = len(results)

    # Build matrix
    matrix = np.zeros((n_layers, len(proj_names)))
    for i, r in enumerate(results):
        for j, p in enumerate(proj_names):
            matrix[i, j] = r[f"{p}_rel"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Projection", fontsize=12)
    ax.set_yticks(range(len(proj_names)))
    ax.set_yticklabels(proj_names, fontsize=10)
    ax.set_title("LoRA Weight Delta (Relative Frobenius Norm)\nVanilla → 4combo_v2_baseline",
                  fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="||W_ft - W_vanilla||_F / ||W_vanilla||_F")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "lora_weight_delta.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")

    # Also plot attention vs MLP per layer
    fig, ax = plt.subplots(figsize=(12, 5))
    attn_mean = matrix[:, :4].mean(axis=1)
    mlp_mean = matrix[:, 4:].mean(axis=1)
    layers = np.arange(n_layers)
    ax.plot(layers, attn_mean, color="#3498db", linewidth=2, marker='o', markersize=3,
            label="Attention (q/k/v/o)")
    ax.plot(layers, mlp_mean, color="#e74c3c", linewidth=2, marker='s', markersize=3,
            label="MLP (gate/up/down)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Relative Delta", fontsize=12)
    ax.set_title("LoRA Weight Change: Attention vs MLP per Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path2 = os.path.join(output_dir, "lora_weight_delta_attn_vs_mlp.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.savefig(save_path2.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="analysis/mechanism_results")
    parser.add_argument("--baseline_lora", type=str,
                        default="/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/3] Loading vanilla model...")
    model_v, _ = load_model(
        "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,"
        "conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,"
        "device_map=auto,force_sample=True"
    )

    print("[2/3] Loading fine-tuned model...")
    model_f, _ = load_model(
        f"lora_pretrained={args.baseline_lora},"
        "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,"
        "conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,"
        "device_map=auto,force_sample=True"
    )

    print("[3/3] Computing weight deltas...")
    results = compute_weight_delta(model_v, model_f)

    # Save
    json_path = os.path.join(args.output_dir, "lora_weight_delta.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {json_path}")

    plot_weight_delta(results, args.output_dir)

    del model_v, model_f
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
