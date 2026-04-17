"""
Exp 0 — Eq.8 Readout Alignment (Kang §2.3 Eq.8)

Core claim (Eq.8):
  logit(A) - logit(C) ≈ (w_A - w_C)^T · h
  → direction axis in h must align with lm_head readout axis for MCQ to work

Measurements:
  h_UD = h_avg(Up) - h_avg(Down)       ← direction axis in hidden
  h_LR = h_avg(Right) - h_avg(Left)
  w_letter_UD = lm_head.weight[A] - [C]  ← canonical letter readout
  w_letter_LR = lm_head.weight[B] - [D]
  w_word_UD = lm_head.weight[Up] - [Down]  ← direction word readout
  w_word_LR = lm_head.weight[Right] - [Left]

Output: matrix cos(h, w) per (model, task, layer) for letter + word readouts.

Usage: python 05_readout_alignment.py
  (loads existing avg_hidden/*.pt and each model's lm_head once per model)
"""

import os, sys, json, gc, argparse
import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT")
os.environ["HF_HOME"] = "/data/datasets/LLaVA-Video-100K-Subset/"
os.environ["HF_DATASETS_CACHE"] = "/local_datasets/vlm_direction/"

VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

AVG_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/avg_hidden"
OUT_PATH = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/readout_alignment.json"

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
DIRS = ["up", "right", "down", "left"]
LAYERS = list(range(15, 22))


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


def token_id_single(tokenizer, cand_list):
    """Return first cand whose encoding is single-token (add_special=False)."""
    for cand in cand_list:
        tids = tokenizer.encode(cand, add_special_tokens=False)
        if len(tids) == 1:
            return tids[0]
    raise ValueError(f"No single-token form in {cand_list}")


def get_readout_axes(model, tokenizer):
    """Extract lm_head rows for letters and direction words."""
    W = model.lm_head.weight.detach().float().cpu()  # (vocab, hidden)
    # Letters (canonical: A=Up, B=Right, C=Down, D=Left)
    tid_A = token_id_single(tokenizer, ["A", " A"])
    tid_B = token_id_single(tokenizer, ["B", " B"])
    tid_C = token_id_single(tokenizer, ["C", " C"])
    tid_D = token_id_single(tokenizer, ["D", " D"])
    letter_UD = W[tid_A] - W[tid_C]    # Up-Down in letter space
    letter_LR = W[tid_B] - W[tid_D]    # Right-Left
    # Direction words
    tid_Up = token_id_single(tokenizer, ["Up", " Up"])
    tid_Dn = token_id_single(tokenizer, ["Down", " Down"])
    tid_Rt = token_id_single(tokenizer, ["Right", " Right"])
    tid_Lf = token_id_single(tokenizer, ["Left", " Left"])
    word_UD = W[tid_Up] - W[tid_Dn]
    word_LR = W[tid_Rt] - W[tid_Lf]
    return {
        "letter_UD": letter_UD, "letter_LR": letter_LR,
        "word_UD": word_UD, "word_LR": word_LR,
        "tokens": {
            "A": tid_A, "B": tid_B, "C": tid_C, "D": tid_D,
            "Up": tid_Up, "Down": tid_Dn, "Right": tid_Rt, "Left": tid_Lf,
        }
    }


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float((a @ b) / denom)


def main():
    results = {}

    for model_name in ["vanilla", "baseline"]:
        print(f"\n=== Loading {model_name} for lm_head extraction ===")
        args_str = VANILLA_ARGS if model_name == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
        tokenizer, model, _, _, _, _ = load_model(args_str)
        axes = get_readout_axes(model, tokenizer)
        del model; gc.collect(); torch.cuda.empty_cache()

        results[model_name] = {"tokens": axes["tokens"], "per_task": {}}

        for task in TASKS:
            pt = torch.load(os.path.join(AVG_ROOT, f"{model_name}_{task}.pt"),
                             map_location="cpu", weights_only=False)
            avg = pt["avg"]  # {dir: {L: tensor}}
            per_layer = {}
            for L in LAYERS:
                h_up = avg["up"][L].float()
                h_dn = avg["down"][L].float()
                h_rt = avg["right"][L].float()
                h_lf = avg["left"][L].float()

                h_UD = h_up - h_dn
                h_LR = h_rt - h_lf

                # Also cross-task axis stability info: magnitudes
                per_layer[L] = {
                    "h_UD_norm": float(h_UD.norm()),
                    "h_LR_norm": float(h_LR.norm()),
                    "cos_h_UD_letter": cosine(h_UD, axes["letter_UD"]),
                    "cos_h_LR_letter": cosine(h_LR, axes["letter_LR"]),
                    "cos_h_UD_word":   cosine(h_UD, axes["word_UD"]),
                    "cos_h_LR_word":   cosine(h_LR, axes["word_LR"]),
                }
            results[model_name]["per_task"][task] = per_layer

        # Cross-task h-axis alignment (identity-conditional axis test)
        print(f"[{model_name}] cross-task axis alignment...")
        cross = {}
        for L in LAYERS:
            cross[L] = {}
            for ta in TASKS:
                avg_a = torch.load(os.path.join(AVG_ROOT, f"{model_name}_{ta}.pt"),
                                   map_location="cpu", weights_only=False)["avg"]
                h_UD_a = (avg_a["up"][L] - avg_a["down"][L]).float()
                h_LR_a = (avg_a["right"][L] - avg_a["left"][L]).float()
                for tb in TASKS:
                    if ta >= tb:
                        continue
                    avg_b = torch.load(os.path.join(AVG_ROOT, f"{model_name}_{tb}.pt"),
                                       map_location="cpu", weights_only=False)["avg"]
                    h_UD_b = (avg_b["up"][L] - avg_b["down"][L]).float()
                    h_LR_b = (avg_b["right"][L] - avg_b["left"][L]).float()
                    cross[L][f"{ta}-{tb}"] = {
                        "cos_UD": cosine(h_UD_a, h_UD_b),
                        "cos_LR": cosine(h_LR_a, h_LR_b),
                    }
        results[model_name]["cross_task_axis"] = cross

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {OUT_PATH}")

    # Quick summary print
    print("\n" + "=" * 100)
    print(f"{'':>12s}{'layer':>6s}{'task':>14s}  {'cos_h_UD_letter':>16s}{'cos_h_UD_word':>14s}{'cos_h_LR_letter':>16s}{'cos_h_LR_word':>14s}")
    for model in ["vanilla", "baseline"]:
        for task in TASKS:
            for L in LAYERS:
                r = results[model]["per_task"][task][L]
                print(f"{model:>12s}{L:>6d}{task:>14s}  "
                      f"{r['cos_h_UD_letter']:>16.3f}"
                      f"{r['cos_h_UD_word']:>14.3f}"
                      f"{r['cos_h_LR_letter']:>16.3f}"
                      f"{r['cos_h_LR_word']:>14.3f}")
        print()


if __name__ == "__main__":
    main()
