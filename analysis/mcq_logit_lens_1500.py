"""
Last layer hidden state에 RMSNorm + lm_head 적용 → letter logit argmax → MCQ accuracy.
Model generate() 없이 즉시 계산.

GT letter는 HuggingFace dataset (R2R_4way_1500)의 'answer' 필드.
"""

import os, sys, json, argparse
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

FEAT_ROOTS = {
    "vanilla": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b",
    "baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline",
    "delta": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta",
}

LORA_PATHS = {
    "baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "delta": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2way_4way_1500_{t}"  # fixed below


def load_model_weights(model_key):
    """Load lm_head + final RMSNorm + tokenizer. Return on GPU (float32)."""
    from core.model_loader import parse_model_args, load_model_from_args

    if model_key == "vanilla":
        args_str = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"
    else:
        lp = LORA_PATHS[model_key]
        args_str = f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"

    ma = parse_model_args(args_str)
    tokenizer, model, _, _, _, _ = load_model_from_args(ma)

    device = torch.device("cuda")
    lm_head_w = model.lm_head.weight.data.float().clone().to(device)
    norm_w = model.model.norm.weight.data.float().clone().to(device)
    del model
    torch.cuda.empty_cache()
    return lm_head_w, norm_w, tokenizer


def rmsnorm(x, w, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * w


def load_dataset_candidates(task, hf_name="R2R_4way_1500"):
    """Load candidates + answer per sample from HuggingFace dataset. key = sample index."""
    from datasets import load_dataset
    cache_dir = os.environ.get("HF_HOME", None)
    ds = load_dataset("takhyun03/vlm_direction_testbed", hf_name + "_" + task,
                      cache_dir=cache_dir, split="val",
                      token=os.environ.get("HF_TOKEN"))
    samples = {}
    for idx, item in enumerate(ds):
        sid = str(item.get("id", idx))
        ans = str(item["answer"]).strip()
        cands = item.get("candidates", [])
        if isinstance(cands, str):
            import ast
            cands = ast.literal_eval(cands)
        # Stored qids are "{id}_{idx}" format
        samples[f"{sid}_{idx}"] = {"answer": ans, "candidates": cands}
    return samples


def evaluate(model_key, task, tokenizer, lm_head_w, norm_w, layer_idx):
    """Last layer hidden state → logit → letter argmax → MCQ acc."""
    feat_root = FEAT_ROOTS[model_key]
    feat_dir = os.path.join(feat_root, "answer_token", f"vlm_direction_testbed_R2R_4way_1500_{task}")

    feat = np.load(os.path.join(feat_dir, f"features_layer_{layer_idx}.npy")).astype(np.float32)
    qids = np.load(os.path.join(feat_dir, "qids.npy"))

    device = torch.device("cuda")
    X = torch.from_numpy(feat).to(device)
    h = rmsnorm(X, norm_w)  # (N, D)
    logits = h @ lm_head_w.T  # (N, vocab)

    # Letter tokens A~H
    LETTERS = "ABCDEFGH"
    letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    letter_logits = logits[:, letter_ids]  # (N, 8)

    # Load GT
    samples = load_dataset_candidates(task)

    correct_by_candidate_count = {4: [0, 0], 8: [0, 0]}
    pred_letter_arr = []
    gt_letter_arr = []

    for i, qid in enumerate(qids):
        s = samples.get(str(qid))
        if s is None:
            continue
        gt = s["answer"]
        n_cand = len(s["candidates"])
        # restrict to actual candidate letters
        active_letters = LETTERS[:n_cand]
        active_ids_in_letter = [LETTERS.index(c) for c in active_letters]
        active_logits = letter_logits[i, active_ids_in_letter]
        pred_idx = active_logits.argmax().item()
        pred = active_letters[pred_idx]

        pred_letter_arr.append(pred)
        gt_letter_arr.append(gt)
        correct_by_candidate_count[n_cand][1] += 1
        if pred == gt:
            correct_by_candidate_count[n_cand][0] += 1

    total_correct = sum(v[0] for v in correct_by_candidate_count.values())
    total = sum(v[1] for v in correct_by_candidate_count.values())
    acc = total_correct / total * 100 if total > 0 else 0

    return {
        "model": model_key, "task": task, "layer": layer_idx,
        "accuracy": acc, "correct": total_correct, "total": total,
        "by_n_cand": {k: {"correct": v[0], "total": v[1], "acc": v[0]/v[1]*100 if v[1] else 0}
                      for k, v in correct_by_candidate_count.items() if v[1] > 0}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=28,
                        help="which layer to use. 28 = after last decoder layer")
    parser.add_argument("--output", default="analysis/mcq_logit_lens_1500.json")
    args = parser.parse_args()

    all_results = {}
    for model_key in ["vanilla", "baseline", "delta"]:
        print(f"\n{'='*60}\n  Loading {model_key}...\n{'='*60}")
        lm_head_w, norm_w, tokenizer = load_model_weights(model_key)

        all_results[model_key] = {}
        for task in TASKS:
            try:
                r = evaluate(model_key, task, tokenizer, lm_head_w, norm_w, args.layer)
                all_results[model_key][task] = r
                bd = r["by_n_cand"]
                print(f"  {task}: {r['accuracy']:.1f}% ({r['correct']}/{r['total']})", end="")
                if bd:
                    extra = " | " + ", ".join(f"{k}-cand: {v['acc']:.1f}%" for k, v in bd.items())
                    print(extra)
                else:
                    print()
            except Exception as e:
                print(f"  {task}: FAILED — {e}")
                all_results[model_key][task] = {"error": str(e)}

        del lm_head_w, norm_w
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
