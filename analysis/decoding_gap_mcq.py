"""
Decoding Gap 분석 (MCQ A/B/C/D 기준).

각 sample의 실제 MCQ candidate를 반영하여 logit lens 수행.
"A"가 Up인 sample, "B"가 Up인 sample이 각각 다르므로 sample별로 처리.

실험:
  Exp1. MCQ Logit Lens — 정답 option letter의 logit이 4개 중 top-1인지
  Exp2. Linear Probe Acc vs Logit Lens Acc per layer
  Exp3. Confusion Matrix (MCQ 기준)

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/decoding_gap_mcq.py \
        --model llava-video-7b_lora_4combo_v2_baseline
"""

import os
import sys
import json
import argparse
import importlib.util

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
}

LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline":
        "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "llava-video-7b_lora_syn_v4_baseline":
        "/data/jong980812/project/LLaVA-NeXT/work_dirs/LLaVA-Video-7B-Qwen2/llava-video-7b-qwen2_lora-r64_motion_v4_100K_baseline_rope1d_f8_ep1_lr1e-5",
}

DIRECTIONS = ["Up", "Down", "Left", "Right"]
LETTERS = ["A", "B", "C", "D"]


MCQ_JSON_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way"

def load_dataset_questions(task):
    """로컬 JSON에서 question + candidates 로드 (HF 의존 제거)."""
    json_path = os.path.join(MCQ_JSON_ROOT, f"{task}.json")
    with open(json_path) as f:
        data = json.load(f)
    # qid 형식 맞추기: "0_0", "1_1" 등
    questions = []
    for item in data:
        item['q_id'] = f"{item['id']}_{item['id']}"
        questions.append(item)
    return questions


def load_answer_layer(feat_root, task, layer_idx):
    d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
    feat = np.array(np.load(os.path.join(d, f"features_layer_{layer_idx}.npy"), mmap_mode='r'))
    qids = np.load(os.path.join(d, "qids.npy"))
    meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
    return feat, qids, meta


def load_model_weights(model_name):
    """lm_head + RMSNorm + tokenizer 로드."""
    sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
    os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
    from core.model_loader import parse_model_args, load_model_from_args

    if model_name == "llava-video-7b":
        args_str = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"
    else:
        lp = LORA_PATHS[model_name]
        args_str = f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"

    model_args = parse_model_args(args_str)
    tokenizer, model, _, _, _, _ = load_model_from_args(model_args)

    lm_head_w = model.lm_head.weight.data.float().clone()
    norm_w = model.model.norm.weight.data.float().clone()

    # A, B, C, D token IDs
    letter_ids = {}
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        letter_ids[letter] = ids[0]

    del model
    return lm_head_w, norm_w, letter_ids, tokenizer


def apply_rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def mcq_logit_lens(feat_root, task, questions, lm_head_w, norm_w, letter_ids, num_layers):
    """MCQ 기준 logit lens: sample별로 정답 letter의 logit rank 확인."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # qid → question 매핑
    q_by_id = {}
    for q in questions:
        qid = q['q_id']
        q_by_id[qid] = q

    results = {"layers": [], "mcq_acc": [], "mean_correct_rank": []}

    for l in range(num_layers):
        feat, qids, _ = load_answer_layer(feat_root, task, l)
        X = torch.from_numpy(feat.astype(np.float32)).to(device)

        # RMSNorm + lm_head
        normed = apply_rmsnorm(X, norm_w.to(device))
        logits = normed @ lm_head_w.to(device).T  # (N, vocab)

        # A, B, C, D logits만 추출
        abcd_ids = torch.tensor([letter_ids[lt] for lt in LETTERS]).to(device)
        abcd_logits = logits[:, abcd_ids]  # (N, 4)

        correct = 0
        ranks = []
        for i, qid in enumerate(qids):
            q = q_by_id.get(str(qid), q_by_id.get(qid))
            if q is None:
                continue

            gt_answer = str(q['answer']).strip().upper()  # "A", "B", "C", or "D"
            gt_idx = LETTERS.index(gt_answer) if gt_answer in LETTERS else -1
            if gt_idx < 0:
                continue

            pred_idx = abcd_logits[i].argmax().item()
            if pred_idx == gt_idx:
                correct += 1

            # Rank of correct answer among A,B,C,D
            sorted_idx = abcd_logits[i].argsort(descending=True)
            rank = (sorted_idx == gt_idx).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

        N = len(ranks)
        acc = correct / N * 100 if N > 0 else 0
        mean_rank = np.mean(ranks) if ranks else -1

        results["layers"].append(l)
        results["mcq_acc"].append(acc)
        results["mean_correct_rank"].append(float(mean_rank))
        print(f"  Layer {l:2d}: MCQ acc={acc:5.1f}%  mean_rank={mean_rank:.2f}/4")

        del X, logits, normed, abcd_logits

    torch.cuda.empty_cache()
    return results


def mcq_confusion(feat_root, task, questions, lm_head_w, norm_w, letter_ids, layer_idx=-1):
    """Last layer MCQ confusion matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_by_id = {q['q_id']: q for q in questions}

    _, _, meta = load_answer_layer(feat_root, task, 0)
    nl = meta["num_layers"]
    li = nl + layer_idx if layer_idx < 0 else layer_idx

    feat, qids, _ = load_answer_layer(feat_root, task, li)
    X = torch.from_numpy(feat.astype(np.float32)).to(device)
    normed = apply_rmsnorm(X, norm_w.to(device))
    logits = normed @ lm_head_w.to(device).T

    abcd_ids = torch.tensor([letter_ids[lt] for lt in LETTERS]).to(device)
    abcd_logits = logits[:, abcd_ids]

    # Direction-level confusion (not letter-level)
    dir_cm = np.zeros((4, 4), dtype=int)  # GT direction vs Pred direction

    for i, qid in enumerate(qids):
        q = q_by_id.get(str(qid), q_by_id.get(qid))
        if q is None: continue

        candidates = q.get('candidates', [])
        if isinstance(candidates, str):
            import ast; candidates = ast.literal_eval(candidates)

        gt_answer = str(q['answer']).strip().upper()
        gt_idx = LETTERS.index(gt_answer) if gt_answer in LETTERS else -1
        if gt_idx < 0 or gt_idx >= len(candidates): continue

        gt_direction = candidates[gt_idx]  # e.g., "Up"
        pred_letter_idx = abcd_logits[i].argmax().item()
        pred_direction = candidates[pred_letter_idx] if pred_letter_idx < len(candidates) else "?"

        gt_dir_idx = DIRECTIONS.index(gt_direction) if gt_direction in DIRECTIONS else -1
        pred_dir_idx = DIRECTIONS.index(pred_direction) if pred_direction in DIRECTIONS else -1
        if gt_dir_idx >= 0 and pred_dir_idx >= 0:
            dir_cm[gt_dir_idx, pred_dir_idx] += 1

    total = dir_cm.sum()
    acc = np.diag(dir_cm).sum() / total * 100 if total > 0 else 0

    print(f"\n  Confusion (direction-level), acc={acc:.1f}%:")
    header = 'GT\\Pred'
    print(f"  {header:>10}", end="")
    for d in DIRECTIONS: print(f"  {d:>6}", end="")
    print(f"  {'row%':>6}")
    for i, d in enumerate(DIRECTIONS):
        print(f"  {d:>10}", end="")
        for j in range(4): print(f"  {dir_cm[i,j]:>6}", end="")
        ra = dir_cm[i,i] / max(dir_cm[i].sum(), 1) * 100
        print(f"  {ra:5.1f}%")

    del X, logits; torch.cuda.empty_cache()
    return {"confusion": dir_cm.tolist(), "accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="analysis/decoding_gap_results")
    args = parser.parse_args()

    feat_root = FEAT_ROOTS.get(args.model)
    if not feat_root:
        print(f"[ERROR] Unknown model: {args.model}"); return

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = TASKS if args.task == "all" else [args.task]

    print("Loading model weights...")
    lm_head_w, norm_w, letter_ids, tokenizer = load_model_weights(args.model)
    print(f"  Letter IDs: {letter_ids}")

    all_results = {}
    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  {args.model} / {task}")
        print(f"{'#'*60}")

        questions = load_dataset_questions(task)
        _, _, meta = load_answer_layer(feat_root, task, 0)
        num_layers = meta["num_layers"]

        task_results = {}
        print("\n  [Logit Lens — MCQ A/B/C/D]")
        task_results["logit_lens"] = mcq_logit_lens(feat_root, task, questions, lm_head_w, norm_w, letter_ids, num_layers)
        print("\n  [Confusion Matrix — MCQ]")
        task_results["confusion"] = mcq_confusion(feat_root, task, questions, lm_head_w, norm_w, letter_ids)
        all_results[task] = task_results

    save_path = os.path.join(args.output_dir, f"decoding_gap_mcq_{args.model}.json")
    def convert(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        return o
    with open(save_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {save_path}")


if __name__ == "__main__":
    main()
