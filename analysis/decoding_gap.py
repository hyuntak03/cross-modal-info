"""
Direction Knowledge vs Decoding Gap 분석.

핵심 질문: Linear probe는 direction을 맞추는데, 왜 lm_head를 통한 실제 decoding은 실패하는가?

실험:
  Exp1. Logit Lens — lm_head를 각 layer에 적용하면 정답 direction token을 예측하는가?
    - answer token h_l → RMSNorm → lm_head → logits
    - 정답 direction token의 logit rank & accuracy
    - Linear probe acc vs Logit lens acc per layer

  Exp2. Direction Centroid — lm_head Alignment
    - Direction별 centroid (mean hidden state) 계산
    - centroid → RMSNorm → lm_head → 정답 token이 top-1인지
    - Task별: alignment 정도 비교

  Exp3. Confusion Matrix
    - Logit lens로 예측한 결과의 confusion matrix
    - 어떤 방향을 어떤 방향으로 혼동하는가?
    - Task별로 systematic bias가 있는지

  Exp4. Cross-Task Logit Lens
    - shape_color에서 학습된 direction 축이 obj_place에서도 lm_head로 decode되는가?
    - Linear probe cross-task acc vs Logit lens cross-task acc

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/decoding_gap.py \
        --model llava-video-7b_lora_4combo_v2_baseline
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
}

DIRECTIONS = ["up", "down", "left", "right"]


def load_metadata(task):
    with open(os.path.join(META_ROOT, f"{task}_metadata.json")) as f:
        return json.load(f)


def get_direction_labels(metadata, qids):
    meta_by_id = {m['id']: m for m in metadata}
    directions = [meta_by_id[int(str(q).split('_')[0])]['direction'] for q in qids]
    le = LabelEncoder()
    le.fit(DIRECTIONS)
    return le.transform(directions), le.classes_


def load_answer_layer(feat_root, task, layer_idx):
    d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
    feat = np.array(np.load(os.path.join(d, f"features_layer_{layer_idx}.npy"), mmap_mode='r'))
    qids = np.load(os.path.join(d, "qids.npy"))
    meta = np.load(os.path.join(d, "meta.npy"), allow_pickle=True).item()
    return feat, qids, meta


# ============================================================
#  Model weights 로드 (lm_head + RMSNorm만)
# ============================================================

def load_lm_head_weights(model_name):
    """lm_head weight + RMSNorm weight + tokenizer 로드. 전체 모델 안 올림."""
    lora_paths = {
        "llava-video-7b_lora_4combo_v2_baseline":
            "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
        "llava-video-7b_lora_syn_v4_baseline":
            "/data/jong980812/project/LLaVA-NeXT/work_dirs/LLaVA-Video-7B-Qwen2/llava-video-7b-qwen2_lora-r64_motion_v4_100K_baseline_rope1d_f8_ep1_lr1e-5",
    }

    # Load full model (LoRA merged) — 필요한 것만 추출 후 삭제
    sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
    os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
    from core.model_loader import parse_model_args, load_model_from_args

    if model_name == "llava-video-7b":
        args_str = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"
    elif model_name in lora_paths:
        lp = lora_paths[model_name]
        args_str = f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model_args = parse_model_args(args_str)
    tokenizer, model, _, _, _, _ = load_model_from_args(model_args)

    # 추출
    lm_head_weight = model.lm_head.weight.data.float().clone()  # (vocab, 3584)
    norm_weight = model.model.norm.weight.data.float().clone()  # (3584,)

    # Direction token IDs
    dir_token_ids = {}
    for d in DIRECTIONS:
        # "Up", "Down", "Left", "Right" (첫 글자 대문자)
        tokens = tokenizer.encode(d.capitalize(), add_special_tokens=False)
        dir_token_ids[d] = tokens[0]
        # 또한 MCQ 답변 형태: "A", "B", "C", "D"

    # MCQ letter token IDs
    mcq_ids = {}
    for letter in ["A", "B", "C", "D"]:
        mcq_ids[letter] = tokenizer.encode(letter, add_special_tokens=False)[0]

    del model
    torch.cuda.empty_cache()

    return lm_head_weight, norm_weight, dir_token_ids, mcq_ids, tokenizer


def apply_rmsnorm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / rms(x)"""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def logit_lens(hidden_states, lm_head_weight, norm_weight):
    """hidden_states → RMSNorm → lm_head → logits. All GPU."""
    device = hidden_states.device
    normed = apply_rmsnorm(hidden_states, norm_weight.to(device))
    logits = normed @ lm_head_weight.to(device).T  # (N, vocab)
    return logits


# ============================================================
#  Exp1: Logit Lens per Layer
# ============================================================

def exp1_logit_lens(feat_root, task, lm_head_w, norm_w, dir_token_ids, num_layers=29):
    """각 layer에서 logit lens → direction token 예측 정확도."""
    print(f"\n{'='*60}")
    print(f"  Exp1: Logit Lens — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"layers": [], "logit_acc": [], "top1_is_direction": [], "mean_direction_rank": []}
    dir_ids = torch.tensor([dir_token_ids[d] for d in DIRECTIONS]).to(device)

    for l in range(num_layers):
        feat, qids, _ = load_answer_layer(feat_root, task, l)
        labels, classes = get_direction_labels(metadata, qids)

        X = torch.from_numpy(feat.astype(np.float32)).to(device)
        y = torch.from_numpy(labels).long().to(device)

        logits = logit_lens(X, lm_head_w, norm_w)  # (N, vocab)

        # Direction token logits만 추출
        dir_logits = logits[:, dir_ids]  # (N, 4) — up, down, left, right 순
        preds = dir_logits.argmax(dim=1)
        acc = (preds == y).float().mean().item() * 100

        # Top-1 predicted token이 4개 direction 중 하나인지
        top1_tokens = logits.argmax(dim=1)
        is_dir = sum(1 for t in top1_tokens if t.item() in dir_token_ids.values())
        top1_dir_pct = is_dir / len(top1_tokens) * 100

        # Direction tokens의 평균 rank (낮을수록 좋음)
        ranks = []
        for i in range(len(X)):
            correct_dir = DIRECTIONS[labels[i]]
            correct_id = dir_token_ids[correct_dir]
            sorted_ids = logits[i].argsort(descending=True)
            rank = (sorted_ids == correct_id).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        mean_rank = np.mean(ranks)

        results["layers"].append(l)
        results["logit_acc"].append(acc)
        results["top1_is_direction"].append(top1_dir_pct)
        results["mean_direction_rank"].append(float(mean_rank))
        print(f"  Layer {l:2d}: logit_acc={acc:5.1f}%  top1_is_dir={top1_dir_pct:5.1f}%  mean_rank={mean_rank:.0f}")

        del X, logits, dir_logits

    torch.cuda.empty_cache()
    return results


# ============================================================
#  Exp2: Direction Centroid — lm_head Alignment
# ============================================================

def exp2_centroid_alignment(feat_root, task, lm_head_w, norm_w, dir_token_ids, layer_idx=-1):
    """Direction centroid → lm_head → 정답 token alignment."""
    print(f"\n{'='*60}")
    print(f"  Exp2: Centroid Alignment — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, meta = load_answer_layer(feat_root, task, 0)
    nl = meta["num_layers"]
    li = nl + layer_idx if layer_idx < 0 else layer_idx

    feat, qids, _ = load_answer_layer(feat_root, task, li)
    labels, classes = get_direction_labels(metadata, qids)
    X = torch.from_numpy(feat.astype(np.float32)).to(device)
    y = torch.from_numpy(labels).long().to(device)

    # Per-direction centroid
    centroids = []
    for d_idx, d_name in enumerate(DIRECTIONS):
        mask = y == d_idx
        centroid = X[mask].mean(dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids)  # (4, 3584)

    # lm_head에서 direction token weight
    dir_ids = [dir_token_ids[d] for d in DIRECTIONS]
    lm_dir_weights = lm_head_w[dir_ids].to(device)  # (4, 3584)

    # Cosine similarity: centroid[i] vs lm_head_weight[j]
    cent_norm = centroids / centroids.norm(dim=1, keepdim=True)
    lm_norm = lm_dir_weights / lm_dir_weights.norm(dim=1, keepdim=True)
    cos_sim = (cent_norm @ lm_norm.T).cpu().numpy()  # (4, 4)

    print(f"  Cosine Similarity (centroid vs lm_head direction tokens):")
    print(f"  {'':>12}", end="")
    for d in DIRECTIONS: print(f"  lm_{d:>5}", end="")
    print()
    for i, d in enumerate(DIRECTIONS):
        print(f"  cent_{d:>5}", end="")
        for j in range(4):
            print(f"  {cos_sim[i,j]:7.4f}", end="")
        print(f"  {'✓' if cos_sim[i].argmax() == i else '✗'}")

    # Centroid → logit lens
    logits = logit_lens(centroids, lm_head_w, norm_w)
    dir_logits = logits[:, [dir_token_ids[d] for d in DIRECTIONS]]
    for i, d in enumerate(DIRECTIONS):
        pred_dir = DIRECTIONS[dir_logits[i].argmax().item()]
        correct = pred_dir == d
        print(f"  Centroid {d:>5} → predicted: {pred_dir:>5} {'✓' if correct else '✗'}")

    del X; torch.cuda.empty_cache()
    return {"cos_sim": cos_sim.tolist(), "diagonal_mean": float(np.diag(cos_sim).mean())}


# ============================================================
#  Exp3: Confusion Matrix (Logit Lens)
# ============================================================

def exp3_confusion(feat_root, task, lm_head_w, norm_w, dir_token_ids, layer_idx=-1):
    """Logit lens confusion matrix."""
    print(f"\n{'='*60}")
    print(f"  Exp3: Confusion Matrix — {task}")
    print(f"{'='*60}")

    metadata = load_metadata(task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, meta = load_answer_layer(feat_root, task, 0)
    nl = meta["num_layers"]
    li = nl + layer_idx if layer_idx < 0 else layer_idx

    feat, qids, _ = load_answer_layer(feat_root, task, li)
    labels, classes = get_direction_labels(metadata, qids)
    X = torch.from_numpy(feat.astype(np.float32)).to(device)

    logits = logit_lens(X, lm_head_w, norm_w)
    dir_ids = [dir_token_ids[d] for d in DIRECTIONS]
    dir_logits = logits[:, dir_ids]
    preds = dir_logits.argmax(dim=1).cpu().numpy()

    # Confusion matrix
    cm = np.zeros((4, 4), dtype=int)
    for gt, pred in zip(labels, preds):
        cm[gt, pred] += 1

    header = 'GT \\ Pred'
    print(f"  {header:>12}", end="")
    for d in DIRECTIONS: print(f"  {d:>6}", end="")
    print(f"  {'acc':>6}")
    for i, d in enumerate(DIRECTIONS):
        print(f"  {d:>12}", end="")
        for j in range(4):
            print(f"  {cm[i,j]:>6}", end="")
        row_acc = cm[i,i] / max(cm[i].sum(), 1) * 100
        print(f"  {row_acc:5.1f}%")

    total_acc = np.diag(cm).sum() / cm.sum() * 100
    print(f"  Total accuracy: {total_acc:.1f}%")

    del X, logits; torch.cuda.empty_cache()
    return {"confusion_matrix": cm.tolist(), "accuracy": float(total_acc)}


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="analysis/decoding_gap_results")
    args = parser.parse_args()

    feat_root = FEAT_ROOTS.get(args.model)
    if not feat_root:
        print(f"[ERROR] Unknown model: {args.model}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = TASKS if args.task == "all" else [args.task]

    print("Loading lm_head + RMSNorm weights...")
    lm_head_w, norm_w, dir_token_ids, mcq_ids, tokenizer = load_lm_head_weights(args.model)
    print(f"  Direction token IDs: {dir_token_ids}")
    print(f"  MCQ token IDs: {mcq_ids}")

    all_results = {}
    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  {args.model} / {task}")
        print(f"{'#'*60}")

        _, _, meta = load_answer_layer(feat_root, task, 0)
        num_layers = meta["num_layers"]

        task_results = {}
        task_results["exp1_logit_lens"] = exp1_logit_lens(feat_root, task, lm_head_w, norm_w, dir_token_ids, num_layers)
        task_results["exp2_alignment"] = exp2_centroid_alignment(feat_root, task, lm_head_w, norm_w, dir_token_ids)
        task_results["exp3_confusion"] = exp3_confusion(feat_root, task, lm_head_w, norm_w, dir_token_ids)
        all_results[task] = task_results

    save_path = os.path.join(args.output_dir, f"decoding_gap_{args.model}.json")
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
