"""
Direction Steering: Spatial ID 방식의 causal intervention.

1. Answer token에서 direction vector 추출:
   vec(Up) = mean(activation | Up) - mean(activation | all)

2. Causal steering:
   Up sample → vec(Up) 빼고 vec(Right) 더하기 → 답이 Right로 바뀌는가?

3. Layer별, model별, task별 steering 성공률 측정.

4. Cross-task steering: task A의 direction vector로 task B를 steering.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/direction_steering.py \
        --model llava-video-7b_lora_4combo_v2_baseline --task obj_place
"""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

MCQ_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way"
META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
    "llava-video-7b_lora_4combo_v2_delta": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_delta",
}

LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "llava-video-7b_lora_4combo_v2_delta": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
}

TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
DIRECTIONS = ["Up", "Down", "Left", "Right"]  # MCQ answer mapping
STEERING_LAYERS = [14, 16, 18, 20, 24]


def model_short(name):
    return name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")


# ============================================================
#  Direction Vector 추출 (from stored features)
# ============================================================

def extract_direction_vectors(feat_root, task, layer):
    """
    Stored answer token features에서 direction vector 추출.
    vec(dir) = mean(activation | dir) - mean(activation | all)
    Returns: dict {direction_str: torch.Tensor(D,)} on GPU
    """
    from sklearn.preprocessing import LabelEncoder

    d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
    feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
    qids = np.load(os.path.join(d, "qids.npy"))

    metadata = json.load(open(os.path.join(META_ROOT, f"{task}_metadata.json")))
    mb = {m['id']: m for m in metadata}
    dir_labels = [str(mb[int(str(q).split('_')[0])]["direction"]).capitalize() for q in qids]

    device = torch.device("cuda")
    X = torch.from_numpy(feat).to(device)
    global_mean = X.mean(0)

    vectors = {}
    for direction in DIRECTIONS:
        mask = [i for i, dl in enumerate(dir_labels) if dl.lower() == direction.lower()]
        if len(mask) == 0:
            continue
        class_mean = X[mask].mean(0)
        vectors[direction] = (class_mean - global_mean).to(torch.float16)

    # Magnitude 정보
    for d_name, vec in vectors.items():
        print(f"    vec({d_name}): norm={vec.float().norm():.2f}")

    del X
    return vectors


# ============================================================
#  Model Loading
# ============================================================

def load_model(model_name):
    from core.model_loader import parse_model_args, load_model_from_args
    if model_name == "llava-video-7b":
        a = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    else:
        lp = LORA_PATHS[model_name]
        a = f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    ma = parse_model_args(a)
    tok, model, ip, cl, mn, ct = load_model_from_args(ma)
    model.eval()
    return tok, model, ip, ct


# ============================================================
#  Steering Evaluation
# ============================================================

def evaluate_steering(model, tokenizer, image_processor, conv_template,
                      task, target_layer, direction_vectors, alpha=1.0, limit=200):
    """
    각 sample에 대해:
      1. No intervention → original prediction
      2. 정답 direction vec 빼고, 다른 3방향 vec 각각 더하기 → steering 성공률
    """
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions

    questions, _ = load_dataset_as_questions(task_name=TASK_FULL(task), limit=limit)
    data_loader = create_data_loader(
        questions, "", 1, 2, tokenizer, image_processor, model.config,
        TASK_FULL(task), conv_template, video_folder="", video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    mcq_data = json.load(open(os.path.join(MCQ_ROOT, f"{task}.json")))
    mcq_by_id = {m['id']: m for m in mcq_data}

    results = {
        "no_intervention": {"correct": 0, "total": 0},
        "steering_success": 0,  # target으로 바뀐 횟수
        "steering_attempts": 0,  # steering 시도 횟수
        "per_direction": {},
    }

    for target_dir in DIRECTIONS:
        results["per_direction"][target_dir] = {"attempts": 0, "success": 0}

    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions), desc=f"  L{target_layer}"
    ):
        sid = int(str(line['q_id']).split('_')[0])
        mcq = mcq_by_id.get(sid)
        if not mcq:
            continue

        gt_answer = mcq['answer']  # "A", "B", "C", "D"
        candidates = mcq.get('candidates', [])
        if isinstance(candidates, str):
            import ast; candidates = ast.literal_eval(candidates)

        gt_idx = "ABCD".index(gt_answer) if gt_answer in "ABCD" else -1
        if gt_idx < 0 or gt_idx >= len(candidates):
            continue
        gt_direction = candidates[gt_idx]  # e.g., "Up"

        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]

        # 1. No intervention
        with torch.inference_mode():
            output = model.generate(
                inputs=input_ids, images=image_tensor, image_sizes=image_sizes,
                modalities=[modality], do_sample=False, max_new_tokens=1,
                use_cache=True, pad_token_id=tokenizer.eos_token_id,
            )
        pred_orig = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        pred_orig_letter = pred_orig[0].upper() if pred_orig else ""

        if pred_orig_letter == gt_answer:
            results["no_intervention"]["correct"] += 1
        results["no_intervention"]["total"] += 1

        # 2. Steering: gt_direction → each other direction
        if gt_direction not in direction_vectors:
            continue

        vec_source = direction_vectors[gt_direction]

        for target_dir in DIRECTIONS:
            if target_dir == gt_direction:
                continue
            if target_dir not in direction_vectors:
                continue

            vec_target = direction_vectors[target_dir]

            # Find target letter in candidates
            if target_dir not in candidates:
                continue
            target_letter = "ABCD"[candidates.index(target_dir)]

            # Hook: subtract source vec, add target vec
            def make_steering_hook(v_src, v_tgt, a):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    dtype = h.dtype
                    delta = (v_tgt.to(dtype) - v_src.to(dtype)) * a
                    h[0, -1, :] += delta
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook_fn

            hook = model.model.layers[target_layer].register_forward_hook(
                make_steering_hook(vec_source, vec_target, alpha)
            )

            with torch.inference_mode():
                output_steered = model.generate(
                    inputs=input_ids, images=image_tensor, image_sizes=image_sizes,
                    modalities=[modality], do_sample=False, max_new_tokens=1,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id,
                )

            hook.remove()

            pred_steered = tokenizer.decode(output_steered[0], skip_special_tokens=True).strip()
            pred_steered_letter = pred_steered[0].upper() if pred_steered else ""

            results["steering_attempts"] += 1
            results["per_direction"][target_dir]["attempts"] += 1

            if pred_steered_letter == target_letter:
                results["steering_success"] += 1
                results["per_direction"][target_dir]["success"] += 1

    # Summary
    ni = results["no_intervention"]
    ni_acc = ni["correct"] / ni["total"] * 100 if ni["total"] > 0 else 0
    steer_rate = results["steering_success"] / results["steering_attempts"] * 100 if results["steering_attempts"] > 0 else 0

    print(f"    No intervention: {ni_acc:.1f}% ({ni['correct']}/{ni['total']})")
    print(f"    Steering success: {steer_rate:.1f}% ({results['steering_success']}/{results['steering_attempts']})")
    for td in DIRECTIONS:
        pd = results["per_direction"][td]
        if pd["attempts"] > 0:
            print(f"      → {td}: {pd['success']}/{pd['attempts']} ({pd['success']/pd['attempts']*100:.1f}%)")

    results["no_intervention_acc"] = ni_acc
    results["steering_rate"] = steer_rate
    return results


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task", default="obj_place")
    parser.add_argument("--vector_task", default=None, help="Direction vector를 추출할 task (cross-task steering용)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output_dir", default="analysis/steering_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    feat_root = FEAT_ROOTS[args.model]
    vector_task = args.vector_task or args.task

    print(f"Loading model: {args.model}")
    tok, model, ip, ct = load_model(args.model)

    all_results = {"model": args.model, "task": args.task, "vector_task": vector_task, "alpha": args.alpha}

    for layer in STEERING_LAYERS:
        print(f"\n{'='*50}")
        print(f"  Layer {layer} — vectors from {vector_task}, eval on {args.task}")
        print(f"{'='*50}")

        direction_vectors = extract_direction_vectors(feat_root, vector_task, layer)
        layer_results = evaluate_steering(
            model, tok, ip, ct, args.task, layer, direction_vectors, args.alpha, args.limit
        )
        all_results[f"layer_{layer}"] = layer_results

    # Summary table
    print(f"\n{'='*50}")
    print(f"  SUMMARY: {args.model} / eval={args.task} / vec={vector_task}")
    print(f"{'='*50}")
    print(f"  {'Layer':>6} {'MCQ (no interv)':>16} {'Steering rate':>14}")
    for layer in STEERING_LAYERS:
        r = all_results.get(f"layer_{layer}", {})
        print(f"  {layer:>6} {r.get('no_intervention_acc', 0):>15.1f}% {r.get('steering_rate', 0):>13.1f}%")

    short = model_short(args.model)
    vtask = model_short(vector_task) if vector_task != args.task else "same"
    sp = os.path.join(args.output_dir, f"steering_{short}_{args.task}_vec_{vtask}.json")
    with open(sp, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {sp}")


if __name__ == "__main__":
    main()
