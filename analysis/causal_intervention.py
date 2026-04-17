"""
Causal Intervention: Direction Subspace Ablation.

Critical layer에서 answer token으로 가는 attention output을 조작:
  (1) No intervention — baseline
  (2) Keep direction — attn_out을 direction subspace로 project
  (3) Remove direction — attn_out에서 direction subspace를 제거
  (4) Keep identity — attn_out을 identity subspace로 project
  (5) Remove identity — attn_out에서 identity subspace를 제거

Direction subspace: stored answer token features에서 between-class scatter의 top-k eigenvectors.

MCQ accuracy 변화로 인과적 증거 제공.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/causal_intervention.py \
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
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
    "llava-video-7b_lora_4combo_v2_delta": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_delta",
}

LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "llava-video-7b_lora_4combo_v2_delta": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
}

TASKS = ["shape_color", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
IDENTITY_ATTRS = {"shape_color": "shape", "obj_color": "obj_class", "shape_place": "place_class", "obj_place": "obj_class"}
INTERVENTION_LAYERS = [10, 14, 16, 18, 20, 24]


# ============================================================
#  Subspace Computation
# ============================================================

def compute_direction_subspace(feat_root, task, layer, k=50):
    """Between-class scatter의 top-k eigenvectors → direction subspace basis."""
    from sklearn.preprocessing import LabelEncoder

    d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
    feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
    qids = np.load(os.path.join(d, "qids.npy"))

    metadata = json.load(open(os.path.join(META_ROOT, f"{task}_metadata.json")))
    mb = {m['id']: m for m in metadata}
    le = LabelEncoder()
    dir_labels = le.fit_transform([str(mb[int(str(q).split('_')[0])]["direction"]) for q in qids])
    nc = len(le.classes_)

    # Between-class scatter
    device = torch.device("cuda")
    X = torch.from_numpy(feat).to(device)
    y = torch.from_numpy(dir_labels).long().to(device)
    gm = X.mean(0)

    S_b = torch.zeros(X.shape[1], X.shape[1], device=device)
    for c in range(nc):
        mask = (y == c)
        if mask.sum() == 0: continue
        cm = X[mask].mean(0) - gm
        S_b += mask.sum().float() * cm.unsqueeze(1) @ cm.unsqueeze(0)

    # Top-k eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(S_b)
    # eigh returns ascending order, take last k
    U_dir = eigvecs[:, -k:].contiguous()  # (D, k)

    del X, y, S_b, eigvals, eigvecs
    torch.cuda.empty_cache()
    return U_dir  # on GPU


def compute_identity_subspace(feat_root, task, layer, k=50):
    """Identity attribute의 between-class scatter top-k eigenvectors."""
    from sklearn.preprocessing import LabelEncoder

    d = os.path.join(feat_root, "answer_token", TASK_FULL(task))
    feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
    qids = np.load(os.path.join(d, "qids.npy"))

    metadata = json.load(open(os.path.join(META_ROOT, f"{task}_metadata.json")))
    mb = {m['id']: m for m in metadata}
    le = LabelEncoder()
    id_attr = IDENTITY_ATTRS[task]
    id_labels = le.fit_transform([str(mb[int(str(q).split('_')[0])][id_attr]) for q in qids])
    nc = len(le.classes_)

    device = torch.device("cuda")
    X = torch.from_numpy(feat).to(device)
    y = torch.from_numpy(id_labels).long().to(device)
    gm = X.mean(0)

    S_b = torch.zeros(X.shape[1], X.shape[1], device=device)
    for c in range(nc):
        mask = (y == c)
        if mask.sum() == 0: continue
        cm = X[mask].mean(0) - gm
        S_b += mask.sum().float() * cm.unsqueeze(1) @ cm.unsqueeze(0)

    eigvals, eigvecs = torch.linalg.eigh(S_b)
    U_id = eigvecs[:, -k:].contiguous()

    del X, y, S_b
    torch.cuda.empty_cache()
    return U_id


# ============================================================
#  Model Loading
# ============================================================

def load_model(model_name):
    from core.model_loader import parse_model_args, load_model_from_args
    lp = LORA_PATHS[model_name]
    a = f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    ma = parse_model_args(a)
    tok, model, ip, cl, mn, ct = load_model_from_args(ma)
    model.eval()
    return tok, model, ip, ct


# ============================================================
#  MCQ Evaluation with Intervention
# ============================================================

def evaluate_with_intervention(model, tokenizer, image_processor, conv_template,
                               task, intervention_type, target_layer, U_sub=None, limit=200):
    """
    intervention_type:
      "none" — no intervention
      "keep_subspace" — project attn output to U_sub
      "remove_subspace" — project OUT U_sub from attn output
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

    correct = 0
    total = 0

    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions),
        desc=f"  {intervention_type}@L{target_layer}"
    ):
        sid = int(str(line['q_id']).split('_')[0])
        mcq = mcq_by_id.get(sid)
        if not mcq:
            continue
        gt_answer = mcq['answer']

        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]

        # Register intervention hook
        hooks = []
        if intervention_type != "none" and U_sub is not None:
            def make_intervention_hook(U):
                def hook_fn(module, input, output):
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # Last token position의 attention output만 조작
                    last_h = attn_out[0, -1, :]  # (D,)
                    dtype = last_h.dtype

                    # Project (match dtype)
                    U_cast = U.to(dtype=dtype)
                    proj = U_cast @ (U_cast.T @ last_h)  # subspace component

                    if intervention_type == "keep_subspace":
                        attn_out[0, -1, :] = proj
                    elif intervention_type == "remove_subspace":
                        attn_out[0, -1, :] = last_h - proj

                    if isinstance(output, tuple):
                        return (attn_out,) + output[1:]
                    return attn_out
                return hook_fn

            # decoder layer hook (self_attn hook은 Qwen2에서 return값이 반영 안 됨)
            hooks.append(model.model.layers[target_layer].register_forward_hook(make_intervention_hook(U_sub)))

        with torch.inference_mode():
            output = model.generate(
                inputs=input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                modalities=[modality],
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for h in hooks:
            h.remove()

        # LLaVA generate returns only generated tokens
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        pred_letter = pred_text[0].upper() if pred_text else ""

        if pred_letter == gt_answer:
            correct += 1
        total += 1

    acc = correct / total * 100 if total > 0 else 0
    return acc, correct, total


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task", default="obj_place")
    parser.add_argument("--subspace_k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output_dir", default="analysis/causal_results")
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    os.makedirs(args.output_dir, exist_ok=True)

    feat_root = FEAT_ROOTS[args.model]

    print("Loading model...")
    tok, model, ip, ct = load_model(args.model)

    all_results = {}

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  {args.model} / {task}")
        print(f"{'#'*60}")

        task_results = {}

        # Baseline (no intervention)
        print("\n  [No intervention]")
        acc, c, t = evaluate_with_intervention(model, tok, ip, ct, task, "none", -1, None, limit=args.limit)
        task_results["no_intervention"] = {"acc": acc, "correct": c, "total": t}
        print(f"    → {acc:.1f}% ({c}/{t})")

        for layer in INTERVENTION_LAYERS:
            print(f"\n  --- Layer {layer} ---")

            # Compute subspaces
            U_dir = compute_direction_subspace(feat_root, task, layer, k=args.subspace_k)
            U_id = compute_identity_subspace(feat_root, task, layer, k=args.subspace_k)

            # Subspace overlap check
            overlap = torch.trace(U_dir.T @ U_id @ U_id.T @ U_dir).item()
            print(f"  Direction-Identity subspace overlap (trace): {overlap:.2f}/{args.subspace_k}")

            for intervention, U, label in [
                ("keep_direction", U_dir, "Keep direction subspace"),
                ("remove_direction", U_dir, "Remove direction subspace"),
                ("keep_identity", U_id, "Keep identity subspace"),
                ("remove_identity", U_id, "Remove identity subspace"),
            ]:
                print(f"\n  [{label} @ Layer {layer}]")
                acc, c, t = evaluate_with_intervention(
                    model, tok, ip, ct, task,
                    "keep_subspace" if "keep" in intervention else "remove_subspace",
                    layer, U, limit=args.limit
                )
                key = f"{intervention}_L{layer}"
                task_results[key] = {"acc": acc, "correct": c, "total": t, "layer": layer}
                print(f"    → {acc:.1f}% ({c}/{t})")

            del U_dir, U_id
            torch.cuda.empty_cache()

        all_results[task] = task_results

        # Summary
        print(f"\n  {'='*50}")
        print(f"  SUMMARY — {task}")
        print(f"  {'='*50}")
        base_acc = task_results["no_intervention"]["acc"]
        print(f"  No intervention: {base_acc:.1f}%")
        for layer in INTERVENTION_LAYERS:
            for intervention in ["keep_direction", "remove_direction", "keep_identity", "remove_identity"]:
                key = f"{intervention}_L{layer}"
                if key in task_results:
                    acc = task_results[key]["acc"]
                    delta = acc - base_acc
                    print(f"  {intervention:>20} @ L{layer}: {acc:.1f}% ({delta:+.1f}%p)")

    # Save
    short = model_short(args.model)
    sp = os.path.join(args.output_dir, f"causal_{short}_{args.task}.json")
    with open(sp, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {sp}")


def model_short(name):
    return name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")


if __name__ == "__main__":
    main()
