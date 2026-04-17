"""
Cross-Combination Evaluation: Projector vs LLM 기여도 분리.

4가지 조합:
  (1) Baseline Projector + Baseline LLM  (원본)
  (2) Delta Projector + Delta LLM        (원본)
  (3) Delta Projector + Baseline LLM     → Projector 기여도
  (4) Baseline Projector + Delta LLM     → LLM 기여도

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/cross_combination_eval.py
"""

import os, sys, json, argparse, math, string, ast
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

MCQ_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way"

CONFIGS = {
    "baseline": {
        "lora": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    },
    "delta": {
        "lora": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    },
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
DIRECTIONS = ["Up", "Down", "Left", "Right"]


def load_model_with_swap(lora_config, projector_config):
    """lora_config의 LoRA + projector_config의 projector로 모델 로드."""
    from core.model_loader import parse_model_args, load_model_from_args

    lora_path = CONFIGS[lora_config]["lora"]
    args_str = f"lora_pretrained={lora_path},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"

    model_args = parse_model_args(args_str)
    tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args)
    model.eval()

    # Projector swap (if different from lora config)
    if projector_config != lora_config:
        proj_path = os.path.join(CONFIGS[projector_config]["lora"], "non_lora_trainables.bin")
        proj_weights = torch.load(proj_path, map_location='cpu')

        print(f"  Swapping projector: {lora_config} LLM + {projector_config} projector")
        for key, value in proj_weights.items():
            if 'mm_projector' in key:
                # base_model.model.model.mm_projector.X.weight/bias
                parts = key.replace('base_model.model.model.', '').split('.')
                module = model.get_model()
                for p in parts[:-1]:
                    if p.isdigit():
                        module = module[int(p)]
                    else:
                        module = getattr(module, p)
                param_name = parts[-1]
                target = getattr(module, param_name)
                getattr(module, param_name).data = value.to(device=target.device, dtype=target.dtype)
                print(f"    Swapped: {key}")

    return tokenizer, model, image_processor, conv_template


def evaluate_mcq(model, tokenizer, image_processor, conv_template, task, limit=200):
    """R2R 4way MCQ evaluation."""
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions

    questions, _ = load_dataset_as_questions(task_name=TASK_FULL(task), limit=limit)
    data_loader = create_data_loader(
        questions, "", 1, 2, tokenizer, image_processor, model.config,
        TASK_FULL(task), conv_template, video_folder="", video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    # Load MCQ answers
    mcq_data = json.load(open(os.path.join(MCQ_ROOT, f"{task}.json")))
    mcq_by_id = {m['id']: m for m in mcq_data}

    correct = 0
    total = 0

    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions), desc=f"  {task}"
    ):
        sid = int(str(line['q_id']).split('_')[0])
        mcq = mcq_by_id.get(sid)
        if not mcq:
            continue

        gt_answer = mcq['answer']  # "A", "B", "C", or "D"

        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]

        if "v1.6" in "llava-video" or "v1.5" in "llava-video":
            eff_mod = modality
        else:
            eff_mod = modality

        with torch.inference_mode():
            output = model.generate(
                inputs=input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                modalities=[eff_mod],
                do_sample=False,
                temperature=0,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # LLaVA generate() returns only generated tokens (not input+output)
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        pred_letter = pred_text[0].upper() if pred_text else ""

        if pred_letter == gt_answer:
            correct += 1
        total += 1

    acc = correct / total * 100 if total > 0 else 0
    return acc, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="analysis/cross_combination_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    combinations = [
        ("baseline", "baseline", "Baseline Proj + Baseline LLM"),
        ("delta",    "delta",    "Delta Proj + Delta LLM"),
        ("baseline", "delta",    "Delta Proj + Baseline LLM"),  # projector 기여
        ("delta",    "baseline", "Baseline Proj + Delta LLM"),  # LLM 기여
    ]

    all_results = {}

    for lora_cfg, proj_cfg, desc in combinations:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"  (LLM LoRA: {lora_cfg}, Projector: {proj_cfg})")
        print(f"{'='*60}")

        tokenizer, model, image_processor, conv_template = load_model_with_swap(lora_cfg, proj_cfg)

        combo_key = f"{proj_cfg}_proj__{lora_cfg}_llm"
        all_results[combo_key] = {"desc": desc}

        for task in TASKS:
            acc, correct, total = evaluate_mcq(model, tokenizer, image_processor, conv_template, task)
            all_results[combo_key][task] = {"acc": acc, "correct": correct, "total": total}
            print(f"    {task}: {acc:.1f}% ({correct}/{total})")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Configuration':<35}", end="")
    for t in TASKS:
        print(f" {t:>12}", end="")
    print()

    for combo_key in all_results:
        r = all_results[combo_key]
        print(f"  {r['desc']:<35}", end="")
        for t in TASKS:
            print(f" {r[t]['acc']:>11.1f}%", end="")
        print()

    # Save
    save_path = os.path.join(args.output_dir, "cross_combination.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {save_path}")


if __name__ == "__main__":
    main()
