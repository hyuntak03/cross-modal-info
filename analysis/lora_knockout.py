"""
LoRA Knockout Experiment.

Merged LoRA 모델 (Baseline)에서 특정 layer의 LoRA 효과만 제거하고 MCQ/direction accuracy 변화 측정.

방법:
  1. Vanilla model의 weight 저장 (W_vanilla)
  2. Baseline model의 weight 저장 (W_baseline = W_vanilla + Δ)
  3. 각 module별 delta: Δ = W_baseline - W_vanilla
  4. Knockout layer L: W_baseline[L] -= Δ[L] → Vanilla weight로 복원
  5. Forward pass → direction word logit argmax → direction accuracy
  6. Restore: W_baseline[L] += Δ[L]

Metric:
  - direction_word: Up/Down/Left/Right logit argmax (candidate format 무관)
  - letter: A/B/C/D logit argmax (MCQ 정답률)

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/lora_knockout.py \
    --mode layer --layers 0,3,7,10,14,18,21,24,27 \
    --n_per_task 300 --metric direction_word
"""

import os, sys, json, argparse, gc
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

VANILLA_ARGS_GPU = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
# Vanilla는 CPU 로드 (delta 계산 후 바로 free)
VANILLA_ARGS_CPU = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=cpu,force_sample=True"

BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"
DIRECTION_WORDS = ["Up", "Down", "Left", "Right"]
LETTERS = ["A", "B", "C", "D"]

TARGET_MODULE_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


# ============================================================
#  Delta 계산 & Knockout
# ============================================================

def compute_lora_deltas(vanilla_model, baseline_model):
    """두 모델의 LLM layer 가중치 차이 계산. CPU에 저장."""
    deltas = {}
    vanilla_params = dict(vanilla_model.named_parameters())
    for name, p_b in baseline_model.named_parameters():
        # LLM layers만 target
        if "model.layers." not in name:
            continue
        # Target suffix만
        if not any(s in name for s in TARGET_MODULE_SUFFIXES):
            continue
        if name in vanilla_params:
            p_v = vanilla_params[name]
            delta = (p_b.data.float() - p_v.data.float()).cpu()
            if delta.abs().max() > 1e-5:
                deltas[name] = delta
    return deltas


def get_layer_idx(name):
    """param name에서 layer index 추출."""
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def knockout_layers(baseline_model, deltas, layer_indices, module_filter=None):
    """
    Specified layers(+optional module filter)의 delta를 baseline에서 뺌.
    Returns: applied_deltas dict {name: delta_on_gpu} for restore.
    """
    layer_set = set(layer_indices)
    applied = {}
    for name, p in baseline_model.named_parameters():
        li = get_layer_idx(name)
        if li is None or li not in layer_set:
            continue
        if name not in deltas:
            continue
        if module_filter and not any(m in name for m in module_filter):
            continue
        delta_gpu = deltas[name].to(p.device, p.dtype)
        p.data -= delta_gpu
        applied[name] = delta_gpu
    return applied


def restore(baseline_model, applied):
    """Restore by adding deltas back."""
    for name, p in baseline_model.named_parameters():
        if name in applied:
            p.data += applied[name]


# ============================================================
#  MCQ eval with optimized forward
# ============================================================

def evaluate(model, tokenizer, image_processor, conv_template, task, n_samples, metric="direction_word"):
    """Forward pass N samples → direction accuracy via lm_head argmax."""
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions

    questions, _ = load_dataset_as_questions(task_name=TASK_FULL(task), limit=n_samples)
    data_loader = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        TASK_FULL(task), conv_template, video_folder="", video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    if metric == "direction_word":
        target_tokens = [tokenizer.encode(w, add_special_tokens=False)[0] for w in DIRECTION_WORDS]
    elif metric == "letter":
        target_tokens = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    else:
        raise ValueError(metric)

    correct = 0
    total = 0

    # lm_head + final norm
    lm_head = model.lm_head
    final_norm = model.model.norm

    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in zip(data_loader, questions):
        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]

        # Forward with prepare+forward (no generate loop)
        with torch.inference_mode():
            (_, position_ids, attention_mask, _, inputs_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, image_tensor,
                    modalities=[modality], image_sizes=image_sizes,
                )
            output = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
            )
            # Last hidden state's last token
            logits = output.logits[0, -1, :]  # (vocab,)

        target_logits = logits[target_tokens]
        pred_idx = target_logits.argmax().item()

        # GT
        if metric == "direction_word":
            # Preferred fields: direction (lowercase) → answer_text (capitalized) → answer_direction
            gt_raw = line.get("direction") or line.get("answer_direction") or line.get("answer_text", "")
            gt_direction = str(gt_raw).strip().capitalize()
            if gt_direction not in DIRECTION_WORDS:
                continue
            gt_idx = DIRECTION_WORDS.index(gt_direction)
        else:  # letter
            gt_letter = str(line.get("answer", "")).strip().upper()
            if gt_letter not in LETTERS:
                continue
            gt_idx = LETTERS.index(gt_letter)

        if pred_idx == gt_idx:
            correct += 1
        total += 1

    acc = correct / total * 100 if total > 0 else 0
    return acc, correct, total


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["layer", "module"], default="layer",
                        help="layer: layer-wise knockout | module: module-wise at critical layer")
    parser.add_argument("--layers", default="0-3,4-7,8-11,12-15,16-19,20-23,24-27",
                        help="comma-separated layer specs. "
                             "Each spec: single (e.g. '14') or range ('12-15'). "
                             "Examples: "
                             "'0,7,14,21' = single-layer knockouts | "
                             "'0-3,4-7,...' = window knockouts | "
                             "'14-20' = one multi-layer knockout")
    parser.add_argument("--critical_layer", type=int, default=14,
                        help="mode=module일 때 knockout할 layer")
    parser.add_argument("--tasks", default="shape_color,obj_place",
                        help="comma-separated tasks")
    parser.add_argument("--n_per_task", type=int, default=300)
    parser.add_argument("--metric", choices=["direction_word", "letter"], default="direction_word")
    parser.add_argument("--output", default="analysis/lora_knockout.json")
    args = parser.parse_args()

    tasks = args.tasks.split(",")

    # Load Vanilla on CPU (delta 계산용, 바로 free)
    print("Loading Vanilla (CPU)...")
    from core.model_loader import parse_model_args, load_model_from_args
    v_args = parse_model_args(VANILLA_ARGS_CPU)
    _, vanilla_model, _, _, _, _ = load_model_from_args(v_args)
    vanilla_model.eval()

    # Load Baseline on GPU
    print("Loading Baseline (GPU)...")
    b_args_str = f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS_GPU}"
    b_args = parse_model_args(b_args_str)
    tokenizer, baseline_model, image_processor, _, _, conv_template = load_model_from_args(b_args)
    baseline_model.eval()

    # Compute deltas
    print("Computing deltas...")
    deltas = compute_lora_deltas(vanilla_model, baseline_model)
    print(f"  {len(deltas)} parameters have non-zero delta")

    # Sanity: layer-wise delta count (7 modules expected per layer if all LoRA applied)
    from collections import Counter
    layer_counts = Counter(get_layer_idx(n) for n in deltas.keys())
    print(f"  Per-layer delta params: {dict(sorted(layer_counts.items()))}")
    expected_per_layer = 7  # q/k/v/o/gate/up/down
    if any(c != expected_per_layer for c in layer_counts.values()):
        print(f"  [WARN] Expected {expected_per_layer} deltas per layer but got irregular counts")

    # Free vanilla memory
    del vanilla_model
    gc.collect()
    torch.cuda.empty_cache()

    # Baseline reference accuracy per task (no knockout)
    print("\n[Baseline no-knockout baseline]")
    ref_acc = {}
    for task in tasks:
        acc, c, t = evaluate(baseline_model, tokenizer, image_processor, conv_template,
                              task, args.n_per_task, args.metric)
        ref_acc[task] = acc
        print(f"  {task}: {acc:.1f}% ({c}/{t})")

    # Run knockouts
    results = {"reference": ref_acc, "knockouts": {}}

    if args.mode == "layer":
        # Parse specs: single ("14") or range ("12-15")
        specs = []
        for spec in args.layers.split(","):
            spec = spec.strip()
            if "-" in spec:
                a, b = spec.split("-")
                specs.append((f"L{a}-{b}", list(range(int(a), int(b) + 1))))
            else:
                specs.append((f"L{spec}", [int(spec)]))
        print(f"\n[Layer-wise knockout] {len(specs)} spec(s):")
        for label, layers in specs:
            print(f"  {label}: layers {layers}")

        for label, layers in specs:
            applied = knockout_layers(baseline_model, deltas, layers)
            print(f"\n  Knockout {label}: {len(applied)} params zeroed")

            task_accs = {}
            for task in tasks:
                acc, c, t = evaluate(baseline_model, tokenizer, image_processor, conv_template,
                                      task, args.n_per_task, args.metric)
                drop = ref_acc[task] - acc
                task_accs[task] = {"acc": acc, "drop": drop}
                print(f"    {task}: {acc:.1f}% (Δ={drop:+.1f})")

            results["knockouts"][label] = {"layers": layers, "tasks": task_accs}
            restore(baseline_model, applied)

    elif args.mode == "module":
        L = args.critical_layer
        print(f"\n[Module-wise knockout at Layer {L}]")

        for module_suffix in TARGET_MODULE_SUFFIXES:
            applied = knockout_layers(baseline_model, deltas, [L], module_filter=[module_suffix])
            if not applied:
                print(f"  {module_suffix}: no deltas found, skipping")
                continue
            print(f"  {module_suffix}: {len(applied)} params")

            task_accs = {}
            for task in tasks:
                acc, c, t = evaluate(baseline_model, tokenizer, image_processor, conv_template,
                                      task, args.n_per_task, args.metric)
                drop = ref_acc[task] - acc
                task_accs[task] = {"acc": acc, "drop": drop}
                print(f"    {task}: {acc:.1f}% (Δ={drop:+.1f})")

            results["knockouts"][f"L{L}_{module_suffix}"] = task_accs
            restore(baseline_model, applied)

    # Summary
    print(f"\n{'='*60}\n  SUMMARY — mode={args.mode}, metric={args.metric}\n{'='*60}")
    print(f"  {'Knockout':>20} {'ref':>6}", end="")
    for task in tasks:
        print(f" {task[:12]:>12}", end="")
    print()
    for task in tasks:
        print(f"{'reference':>28} ", end="")
        print(f"{ref_acc[task]:>11.1f}%", end="")
    print()
    for key, entry in results["knockouts"].items():
        task_accs = entry if "acc" in next(iter(entry.values())) else entry["tasks"]
        print(f"  {key:>20}", end="")
        for task in tasks:
            acc = task_accs[task]["acc"]
            drop = task_accs[task]["drop"]
            print(f" {acc:>6.1f}({drop:+.1f})", end="")
        print()

    # Final sanity check: verify reference accuracy still works (weights fully restored)
    print("\n[Final Sanity Check] Re-evaluating after all knockouts (should match ref)...")
    sanity_pass = True
    for task in tasks:
        acc, c, t = evaluate(baseline_model, tokenizer, image_processor, conv_template,
                              task, min(args.n_per_task, 100), args.metric)
        diff = abs(acc - ref_acc[task])
        status = "OK" if diff < 2.0 else "⚠ MISMATCH"
        print(f"  {task}: ref={ref_acc[task]:.1f}% final={acc:.1f}% diff={diff:.1f} [{status}]")
        if diff > 2.0:
            sanity_pass = False

    if not sanity_pass:
        print("\n[WARNING] Restore inconsistency detected. Results may be unreliable.")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
