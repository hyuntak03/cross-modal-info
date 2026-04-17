"""
Stage 1-a-2: Window swap (L15-L21) with averaged source hidden.

For each target sample:
  1. Compute target's expected letter (from canonical: up=A, right=B, down=C, left=D)
  2. Forward with hooks on layers 15..21 that replace last token hidden with
     averaged source hidden (matched by direction)
  3. Measure: predicted letter == expected?

Source pool selection:
  - in_domain  : source_task == target_task (same task)
  - cross_domain: source_task != target_task

Also runs no-swap baseline (no hook) for comparison.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 03_swap_experiment.py \
      --model baseline --target_task shape_color --source_task shape_color \
      --pair_type in_domain --n_targets 100
"""

import argparse
import gc
import json
import os
import sys

import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT")
os.environ["HF_HOME"] = "/data/datasets/LLaVA-Video-100K-Subset/"
os.environ["HF_DATASETS_CACHE"] = "/local_datasets/vlm_direction/"

VIDEO_FOLDER = "/local_datasets/vlm_direction/"

VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

CANON_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/canonical_R2R"
AVG_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/avg_hidden"
OUT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/swap_results"

DIRS = ["up", "right", "down", "left"]
DIR_TO_LETTER = {"up": "A", "right": "B", "down": "C", "left": "D"}
LAYERS = list(range(15, 22))


def build_questions(task, n_per_dir, offset=0):
    """Use samples *after* offset (avoid overlap with extraction set)."""
    with open(os.path.join(CANON_ROOT, f"{task}.json")) as f:
        data = json.load(f)
    by_dir = {d: [] for d in DIRS}
    for s in data:
        by_dir[s["direction"]].append(s)
    qs = []
    for d in DIRS:
        for s in by_dir[d][offset:offset + n_per_dir]:
            cand = s["candidates"]
            prompt_text = s["question"] + "\n"
            for i, opt in enumerate(cand):
                prompt_text += f"{chr(ord('A') + i)}. {opt}\n"
            prompt_text += "Answer with the option letter only."
            qs.append({
                "q_id": f"{s['id']}_{d}",
                "question": prompt_text,
                "answer": s["answer"],
                "direction": d,
                "video": s["video"],
            })
    return qs


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


def make_hook(replacement_vec):
    """forward_hook that replaces output[0][:, -1, :] with replacement_vec."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] = replacement_vec.to(h.device, h.dtype)
            return (h,) + output[1:]
        else:
            output[:, -1, :] = replacement_vec.to(output.device, output.dtype)
            return output
    return hook


def get_letter_token_ids(tokenizer):
    ids = {}
    for ltr in ["A", "B", "C", "D"]:
        for cand in [ltr, " " + ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]
                break
    return ids


@torch.no_grad()
def run_target(model, tokenizer, image_processor, conv_template, target_task, source_avg, n_per_dir, offset, do_swap):
    from core.data_pipeline import create_data_loader
    questions = build_questions(target_task, n_per_dir, offset=offset)
    dl = create_data_loader(
        questions, "", 1, 8, tokenizer, image_processor, model.config,
        f"vlm_direction_testbed_R2R_4way_1500_{target_task}",
        conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    letter_ids = get_letter_token_ids(tokenizer)  # {"A": 32, ...}
    letter_id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_token_ids = list(letter_ids.values())

    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    per_dir = {d: {"n": 0, "correct": 0} for d in DIRS}
    samples = []

    for batch, line in tqdm(zip(dl, questions), total=len(questions),
                              desc=f"swap={do_swap}", leave=True):
        if batch is None:
            continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d = line["direction"]
        expected_letter = DIR_TO_LETTER[d]

        hooks = []
        if do_swap:
            for l in LAYERS:
                vec = source_avg[d][l]
                h = decoder_layers[l].register_forward_hook(make_hook(vec))
                hooks.append(h)

        try:
            (_, position_ids, attention_mask, _, inputs_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, image_tensor,
                    modalities=[modality], image_sizes=image_sizes,
                )
            out = model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, return_dict=True,
            )
            logits = out.logits[0, -1, :]
            # restrict to letter ids
            sub = logits[letter_token_ids]
            pred_letter = letter_id_to_letter[letter_token_ids[int(sub.argmax())]]
        except Exception as e:
            print(f"[WARN] {line['q_id']}: {e}")
            for h in hooks: h.remove()
            continue
        finally:
            for h in hooks: h.remove()

        per_dir[d]["n"] += 1
        if pred_letter == expected_letter:
            per_dir[d]["correct"] += 1
        samples.append({"q_id": line["q_id"], "dir": d,
                        "expected": expected_letter, "pred": pred_letter})

    return per_dir, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "baseline"])
    ap.add_argument("--target_task", required=True)
    ap.add_argument("--source_task", required=True,
                    help="task to load averaged hidden from; same as target → in_domain")
    ap.add_argument("--n_targets", type=int, default=100, help="samples per direction")
    ap.add_argument("--target_offset", type=int, default=200,
                    help="skip first N samples per dir (avoid extraction overlap)")
    args = ap.parse_args()

    pair_type = "in_domain" if args.source_task == args.target_task else "cross_domain"

    # Load source averaged hidden
    src_path = os.path.join(AVG_ROOT, f"{args.model}_{args.source_task}.pt")
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    source_avg = src["avg"]
    print(f"[load source avg] {src_path}  counts={src['counts']}")

    # Load model
    args_str = VANILLA_ARGS if args.model == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    print(f"[load model] {args.model}")
    tokenizer, model, image_processor, _, _, conv_template = load_model(args_str)
    model.eval()

    # No-swap baseline
    print("\n--- no-swap baseline ---")
    base_dir, base_samples = run_target(
        model, tokenizer, image_processor, conv_template,
        args.target_task, source_avg, args.n_targets, args.target_offset, do_swap=False,
    )

    # With-swap
    print("\n--- with swap (window L15-L21) ---")
    sw_dir, sw_samples = run_target(
        model, tokenizer, image_processor, conv_template,
        args.target_task, source_avg, args.n_targets, args.target_offset, do_swap=True,
    )

    def acc(per_dir):
        n = sum(v["n"] for v in per_dir.values())
        c = sum(v["correct"] for v in per_dir.values())
        return c / n * 100 if n > 0 else 0.0

    result = {
        "model": args.model, "target_task": args.target_task,
        "source_task": args.source_task, "pair_type": pair_type,
        "layers": LAYERS,
        "no_swap": {"per_dir": base_dir, "acc": acc(base_dir)},
        "swap":    {"per_dir": sw_dir,   "acc": acc(sw_dir)},
        "n_targets_per_dir": args.n_targets,
        "target_offset": args.target_offset,
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    tag = f"{args.model}_src-{args.source_task}_tgt-{args.target_task}_{pair_type}"
    out_path = os.path.join(OUT_ROOT, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    print(f"  no-swap acc: {result['no_swap']['acc']:.1f}%")
    print(f"  swap    acc: {result['swap']['acc']:.1f}%  Δ={result['swap']['acc']-result['no_swap']['acc']:+.1f}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
