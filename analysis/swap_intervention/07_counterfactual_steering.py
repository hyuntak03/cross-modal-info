"""
Exp 1 — Counterfactual Steering (Kang §3 Alg.2 style)

Pure direction vector Δ(d) = h_avg(d) - h_avg(all_dirs)
  → identity components cancel in averaging across all dirs
  → Δ(d) isolates "direction=d" component

Steering at layer L (additive, norm-ish preserving):
  h_new = h_original - Δ(current_dir) + Δ(flip_dir)

Opposite flip:
  Up ↔ Down (letter A ↔ C)
  Left ↔ Right (letter D ↔ B)

Measurement: target letter shifts to flip direction's letter?
  Flip rate >> chance → binding reads direction linearly (identity-irrelevant)
  Flip rate ≈ chance → binding not linear in direction component

Contrast with Stage 1-a: there we REPLACED whole h (confounds identity).
  Here we ADD direction vector only, preserves target's identity info.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 07_counterfactual_steering.py \
      --model baseline --task shape_color --n_per_dir 200
"""

import argparse, os, sys, json, gc
import numpy as np
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
OUT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/steering_results"

DIRS = ["up", "right", "down", "left"]
DIR_TO_LETTER = {"up": "A", "right": "B", "down": "C", "left": "D"}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
LAYERS = list(range(15, 22))


def build_questions(task, n_per_dir, offset=0):
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
                "direction": d,
                "video": s["video"],
            })
    return qs


def compute_deltas(avg):
    """Δ(d)[L] = h_avg(d)[L] - h_avg(all)[L]"""
    deltas = {}
    for L in LAYERS:
        h_all = sum(avg[d][L] for d in DIRS) / len(DIRS)
        for d in DIRS:
            deltas.setdefault(d, {})[L] = (avg[d][L] - h_all).float()
    return deltas


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A", "B", "C", "D"]:
        for cand in [ltr, " " + ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]; break
    return ids


def make_additive_hook(sub_vec, add_vec):
    """output[0][:, -1, :] ← x - sub + add (direction swap, norm-preserving-ish)."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] = h[:, -1, :] - sub_vec.to(h.device, h.dtype) + add_vec.to(h.device, h.dtype)
            return (h,) + output[1:]
        else:
            output[:, -1, :] = output[:, -1, :] - sub_vec.to(output.device, output.dtype) + add_vec.to(output.device, output.dtype)
            return output
    return hook


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, task, deltas, n_per_dir, offset):
    from core.data_pipeline import create_data_loader
    questions = build_questions(task, n_per_dir, offset=offset)
    print(f"[{task}] {len(questions)} samples (2x forward per sample)")
    dl = create_data_loader(
        questions, "", 1, 8, tokenizer, image_processor, model.config,
        f"vlm_direction_testbed_R2R_4way_1500_{task}",
        conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())
    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    # Track per-direction stats
    stats = {d: {"n": 0, "orig_correct": 0, "flip_to_target": 0,
                 "flip_unchanged": 0, "flip_to_other": 0}
             for d in DIRS}
    samples_log = []

    def forward_once(input_ids, image_tensor, image_sizes, modality, hooks):
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
        sub = logits[letter_tok_ids]
        return id_to_letter[letter_tok_ids[int(sub.argmax())]]

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc=task, leave=True):
        if batch is None:
            continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]

        cur_dir = line["direction"]
        flip_dir = OPPOSITE[cur_dir]
        expected_orig = DIR_TO_LETTER[cur_dir]
        expected_flip = DIR_TO_LETTER[flip_dir]

        try:
            # 1) No-steering baseline
            pred_orig = forward_once(input_ids, image_tensor, image_sizes, modality, [])

            # 2) Counterfactual: replace direction component
            hooks = []
            for L in LAYERS:
                sub_vec = deltas[cur_dir][L]
                add_vec = deltas[flip_dir][L]
                h = decoder_layers[L].register_forward_hook(make_additive_hook(sub_vec, add_vec))
                hooks.append(h)
            try:
                pred_flip = forward_once(input_ids, image_tensor, image_sizes, modality, hooks)
            finally:
                for h in hooks: h.remove()

        except Exception as e:
            print(f"[WARN] {line['q_id']}: {e}")
            continue

        stats[cur_dir]["n"] += 1
        if pred_orig == expected_orig:
            stats[cur_dir]["orig_correct"] += 1
        if pred_flip == expected_flip:
            stats[cur_dir]["flip_to_target"] += 1
        elif pred_flip == expected_orig:
            stats[cur_dir]["flip_unchanged"] += 1
        else:
            stats[cur_dir]["flip_to_other"] += 1
        samples_log.append({
            "q_id": line["q_id"], "dir": cur_dir, "flip_dir": flip_dir,
            "expected_orig": expected_orig, "expected_flip": expected_flip,
            "pred_orig": pred_orig, "pred_flip": pred_flip,
        })

    return stats, samples_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "baseline"])
    ap.add_argument("--task", required=True, choices=["shape_color", "obj_color", "shape_place", "obj_place"])
    ap.add_argument("--n_per_dir", type=int, default=200)
    ap.add_argument("--offset", type=int, default=200, help="skip extraction set range")
    ap.add_argument("--delta_source_task", default=None,
                    help="task to load avg_hidden from (default: same as --task)")
    args = ap.parse_args()

    src_task = args.delta_source_task or args.task
    avg_path = os.path.join(AVG_ROOT, f"{args.model}_{src_task}.pt")
    src = torch.load(avg_path, map_location="cpu", weights_only=False)
    deltas = compute_deltas(src["avg"])
    print(f"[delta] from {avg_path}")

    args_str = VANILLA_ARGS if args.model == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    print(f"[load] {args.model}")
    tokenizer, model, image_processor, _, _, conv_template = load_model(args_str)
    model.eval()

    stats, samples = run(model, tokenizer, image_processor, conv_template,
                          args.task, deltas, args.n_per_dir, args.offset)

    # Aggregate
    n_tot = sum(v["n"] for v in stats.values())
    orig_acc = sum(v["orig_correct"] for v in stats.values()) / max(n_tot, 1) * 100
    flip_rate = sum(v["flip_to_target"] for v in stats.values()) / max(n_tot, 1) * 100
    unchanged = sum(v["flip_unchanged"] for v in stats.values()) / max(n_tot, 1) * 100
    other = sum(v["flip_to_other"] for v in stats.values()) / max(n_tot, 1) * 100

    result = {
        "model": args.model, "task": args.task,
        "delta_source_task": src_task,
        "layers": LAYERS,
        "n_total": n_tot,
        "orig_acc": orig_acc,
        "flip_to_target_rate": flip_rate,
        "flip_unchanged_rate": unchanged,
        "flip_to_other_rate": other,
        "per_direction": stats,
        "samples": samples[:50],  # save first 50 sample logs for inspection
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    tag = f"{args.model}_{args.task}"
    if src_task != args.task:
        tag += f"_from-{src_task}"
    out_path = os.path.join(OUT_ROOT, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    print(f"  orig_acc          : {orig_acc:.1f}%")
    print(f"  flip→target_rate  : {flip_rate:.1f}%   (chance 25%)")
    print(f"  flip unchanged    : {unchanged:.1f}%")
    print(f"  flip→other        : {other:.1f}%")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
