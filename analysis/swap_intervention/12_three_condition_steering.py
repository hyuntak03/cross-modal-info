"""
Step 1-2: 3-condition steering.

For each target obj_place sample (identity=o, direction=d, expected letter=L):
  (a) No swap: baseline forward
  (b) Within-identity avg swap: h_avg(o, d) at L=15..21
  (c) Across-identity avg swap (Stage 1-a): h_avg(any identity, d) at L=15..21

Decompose:
  direction_noise_effect = (b) - (a)         ← removing within-(id,dir) instance noise
  identity_removal_effect = (c) - (b)        ← removing identity (banana → "anything")
  combined = (c) - (a)

Usage:
  CUDA_VISIBLE_DEVICES=0 python 12_three_condition_steering.py \
      --model baseline --task obj_place --n_per_cell 20 --offset 30
"""

import argparse, os, sys, json, gc
import numpy as np
import torch
from collections import defaultdict
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
META_ROOT = "/local_datasets/vlm_direction/vlm_direction_testbed/R2R_video_1500"
WITHIN_AVG_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/per_identity_avg"
ACROSS_AVG_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/avg_hidden"
OUT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/three_condition_results"

DIRS = ["up", "right", "down", "left"]
DIR_TO_LETTER = {"up": "A", "right": "B", "down": "C", "left": "D"}
LAYERS = list(range(15, 22))


def get_identity_key(task):
    return "obj_class" if "obj" in task else "shape"


def select_target_samples(task, n_per_cell, offset):
    canon = json.load(open(os.path.join(CANON_ROOT, f"{task}.json")))
    canon_by_id = {s["id"]: s for s in canon}
    meta = json.load(open(os.path.join(META_ROOT, f"{task}_metadata.json")))
    id_key = get_identity_key(task)

    by_id_dir = defaultdict(lambda: defaultdict(list))
    for m in meta:
        by_id_dir[m[id_key]][m["direction"]].append(m["id"])

    selected = []
    for ident in sorted(by_id_dir.keys()):
        for d in DIRS:
            ids = sorted(by_id_dir[ident][d])[offset:offset + n_per_cell]
            for sid in ids:
                if sid in canon_by_id:
                    selected.append((canon_by_id[sid], ident, d))
    return selected


def build_questions(samples):
    qs = []
    for s, ident, d in samples:
        cand = s["candidates"]
        prompt_text = s["question"] + "\n"
        for i, opt in enumerate(cand):
            prompt_text += f"{chr(ord('A') + i)}. {opt}\n"
        prompt_text += "Answer with the option letter only."
        qs.append({
            "q_id": f"{s['id']}_{d}",
            "question": prompt_text,
            "direction": d,
            "identity": ident,
            "video": s["video"],
        })
    return qs


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


def make_hook(replacement_vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] = replacement_vec.to(h.device, h.dtype)
            return (h,) + output[1:]
        return output
    return hook


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A", "B", "C", "D"]:
        for cand in [ltr, " " + ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]; break
    return ids


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, task,
        within_avg, across_avg, n_per_cell, offset):
    from core.data_pipeline import create_data_loader
    samples = select_target_samples(task, n_per_cell, offset)
    questions = build_questions(samples)
    print(f"[{task}] {len(questions)} target samples")
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

    # Result accumulators per (identity, direction) per condition
    stats = defaultdict(lambda: {"a": {"n": 0, "correct": 0},
                                  "b": {"n": 0, "correct": 0},
                                  "c": {"n": 0, "correct": 0}})

    def forward_letter(input_ids, image_tensor, image_sizes, modality, hooks):
        try:
            for h in hooks: pass  # noop, just ensuring hooks parameter exists
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
        except Exception as e:
            print(f"[ERR] {e}"); return None

    def hooked_forward(input_ids, image_tensor, image_sizes, modality, vec_dict):
        # vec_dict: {L: tensor}
        hooks = []
        for L in LAYERS:
            h = decoder_layers[L].register_forward_hook(make_hook(vec_dict[L]))
            hooks.append(h)
        try:
            return forward_letter(input_ids, image_tensor, image_sizes, modality, hooks)
        finally:
            for h in hooks: h.remove()

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc=task, leave=True):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]

        ident = line["identity"]; d = line["direction"]
        expected = DIR_TO_LETTER[d]
        key = (ident, d)

        # (a) No swap
        pred_a = forward_letter(input_ids, image_tensor, image_sizes, modality, [])
        if pred_a is not None:
            stats[key]["a"]["n"] += 1
            if pred_a == expected: stats[key]["a"]["correct"] += 1

        # (b) Within-identity avg swap
        if ident in within_avg and d in within_avg[ident]:
            vec_dict = within_avg[ident][d]
            pred_b = hooked_forward(input_ids, image_tensor, image_sizes, modality, vec_dict)
            if pred_b is not None:
                stats[key]["b"]["n"] += 1
                if pred_b == expected: stats[key]["b"]["correct"] += 1

        # (c) Across-identity avg swap
        if d in across_avg:
            vec_dict = across_avg[d]
            pred_c = hooked_forward(input_ids, image_tensor, image_sizes, modality, vec_dict)
            if pred_c is not None:
                stats[key]["c"]["n"] += 1
                if pred_c == expected: stats[key]["c"]["correct"] += 1

    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "baseline"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_per_cell", type=int, default=20)
    ap.add_argument("--offset", type=int, default=30)
    args = ap.parse_args()

    # Load within-identity avg
    within_path = os.path.join(WITHIN_AVG_ROOT, f"{args.model}_{args.task}.pt")
    within = torch.load(within_path, map_location="cpu", weights_only=False)["avg"]
    print(f"[within-id avg] {within_path}: {len(within)} identities")

    # Load across-identity avg (Stage 1-a)
    across_path = os.path.join(ACROSS_AVG_ROOT, f"{args.model}_{args.task}.pt")
    across = torch.load(across_path, map_location="cpu", weights_only=False)["avg"]
    print(f"[across-id avg] {across_path}: {list(across.keys())} dirs")

    args_str = VANILLA_ARGS if args.model == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    print(f"[load model] {args.model}")
    tokenizer, model, image_processor, _, _, conv_template = load_model(args_str)
    model.eval()

    stats = run(model, tokenizer, image_processor, conv_template,
                 args.task, within, across, args.n_per_cell, args.offset)

    # Aggregate
    def acc(cond_key):
        n = sum(stats[k][cond_key]["n"] for k in stats)
        c = sum(stats[k][cond_key]["correct"] for k in stats)
        return c / n * 100 if n > 0 else 0.0

    a_acc = acc("a"); b_acc = acc("b"); c_acc = acc("c")
    print(f"\n=== Aggregate ===")
    print(f"  (a) no-swap                 : {a_acc:.1f}%")
    print(f"  (b) within-identity avg swap: {b_acc:.1f}%")
    print(f"  (c) across-identity avg swap: {c_acc:.1f}%")
    print(f"  direction_noise_effect (b-a): {b_acc - a_acc:+.1f}%p")
    print(f"  identity_removal_effect (c-b): {c_acc - b_acc:+.1f}%p")
    print(f"  combined            (c-a)   : {c_acc - a_acc:+.1f}%p")

    # Per-direction breakdown
    print(f"\n=== Per direction ===")
    for d in DIRS:
        cells = {k: v for k, v in stats.items() if k[1] == d}
        n = sum(c["a"]["n"] for c in cells.values())
        if n == 0: continue
        a = sum(c["a"]["correct"] for c in cells.values()) / n * 100
        nb = sum(c["b"]["n"] for c in cells.values())
        b = sum(c["b"]["correct"] for c in cells.values()) / max(nb, 1) * 100
        nc = sum(c["c"]["n"] for c in cells.values())
        c = sum(c["c"]["correct"] for c in cells.values()) / max(nc, 1) * 100
        print(f"  {d:>6s}: (a) {a:5.1f}%  (b) {b:5.1f}%  (c) {c:5.1f}%   "
              f"b-a={b-a:+.1f}  c-b={c-b:+.1f}")

    # Save
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"{args.model}_{args.task}.json")
    save_stats = {f"{ident}__{d}": s for (ident, d), s in stats.items()}
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task,
            "aggregate": {"a": a_acc, "b": b_acc, "c": c_acc,
                          "direction_noise_effect": b_acc - a_acc,
                          "identity_removal_effect": c_acc - b_acc,
                          "combined": c_acc - a_acc},
            "per_cell": save_stats,
        }, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
