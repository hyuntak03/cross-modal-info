"""
Step 1-1: Per-identity averaged hidden 추출.

각 (identity, direction)별 N sample forward → mean last token hidden L=15..21.
Output: h_avg[identity][direction][L] tensor.

This averaging:
  - Removes background instance variation (within same identity-direction pool)
  - Preserves identity (banana)
  - Preserves direction (Up)

Compare with Stage 1-a's across-identity averaging:
  - Stage 1-a removes both instance noise AND identity (mixed identity pool)

Usage:
  CUDA_VISIBLE_DEVICES=0 python 11_extract_per_identity.py \
      --model baseline --task obj_place --n_per_cell 30
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
OUT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/per_identity_avg"

DIRS = ["up", "right", "down", "left"]
LAYERS = list(range(15, 22))


def load_canonical(task):
    return json.load(open(os.path.join(CANON_ROOT, f"{task}.json")))


def load_metadata(task):
    return json.load(open(os.path.join(META_ROOT, f"{task}_metadata.json")))


def get_identity_key(task):
    return "obj_class" if "obj" in task else "shape"


def select_extraction_samples(task, n_per_cell):
    """Return list of (sample_dict, identity) for extraction."""
    canon = load_canonical(task)
    canon_by_id = {s["id"]: s for s in canon}
    meta = load_metadata(task)
    id_key = get_identity_key(task)

    by_id_dir = defaultdict(lambda: defaultdict(list))
    for m in meta:
        by_id_dir[m[id_key]][m["direction"]].append(m["id"])

    selected = []  # (canonical_sample, identity, direction)
    for ident in sorted(by_id_dir.keys()):
        for d in DIRS:
            ids_in_cell = sorted(by_id_dir[ident][d])[:n_per_cell]
            for sid in ids_in_cell:
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
            "answer": s["answer"],
            "direction": d,
            "identity": ident,
            "video": s["video"],
        })
    return qs


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


@torch.no_grad()
def extract(model, tokenizer, image_processor, conv_template, task, samples):
    from core.data_pipeline import create_data_loader
    questions = build_questions(samples)
    print(f"[{task}] {len(questions)} samples ({len(set((q['identity'], q['direction']) for q in questions))} cells)")
    dl = create_data_loader(
        questions, "", 1, 8, tokenizer, image_processor, model.config,
        f"vlm_direction_testbed_R2R_4way_1500_{task}",
        conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    counts = defaultdict(lambda: defaultdict(int))

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc=task, leave=True):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]

        try:
            (_, position_ids, attention_mask, _, inputs_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, image_tensor,
                    modalities=[modality], image_sizes=image_sizes,
                )
            output = model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, output_hidden_states=True, return_dict=True,
            )
        except Exception as e:
            print(f"[WARN] {line['q_id']}: {e}")
            continue

        ident = line["identity"]; d = line["direction"]
        for L in LAYERS:
            h = output.hidden_states[L + 1][0, -1, :].detach().to(torch.float32).cpu()
            if sums[ident][d][L] is None:
                sums[ident][d][L] = h.clone()
            else:
                sums[ident][d][L] += h
        counts[ident][d] += 1
        del output
        if (sum(counts[ident].values())) % 50 == 0:
            torch.cuda.empty_cache()

    avg = {}
    for ident in sums:
        avg[ident] = {}
        for d in DIRS:
            if counts[ident][d] > 0:
                avg[ident][d] = {L: sums[ident][d][L] / counts[ident][d] for L in LAYERS}
    return avg, dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "baseline"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_per_cell", type=int, default=30)
    args = ap.parse_args()

    samples = select_extraction_samples(args.task, args.n_per_cell)
    print(f"Selected {len(samples)} samples for extraction")

    args_str = VANILLA_ARGS if args.model == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    print(f"[load] {args.model}")
    tokenizer, model, image_processor, _, _, conv_template = load_model(args_str)
    model.eval()

    avg, counts = extract(model, tokenizer, image_processor, conv_template, args.task, samples)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"{args.model}_{args.task}.pt")
    # Convert to dict-of-dict-of-dict tensor for save
    torch.save({"avg": avg, "counts": dict(counts), "layers": LAYERS}, out_path)
    print(f"\n[SAVED] {out_path}")
    print(f"  identities: {len(avg)}")
    sample_id = list(avg.keys())[0]
    print(f"  sample {sample_id}: {[(d, list(avg[sample_id].get(d, {}).keys())[:1]) for d in DIRS]}")
    print(f"  counts (first 3 ids):")
    for i, ident in enumerate(list(counts.keys())[:3]):
        print(f"    {ident}: {dict(counts[ident])}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
