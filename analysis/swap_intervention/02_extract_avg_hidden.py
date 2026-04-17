"""
Stage 1-a-1: (model, task, direction)별 last token hidden을 N sample 평균.

각 layer L=15..21에서:
  h_avg(model, task, direction, L) = mean_i h_L(sample_i)[last_token]

이 averaged hidden이 swap source로 사용됨 (Kang Eq.3 style).

Usage:
  CUDA_VISIBLE_DEVICES=0 python 02_extract_avg_hidden.py \
      --model vanilla --task shape_color --n_per_dir 100
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
OUT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/avg_hidden"

DIRS = ["up", "right", "down", "left"]
LAYERS = list(range(15, 22))  # L15..L21


def build_questions(task, n_per_dir):
    """Canonical JSON → questions list (per-dir balanced sampling)."""
    with open(os.path.join(CANON_ROOT, f"{task}.json")) as f:
        data = json.load(f)
    by_dir = {d: [] for d in DIRS}
    for s in data:
        by_dir[s["direction"]].append(s)
    qs = []
    for d in DIRS:
        for s in by_dir[d][:n_per_dir]:
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


@torch.no_grad()
def extract(model, tokenizer, image_processor, conv_template, task, n_per_dir):
    from core.data_pipeline import create_data_loader
    questions = build_questions(task, n_per_dir)
    print(f"[{task}] {len(questions)} samples ({n_per_dir}/dir)")
    dl = create_data_loader(
        questions, "", 1, 8, tokenizer, image_processor, model.config,
        f"vlm_direction_testbed_R2R_4way_1500_{task}",
        conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    sums = {d: {l: None for l in LAYERS} for d in DIRS}
    counts = {d: 0 for d in DIRS}

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc=task, leave=True):
        if batch is None:
            continue
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
            print(f"[WARN] forward 실패 q_id={line['q_id']}: {e}")
            continue

        d = line["direction"]
        for l in LAYERS:
            # hidden_states[l+1] = output of decoder layer l (0-indexed)
            h = output.hidden_states[l + 1][0, -1, :].detach().to(torch.float32).cpu()
            sums[d][l] = h if sums[d][l] is None else sums[d][l] + h
        counts[d] += 1

        del output
        if counts[d] % 20 == 0:
            torch.cuda.empty_cache()

    avg = {d: {l: (sums[d][l] / counts[d]) for l in LAYERS} for d in DIRS if counts[d] > 0}
    return avg, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "baseline"])
    ap.add_argument("--task", required=True, choices=["shape_color", "obj_color", "shape_place", "obj_place"])
    ap.add_argument("--n_per_dir", type=int, default=100)
    args = ap.parse_args()

    args_str = VANILLA_ARGS if args.model == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    print(f"[load] {args.model}")
    tokenizer, model, image_processor, _, _, conv_template = load_model(args_str)
    model.eval()

    avg, counts = extract(model, tokenizer, image_processor, conv_template, args.task, args.n_per_dir)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"{args.model}_{args.task}.pt")
    torch.save({"avg": avg, "counts": counts, "layers": LAYERS}, out_path)
    print(f"[SAVED] {out_path}  counts={counts}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
