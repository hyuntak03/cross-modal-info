"""
Exp 2 — Per-sample last token hidden extraction (no averaging).

Saves each sample's last token hidden at L=15..27 + MCQ prediction + direction label.

Used for:
  - Per-sample alignment with readout axes (Eq.8 sample-level test)
  - Variance analysis: how much each sample deviates from its direction mean
  - Correlation between alignment and MCQ correctness

Usage:
  CUDA_VISIBLE_DEVICES=0 python 06_extract_per_sample.py \
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
OUT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/per_sample"

DIRS = ["up", "right", "down", "left"]
DIR_TO_LETTER = {"up": "A", "right": "B", "down": "C", "left": "D"}
LAYERS = list(range(15, 28))  # 15..27


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
                "answer": s["answer"],
                "direction": d,
                "video": s["video"],
            })
    return qs


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


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, task, n_per_dir, offset):
    from core.data_pipeline import create_data_loader
    questions = build_questions(task, n_per_dir, offset=offset)
    print(f"[{task}] {len(questions)} samples")
    dl = create_data_loader(
        questions, "", 1, 8, tokenizer, image_processor, model.config,
        f"vlm_direction_testbed_R2R_4way_1500_{task}",
        conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())

    # Storage
    n_total = len(questions)
    hidden_dim = model.config.hidden_size
    hiddens = np.zeros((n_total, len(LAYERS), hidden_dim), dtype=np.float16)
    directions = []
    expected_letters = []
    pred_letters = []
    correctness = []
    q_ids = []
    idx = 0

    for batch, line in tqdm(zip(dl, questions), total=n_total, desc=task, leave=True):
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
            out = model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, output_hidden_states=True, return_dict=True,
            )
        except Exception as e:
            print(f"[WARN] {line['q_id']}: {e}")
            continue

        for i, L in enumerate(LAYERS):
            hiddens[idx, i, :] = out.hidden_states[L + 1][0, -1, :].detach().to(torch.float16).cpu().numpy()

        logits = out.logits[0, -1, :]
        sub = logits[letter_tok_ids]
        pred_letter = id_to_letter[letter_tok_ids[int(sub.argmax())]]

        directions.append(line["direction"])
        expected_letters.append(DIR_TO_LETTER[line["direction"]])
        pred_letters.append(pred_letter)
        correctness.append(int(pred_letter == DIR_TO_LETTER[line["direction"]]))
        q_ids.append(line["q_id"])
        idx += 1

        del out
        if idx % 50 == 0:
            torch.cuda.empty_cache()

    return {
        "hiddens": hiddens[:idx],
        "directions": np.array(directions),
        "expected": np.array(expected_letters),
        "pred": np.array(pred_letters),
        "correct": np.array(correctness),
        "q_ids": np.array(q_ids),
        "layers": LAYERS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "baseline"])
    ap.add_argument("--task", required=True, choices=["shape_color", "obj_color", "shape_place", "obj_place"])
    ap.add_argument("--n_per_dir", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    args_str = VANILLA_ARGS if args.model == "vanilla" else f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    print(f"[load] {args.model}")
    tokenizer, model, image_processor, _, _, conv_template = load_model(args_str)
    model.eval()

    data = run(model, tokenizer, image_processor, conv_template, args.task, args.n_per_dir, args.offset)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"{args.model}_{args.task}.npz")
    np.savez_compressed(out_path,
                        hiddens=data["hiddens"],
                        directions=data["directions"],
                        expected=data["expected"],
                        pred=data["pred"],
                        correct=data["correct"],
                        q_ids=data["q_ids"],
                        layers=np.array(data["layers"]))
    acc = data["correct"].mean() * 100
    print(f"[SAVED] {out_path}  n={len(data['correct'])}  acc={acc:.1f}%")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
