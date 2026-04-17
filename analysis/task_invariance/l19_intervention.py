"""
L19 single-layer direction averaging intervention.

Tests the hypothesis: L19 is the direction amplifier. If we inject averaged
direction prototype at L19 (replacing last token), does OOD recover?

Compare: no-swap vs L19-only swap vs L20-only swap vs (L19+L20) swap.

Reuses factorial hidden cache. Δ_d = h_avg(direction=d, over 4 MCQ variants)
from factorial dataset.

Usage:
  CUDA_VISIBLE_DEVICES=0 python l19_intervention.py --condition obj_place --n_target 500
"""
import argparse, os, sys, json, gc, glob
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info"
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT")
os.environ["HF_HOME"] = "/data/datasets/LLaVA-Video-100K-Subset/"
os.environ["HF_DATASETS_CACHE"] = "/local_datasets/vlm_direction/"

VIDEO_FOLDER = "/local_datasets/vlm_direction/"
VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

JSON_ROOT = os.path.join(_PROJECT_ROOT, "analysis/factorial_experiment/json")
HIDDENS_ROOT = "/local_datasets/vlm_direction/factorial_dataset/hiddens"
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/l19_results")

LAYERS_TO_TEST = [19, 20, 21, [19, 20], [18, 19, 20], [19, 20, 21]]


def load_hiddens(condition):
    pattern = os.path.join(HIDDENS_ROOT, f"baseline_{condition}_4variants*.npz")
    files = sorted(glob.glob(pattern))
    arrays = {"hiddens": [], "directions": []}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in arrays: arrays[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arrays.items()}


def compute_dir_avg(data):
    """h_avg(d) per direction for all 28 layers."""
    H = data["hiddens"].astype(np.float32)  # (N, 28, D)
    dirs = data["directions"]
    avg = {}
    for d in np.unique(dirs):
        avg[d] = H[dirs == d].mean(axis=0)  # (28, D)
    return avg


def load_model_lora():
    from core.model_loader import parse_model_args, load_model_from_args
    args_str = f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    a = parse_model_args(args_str)
    return load_model_from_args(a)


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A","B","C","D"]:
        for cand in [ltr, " "+ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]; break
    return ids


def build_prompt(q):
    text = q["question"] + "\n"
    for i, opt in enumerate(q["candidates"]):
        text += f"{chr(ord('A')+i)}. {opt}\n"
    text += "Answer with the option letter only."
    return text


def build_questions(qa_list):
    out = []
    for q in qa_list:
        vp = q["video"]
        if vp.startswith(VIDEO_FOLDER): vp = vp[len(VIDEO_FOLDER):]
        out.append({
            "q_id": f"{q['id']}_v{q.get('variant_id',0)}",
            "question": build_prompt(q),
            "answer": q["answer"],
            "direction": q["direction"],
            "video": vp,
        })
    return out


def make_replace_hook(target_vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] = target_vec.to(h.device, h.dtype)
            return (h,) + output[1:]
        output[:, -1, :] = target_vec.to(output.device, output.dtype)
        return output
    return hook


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, questions, dir_avg_tensors):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        "l19_intervention", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())
    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    conditions = ["no_swap"]
    for L in LAYERS_TO_TEST:
        if isinstance(L, int):
            conditions.append(f"L{L}")
        else:
            conditions.append("L" + "-".join(str(l) for l in L))
    stats = {c: {"n": 0, "correct": 0} for c in conditions}

    def forward_letter(input_ids, image_tensor, image_sizes, modality):
        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor,
                modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                      position_ids=position_ids, return_dict=True)
        logits = out.logits[0, -1, :]
        return id_to_letter[letter_tok_ids[int(logits[letter_tok_ids].argmax())]]

    for batch, line in tqdm(zip(dl, questions), total=len(questions)):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d = line["direction"]
        expected = line["answer"]

        try:
            # No swap
            pred = forward_letter(input_ids, image_tensor, image_sizes, modality)
            stats["no_swap"]["n"] += 1
            if pred == expected: stats["no_swap"]["correct"] += 1

            # Each layer condition
            for L in LAYERS_TO_TEST:
                L_list = [L] if isinstance(L, int) else L
                cond_name = f"L{L}" if isinstance(L, int) else "L" + "-".join(str(l) for l in L)
                hooks = []
                for Lk in L_list:
                    h = decoder_layers[Lk].register_forward_hook(
                        make_replace_hook(dir_avg_tensors[d][Lk]))
                    hooks.append(h)
                try:
                    pred = forward_letter(input_ids, image_tensor, image_sizes, modality)
                finally:
                    for h in hooks: h.remove()
                stats[cond_name]["n"] += 1
                if pred == expected: stats[cond_name]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="obj_place", choices=["obj_place", "shape_color"])
    ap.add_argument("--n_target", type=int, default=500)
    ap.add_argument("--target_offset", type=int, default=0)
    args = ap.parse_args()

    print(f"[load hiddens] {args.condition}")
    data = load_hiddens(args.condition)
    dir_avg = compute_dir_avg(data)  # {d: (28, D)}
    dir_avg_tensors = {d: torch.from_numpy(arr).float() for d, arr in dir_avg.items()}

    qa_all = json.load(open(os.path.join(JSON_ROOT, f"{args.condition}_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[args.target_offset:args.target_offset + args.n_target]
    print(f"[target] {len(qa_target)} samples from {args.condition} variant_id=0")

    questions = build_questions(qa_target)
    tokenizer, model, image_processor, _, _, conv_template = load_model_lora()
    model.eval()

    stats = run(model, tokenizer, image_processor, conv_template, questions, dir_avg_tensors)

    print(f"\n=== L19 intervention sweep: {args.condition} ===")
    for cond, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {cond:>15s}: {acc:.2f}%  (n={s['n']})")

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"l19_{args.condition}.json")
    json.dump({"condition": args.condition, "n_target": len(qa_target), "stats": stats},
              open(out_path, "w"), indent=2)
    print(f"[SAVED] {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
