"""
Vision intervention v2: larger α sweep + Delta model control.

Tests whether:
  (a) Baseline responds to much larger α (magnitude issue?)
  (b) Delta responds to intervention (if yes, coadaptation hypothesis weaker)

V1 amplify:  v += α · Δ_d_model_OP  (own OP axis of the current model)
V2 align:    v += α · (Δ_d_model_IN - Δ_d_model_OP)  (shift OP → IN)
V_delta_replace: v += α · (Δ_d_DELTA_OP - Δ_d_BASELINE_OP)  (shift Baseline's dir toward Delta's dir)
"""
import argparse, os, sys, json, gc
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info"
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT")
os.environ["HF_HOME"] = "/data/datasets/LLaVA-Video-100K-Subset/"
os.environ["HF_DATASETS_CACHE"] = "/local_datasets/vlm_direction/"
torch.set_grad_enabled(False)

VIDEO_FOLDER = "/local_datasets/vlm_direction/"
VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"
DELTA_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

JSON_ROOT = os.path.join(_PROJECT_ROOT, "analysis/factorial_experiment/json")
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/vision_results")
AXES_DIR = os.path.join(_PROJECT_ROOT, "analysis/task_invariance")


def load_model_args(model_name):
    if model_name == "baseline":
        return f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    elif model_name == "delta":
        return f"lora_pretrained={DELTA_LORA},{VANILLA_ARGS}"
    raise ValueError(model_name)


def load_model_via_args(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
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
        out.append({"q_id": f"{q['id']}_v{q.get('variant_id',0)}",
                    "question": build_prompt(q), "answer": q["answer"],
                    "direction": q["direction"], "video": vp})
    return out


def load_axes(model_name):
    IN = np.load(f"{AXES_DIR}/vision_axes_{model_name}_IN.npz")
    OP = np.load(f"{AXES_DIR}/vision_axes_{model_name}_OP.npz")
    DIRS = ["up", "right", "down", "left"]
    return {d: torch.from_numpy(IN[f"Delta_{d}"]).float() for d in DIRS}, \
           {d: torch.from_numpy(OP[f"Delta_{d}"]).float() for d in DIRS}


@torch.no_grad()
def run(model_name, model, tokenizer, image_processor, conv_template, questions):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        "vis_v2", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())

    # Load this model's own axes + other model's axes for replace experiment
    IN_axes_self, OP_axes_self = load_axes(model_name)
    other_model = "delta" if model_name == "baseline" else "baseline"
    IN_axes_other, OP_axes_other = load_axes(other_model)

    mm_projector = model.model.mm_projector if hasattr(model.model, "mm_projector") else model.get_model().mm_projector
    state = {"mode": "off", "dir": None, "alpha": 0.0}

    def hook(module, inputs, output):
        if state["mode"] == "off": return output
        d = state["dir"]; alpha = state["alpha"]
        def compute_shift():
            if state["mode"] == "V1_amp":
                return alpha * OP_axes_self[d]
            elif state["mode"] == "V2_align":
                return alpha * (IN_axes_self[d] - OP_axes_self[d])
            elif state["mode"] == "V_cross":
                # shift own OP toward OTHER model's OP dir
                return alpha * (OP_axes_other[d] - OP_axes_self[d])
            return None
        shift = compute_shift()
        def modify(t):
            if t is None or t.dim() < 2: return t
            orig = t.dtype
            s = shift.to(t.device).float()
            t32 = t.float()
            if t.dim() == 3: t32 = t32 + s[None, None, :]
            else: t32 = t32 + s[None, :]
            return t32.to(orig)
        if isinstance(output, (list, tuple)):
            return type(output)(modify(t) for t in output)
        return modify(output)

    handle = mm_projector.register_forward_hook(hook)

    # Conditions
    conditions = [("no_swap", "off", 0.0)]
    for alpha in [1.0, 5.0, 20.0, 50.0]:
        conditions.append((f"V1_amp_a{alpha}", "V1_amp", alpha))
        conditions.append((f"V2_align_a{alpha}", "V2_align", alpha))
    conditions.append(("V_cross_a1.0", "V_cross", 1.0))
    conditions.append(("V_cross_a5.0", "V_cross", 5.0))

    stats = {c[0]: {"n":0, "correct":0} for c in conditions}

    def fwd_letter(input_ids, image_tensor, image_sizes, modality):
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
        state["dir"] = d
        try:
            for cname, mode, alpha in conditions:
                state["mode"] = mode; state["alpha"] = alpha
                pred = fwd_letter(input_ids, image_tensor, image_sizes, modality)
                stats[cname]["n"] += 1
                if pred == expected: stats[cname]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")
        state["mode"] = "off"

    handle.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["baseline", "delta"])
    ap.add_argument("--condition", default="obj_place")
    ap.add_argument("--n_target", type=int, default=300)
    ap.add_argument("--target_offset", type=int, default=0)
    args = ap.parse_args()

    qa_all = json.load(open(os.path.join(JSON_ROOT, f"{args.condition}_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[args.target_offset:args.target_offset + args.n_target]
    questions = build_questions(qa_target)
    print(f"[target] {len(qa_target)} samples")

    tokenizer, model, image_processor, _, _, conv_template = load_model_via_args(load_model_args(args.model))
    model.eval()

    stats = run(args.model, model, tokenizer, image_processor, conv_template, questions)

    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"\n=== {args.model}/{args.condition} (baseline={base:.1f}%) ===")
    for cond, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        delta = acc - base
        print(f"  {cond:>18s}: {acc:6.2f}%  Δ={delta:+5.2f}pp")

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, f"vision_v2_{args.model}_{args.condition}.json")
    json.dump({"model": args.model, "condition": args.condition, "n_target": len(qa_target), "stats": stats},
              open(out_path, "w"), indent=2)
    print(f"[SAVED] {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
