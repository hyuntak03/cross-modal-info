"""
Vision-level direction intervention experiments.

Hypotheses:
  V1 (amplify): OOD direction signal magnitude ↑ → better binding
       v[p] += α · Δ̂_d_OP_vision (own direction axis, amplify)
  V2 (align): shift OOD direction onto IN axis → Delta-like effect
       v[p] += α · (Δ_d_IN - Δ_d_OP) (vision-level alignment)
  V3 (axis swap): remove OP axis component, add IN axis component
       v[p] -= ⟨v[p], Δ̂_d_OP⟩·Δ̂_d_OP + ⟨v[p], Δ̂_d_IN⟩·Δ̂_d_IN

Hook mm_projector output, apply broadcast shift/projection to each vision token position.

Usage:
  CUDA_VISIBLE_DEVICES=0 python vision_intervention.py --n_target 500
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

JSON_ROOT = os.path.join(_PROJECT_ROOT, "analysis/factorial_experiment/json")
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/vision_results")
AXES_PATH = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/vision_axes.npz")


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


def load_axes():
    d = np.load(AXES_PATH)
    DIRS = ["up", "right", "down", "left"]
    Delta_IN = {k: torch.from_numpy(d[f"Delta_IN_{k}"]).float() for k in DIRS}
    Delta_OP = {k: torch.from_numpy(d[f"Delta_OP_{k}"]).float() for k in DIRS}
    # Unit vectors
    Delta_IN_hat = {k: v/(v.norm()+1e-9) for k,v in Delta_IN.items()}
    Delta_OP_hat = {k: v/(v.norm()+1e-9) for k,v in Delta_OP.items()}
    return Delta_IN, Delta_OP, Delta_IN_hat, Delta_OP_hat


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, questions,
        Delta_IN, Delta_OP, Delta_IN_hat, Delta_OP_hat):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        "vision_intervention", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())

    mm_projector = model.model.mm_projector if hasattr(model.model, "mm_projector") else model.get_model().mm_projector
    state = {"mode": "off", "dir": None}

    def hook(module, inputs, output):
        if state["mode"] == "off": return output
        d = state["dir"]
        # output can be tuple or tensor
        def modify(t):
            if t is None or t.dim() < 2: return t
            orig_dtype = t.dtype
            t32 = t.float()
            delta_IN = Delta_IN[d].to(t.device)
            delta_OP = Delta_OP[d].to(t.device)
            hat_IN = Delta_IN_hat[d].to(t.device)
            hat_OP = Delta_OP_hat[d].to(t.device)
            alpha = state["alpha"]
            if state["mode"] == "V1_amplify":
                # amplify own direction: v += α · Δ̂_OP · ||Δ_OP||
                shift = alpha * delta_OP
                if t.dim() == 3: t32 = t32 + shift[None, None, :]
                else: t32 = t32 + shift[None, :]
            elif state["mode"] == "V2_align":
                # shift toward IN axis
                shift = alpha * (delta_IN - delta_OP)
                if t.dim() == 3: t32 = t32 + shift[None, None, :]
                else: t32 = t32 + shift[None, :]
            elif state["mode"] == "V3_axis_swap":
                # remove on-axis OP component, add on-axis IN component
                # (keep spatial structure by only modifying direction-axis subspace)
                # For each position p, compute ⟨v[p], Δ̂_OP⟩
                # Replace with ⟨v[p], Δ̂_OP⟩·(some target IN-axis projection)
                if t.dim() == 3:  # (B, N, D)
                    proj_OP = (t32 * hat_OP[None, None, :]).sum(dim=-1, keepdim=True)  # (B, N, 1)
                    # Remove OP axis, add IN axis with same magnitude scaled by alpha
                    t32 = t32 - proj_OP * hat_OP[None, None, :] + alpha * proj_OP * hat_IN[None, None, :]
                else:
                    proj_OP = (t32 * hat_OP[None, :]).sum(dim=-1, keepdim=True)
                    t32 = t32 - proj_OP * hat_OP[None, :] + alpha * proj_OP * hat_IN[None, :]
            return t32.to(orig_dtype)
        if isinstance(output, (list, tuple)):
            return type(output)(modify(t) for t in output)
        return modify(output)

    handle = mm_projector.register_forward_hook(hook)

    # Conditions to test
    conditions = [
        ("no_swap", "off", 0.0),
        ("V1_amp_a0.5", "V1_amplify", 0.5),
        ("V1_amp_a1.0", "V1_amplify", 1.0),
        ("V1_amp_a2.0", "V1_amplify", 2.0),
        ("V2_align_a0.5", "V2_align", 0.5),
        ("V2_align_a1.0", "V2_align", 1.0),
        ("V2_align_a2.0", "V2_align", 2.0),
        ("V3_swap_a1.0", "V3_axis_swap", 1.0),
    ]
    stats = {c[0]: {"n":0, "correct":0} for c in conditions}

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
        state["dir"] = d

        try:
            for cond_name, mode, alpha in conditions:
                state["mode"] = mode
                state["alpha"] = alpha
                pred = forward_letter(input_ids, image_tensor, image_sizes, modality)
                stats[cond_name]["n"] += 1
                if pred == expected: stats[cond_name]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")
        state["mode"] = "off"

    handle.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="obj_place")
    ap.add_argument("--n_target", type=int, default=500)
    ap.add_argument("--target_offset", type=int, default=0)
    args = ap.parse_args()

    Delta_IN, Delta_OP, Delta_IN_hat, Delta_OP_hat = load_axes()

    qa_all = json.load(open(os.path.join(JSON_ROOT, f"{args.condition}_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[args.target_offset:args.target_offset + args.n_target]
    print(f"[target] {len(qa_target)} samples from {args.condition} variant_id=0")
    questions = build_questions(qa_target)

    tokenizer, model, image_processor, _, _, conv_template = load_model_lora()
    model.eval()

    stats = run(model, tokenizer, image_processor, conv_template, questions,
                 Delta_IN, Delta_OP, Delta_IN_hat, Delta_OP_hat)

    print(f"\n=== Vision intervention: {args.condition} ===")
    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"{'Condition':>20s} | {'acc':>7s} | {'Δ vs no_swap':>15s}")
    print("-"*52)
    for cond, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        delta = acc - base
        print(f"{cond:>20s} | {acc:>6.2f}% | {delta:>+14.2f}%p")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"condition": args.condition, "n_target": len(qa_target), "stats": stats},
              open(os.path.join(OUT_ROOT, f"vision_intervention_{args.condition}.json"), "w"), indent=2)
    print(f"[SAVED]")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
