"""
Vision-level magnitude amplification (analogue to L21 amp_2x but at projector).

Previous V1 was broadcast-ADD of Δ_d (uniform shift). Tiny perturbation.
This test: per-position SCALE of on-axis component → true amplification.

For each position p in projector output:
  proj_p = ⟨v[p], Δ̂_d⟩
  v[p] += (k-1) · proj_p · Δ̂_d    (scales on-axis by k)

Also tests: amplify along SC (IN) axis vs OP's own axis.

Conditions:
  no_swap
  V_amp_own_2x, _5x, _10x       (amplify on OP's own vision axis)
  V_amp_in_2x,  _5x              (amplify on IN/SC's vision axis)
  V_clean_sc                     (set on-axis projection to SC magnitude × proj_p direction)
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
AXES_DIR = os.path.join(_PROJECT_ROOT, "analysis/task_invariance")
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/vision_amp_results")


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
    t = q["question"] + "\n"
    for i, opt in enumerate(q["candidates"]):
        t += f"{chr(ord('A')+i)}. {opt}\n"
    t += "Answer with the option letter only."
    return t


def build_questions(qa):
    out = []
    for q in qa:
        vp = q["video"]
        if vp.startswith(VIDEO_FOLDER): vp = vp[len(VIDEO_FOLDER):]
        out.append({"q_id": f"{q['id']}_v{q.get('variant_id',0)}",
                    "question": build_prompt(q), "answer": q["answer"],
                    "direction": q["direction"], "video": vp})
    return out


def load_vision_axes():
    """Load pre-computed vision-level axes (3584-dim)."""
    IN = np.load(f"{AXES_DIR}/vision_axes_baseline_IN.npz")
    OP = np.load(f"{AXES_DIR}/vision_axes_baseline_OP.npz")
    DIRS = ["up", "right", "down", "left"]
    IN_axes = {d: torch.from_numpy(IN[f"Delta_{d}"]).float() for d in DIRS}
    OP_axes = {d: torch.from_numpy(OP[f"Delta_{d}"]).float() for d in DIRS}
    IN_hat = {d: v/(v.norm()+1e-9) for d,v in IN_axes.items()}
    OP_hat = {d: v/(v.norm()+1e-9) for d,v in OP_axes.items()}
    IN_mag = {d: float(v.norm()) for d,v in IN_axes.items()}
    OP_mag = {d: float(v.norm()) for d,v in OP_axes.items()}
    return IN_hat, OP_hat, IN_mag, OP_mag


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, questions,
        IN_hat, OP_hat, IN_mag, OP_mag):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        "vamp", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())

    mm_projector = model.model.mm_projector if hasattr(model.model, "mm_projector") else model.get_model().mm_projector
    state = {"mode": "off", "dir": None, "k": 1.0}

    def hook(module, inputs, output):
        if state["mode"] == "off": return output
        d = state["dir"]; k = state["k"]
        def modify(t):
            if t is None or t.dim() < 2: return t
            orig_dtype = t.dtype
            t32 = t.float()
            # Select axis
            if state["mode"].startswith("amp_own"):
                axis = OP_hat[d].to(t.device).float()
            elif state["mode"].startswith("amp_in"):
                axis = IN_hat[d].to(t.device).float()
            elif state["mode"] == "clean_sc":
                # replace on-axis projection with SC-mean magnitude along OWN axis
                axis = OP_hat[d].to(t.device).float()
                target_mag = IN_mag[d] / OP_mag[d]  # ratio
            elif state["mode"] == "push_in_mag":
                # scale on-axis to reach IN's magnitude, along OP axis (preserves axis)
                axis = OP_hat[d].to(t.device).float()
            # Per-position projection
            if t.dim() == 3:
                proj = (t32 * axis[None, None, :]).sum(dim=-1, keepdim=True)  # (B, N, 1)
                if state["mode"] == "clean_sc":
                    # v[p] = v[p] - proj · axis + (IN_mag_scale · proj) · axis_IN
                    axis_in = IN_hat[d].to(t.device).float()
                    t32 = t32 - proj * axis[None, None, :] + (IN_mag[d]/OP_mag[d]) * proj * axis_in[None, None, :]
                elif state["mode"] == "push_in_mag":
                    # scale projection to reach SC magnitude
                    scale = IN_mag[d] / OP_mag[d]
                    t32 = t32 + (scale - 1.0) * proj * axis[None, None, :]
                else:  # amp modes
                    t32 = t32 + (k - 1.0) * proj * axis[None, None, :]
            else:
                proj = (t32 * axis[None, :]).sum(dim=-1, keepdim=True)
                t32 = t32 + (k - 1.0) * proj * axis[None, :]
            return t32.to(orig_dtype)
        if isinstance(output, (list, tuple)):
            return type(output)(modify(t) for t in output)
        return modify(output)

    handle = mm_projector.register_forward_hook(hook)

    conditions = [
        ("no_swap", "off", 1.0),
        ("amp_own_2x", "amp_own", 2.0),
        ("amp_own_5x", "amp_own", 5.0),
        ("amp_own_10x", "amp_own", 10.0),
        ("amp_in_2x", "amp_in", 2.0),
        ("amp_in_5x", "amp_in", 5.0),
        ("push_in_mag", "push_in_mag", 1.0),
        ("clean_sc", "clean_sc", 1.0),
    ]
    stats = {c[0]: {"n":0, "correct":0} for c in conditions}

    def fwd(input_ids, image_tensor, image_sizes, modality):
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
            for cname, mode, k in conditions:
                state["mode"] = mode; state["k"] = k
                pred = fwd(input_ids, image_tensor, image_sizes, modality)
                stats[cname]["n"] += 1
                if pred == expected: stats[cname]["correct"] += 1
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

    IN_hat, OP_hat, IN_mag, OP_mag = load_vision_axes()
    print(f"Vision-level axis magnitudes:")
    for d in ["up","right","down","left"]:
        print(f"  {d}: OP={OP_mag[d]:.3f}  IN={IN_mag[d]:.3f}  ratio={IN_mag[d]/OP_mag[d]:.2f}x")

    qa_all = json.load(open(os.path.join(JSON_ROOT, f"{args.condition}_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[args.target_offset:args.target_offset + args.n_target]
    questions = build_questions(qa_target)
    print(f"\n[target] {len(qa_target)} samples")

    tokenizer, model, image_processor, _, _, conv_template = load_model_lora()
    model.eval()

    stats = run(model, tokenizer, image_processor, conv_template, questions,
                 IN_hat, OP_hat, IN_mag, OP_mag)

    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"\n=== Vision amp: {args.condition} (baseline={base:.2f}%) ===")
    for c, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>14s}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"condition": args.condition, "n_target": len(qa_target), "stats": stats},
              open(os.path.join(OUT_ROOT, f"vision_amp_{args.condition}.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
