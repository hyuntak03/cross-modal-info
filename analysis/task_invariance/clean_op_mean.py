"""
Clean to OP's own mean magnitude at L21 — isolates per-sample variance reduction
WITHOUT magnitude boost.

For each OP sample with direction d:
  proj_own = ⟨h[-1] - g, Δ̂_d⟩   (sample's own on-axis projection)
  h[-1] -= proj_own · Δ̂_d       (remove own)
  h[-1] += mag_OP · Δ̂_d         (set to OP's mean magnitude — NO boost)

Comparison:
  clean_op_mean: set to OP mean (28)   — pure noise reduction
  clean_sc_mean: set to SC mean (48)   — noise reduction + magnitude boost (known +10pp)
  add_canon:     add SC mean (76)      — magnitude peak (known +11.6pp)

If clean_op ≈ clean_sc → noise dominant
If clean_op << clean_sc → magnitude dominant
"""
import os, sys, json, gc, glob
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
HIDDENS_ROOT = "/local_datasets/vlm_direction/factorial_dataset/hiddens"
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/mech_results")
L_TARGET = 21


def load_factorial(cond):
    arr = {"hiddens": [], "directions": []}
    for f in sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz")):
        d = np.load(f, allow_pickle=True)
        for k in arr: arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


def compute_stats(data, L):
    H = data["hiddens"].astype(np.float32)
    dirs = data["directions"]
    g = H.mean(0)[L]
    stats = {}
    for d in ["up", "right", "down", "left"]:
        avg = H[dirs == d].mean(0)[L]
        Delta = avg - g
        mag = np.linalg.norm(Delta)
        stats[d] = {"Delta_hat": Delta / (mag + 1e-9), "mag": float(mag), "h_avg": avg}
    return g, stats


def load_model():
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}")
    return load_model_from_args(a)


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A", "B", "C", "D"]:
        for cand in [ltr, " " + ltr]:
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


def make_hook(mode, d, g_L, stats_OP, mag_SC_d):
    g = torch.from_numpy(g_L).float()
    dhat = torch.from_numpy(stats_OP[d]["Delta_hat"]).float()
    mag_OP = stats_OP[d]["mag"]

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        dev, dt = h.device, h.dtype
        last = h[:, -1, :].float()
        centered = last - g.to(dev).float()
        proj = (centered * dhat.to(dev).float()).sum(dim=-1, keepdim=True)
        if mode == "clean_op_mean":
            new_mag = mag_OP
            last = last - proj * dhat.to(dev).float() + new_mag * dhat.to(dev).float()
        elif mode == "clean_sc_mean":
            new_mag = mag_SC_d
            last = last - proj * dhat.to(dev).float() + new_mag * dhat.to(dev).float()
        elif mode == "clean_op_half":
            new_mag = mag_OP * 0.5  # reduce magnitude below OP mean
            last = last - proj * dhat.to(dev).float() + new_mag * dhat.to(dev).float()
        elif mode == "clean_2x_sc":
            new_mag = mag_SC_d * 2.0  # 2x SC magnitude
            last = last - proj * dhat.to(dev).float() + new_mag * dhat.to(dev).float()
        h = h.clone()
        h[:, -1, :] = last.to(dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, questions,
        g_L, stats_OP, stats_SC):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        "cleanop", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())
    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    conditions = [
        ("no_swap", None),
        ("clean_op_half", "clean_op_half"),   # below mean
        ("clean_op_mean", "clean_op_mean"),   # OP mean (no boost)
        ("clean_sc_mean", "clean_sc_mean"),   # SC mean (known +10pp)
        ("clean_2x_sc", "clean_2x_sc"),        # 2x SC (over-boost)
    ]
    stats_out = {c[0]: {"n": 0, "correct": 0} for c in conditions}

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
        try:
            for cname, mode in conditions:
                if mode is None:
                    pred = fwd(input_ids, image_tensor, image_sizes, modality)
                else:
                    h = decoder_layers[L_TARGET].register_forward_hook(
                        make_hook(mode, d, g_L, stats_OP, stats_SC[d]["mag"]))
                    try:
                        pred = fwd(input_ids, image_tensor, image_sizes, modality)
                    finally:
                        h.remove()
                stats_out[cname]["n"] += 1
                if pred == expected: stats_out[cname]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")

    return stats_out


def main():
    print("[load factorial OP]")
    OP_data = load_factorial("obj_place")
    g_L_OP, stats_OP = compute_stats(OP_data, L_TARGET)

    print("[load factorial SC]")
    SC_data = load_factorial("shape_color")
    _, stats_SC = compute_stats(SC_data, L_TARGET)

    print(f"\nMagnitudes at L{L_TARGET}:")
    for d in stats_OP:
        print(f"  {d}: OP_mean={stats_OP[d]['mag']:.2f}  SC_mean={stats_SC[d]['mag']:.2f}")

    qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[:500]
    questions = build_questions(qa_target)

    tokenizer, model, image_processor, _, _, conv_template = load_model()
    model.eval()

    stats = run(model, tokenizer, image_processor, conv_template, questions,
                 g_L_OP, stats_OP, stats_SC)

    base = stats["no_swap"]["correct"] / max(stats["no_swap"]["n"], 1) * 100
    print(f"\n=== clean to different magnitudes at L21: obj_place (base={base:.2f}%) ===")
    for c, s in stats.items():
        acc = s["correct"] / max(s["n"], 1) * 100
        print(f"  {c:>18s}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"n_target": len(qa_target), "stats": stats, "L": L_TARGET},
              open(os.path.join(OUT_ROOT, "clean_op_mean.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
