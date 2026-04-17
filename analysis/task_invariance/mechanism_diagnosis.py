"""
Mechanism diagnosis: last-token direction intervention at L14/L16/L18/L21.

Tests three hypotheses for binding gap:
  H-mag:   OOD direction signal magnitude is insufficient on canonical axis
  H-noise: Off-axis noise corrupts the binding readout
  H-layer: Direction needs canonical-axis alignment early (pre-binding)

Per OP sample, for each condition apply intervention at target layer L on last
token, continue forward, measure MCQ acc.

Conditions at L21 (unless noted):
  (1) no_swap
  (2) amp_2x    : h[-1] += 1.0 · ⟨h[-1]-g, Δ̂⟩ · Δ̂     (scale own on-axis by 2)
  (3) clean_sc  : h[-1] -= proj_OP · Δ̂ + mag_SC · Δ̂   (own proj replaced by SC magnitude)
  (4) add_canon : h[-1] += mag_SC · Δ̂                 (add SC magnitude, no removal)
  (5) on_axis   : h[-1] = g + proj_OP · Δ̂             (keep only on-axis; nuke off-axis)
  (6) full_rep  : h[-1] = h_avg_OP[d]                  (cond-e reference)
  (7) L18 clean_sc
  (8) L14 clean_sc
  (9) L16 clean_sc
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
torch.set_grad_enabled(False)

VIDEO_FOLDER = "/local_datasets/vlm_direction/"
VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"
JSON_ROOT = os.path.join(_PROJECT_ROOT, "analysis/factorial_experiment/json")
HIDDENS_ROOT = "/local_datasets/vlm_direction/factorial_dataset/hiddens"
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/mech_results")

# 28 factorial layers (L0..L27)
ALL_LAYERS = list(range(28))


def load_factorial(cond):
    files = sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz"))
    arr = {"hiddens": [], "directions": []}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in arr: arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


def compute_stats(data):
    """For each direction d and each layer L:
       Δ_d_L, Δ̂_d_L, mag_d_L, h_avg_d_L, g_L."""
    H = data["hiddens"].astype(np.float32)  # (N, 28, D)
    dirs = data["directions"]
    DIRS = ["up", "right", "down", "left"]
    h_avg_d = {d: H[dirs == d].mean(axis=0) for d in DIRS}  # each (28, D)
    g = H.mean(axis=0)  # (28, D)
    Delta = {d: h_avg_d[d] - g for d in DIRS}  # (28, D)
    Delta_hat = {d: Delta[d] / (np.linalg.norm(Delta[d], axis=1, keepdims=True) + 1e-9)
                 for d in DIRS}
    mag = {d: np.linalg.norm(Delta[d], axis=1) for d in DIRS}  # (28,)
    return dict(g=g, h_avg_d=h_avg_d, Delta=Delta, Delta_hat=Delta_hat, mag=mag, dirs=DIRS)


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


def make_hook(mode, d, L, OP_stats, SC_mag_L_d, h_avg_OP_L_d):
    """Return a forward hook for decoder_layer[L]."""
    g_L = torch.from_numpy(OP_stats["g"][L]).float()                      # (D,)
    Dhat_L = torch.from_numpy(OP_stats["Delta_hat"][d][L]).float()         # (D,)
    Delta_L = torch.from_numpy(OP_stats["Delta"][d][L]).float()            # (D,)
    mag_OP = float(OP_stats["mag"][d][L])
    mag_SC = float(SC_mag_L_d)
    h_avg = torch.from_numpy(h_avg_OP_L_d).float()                         # (D,)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        device, dt = h.device, h.dtype
        g = g_L.to(device, dt); dhat = Dhat_L.to(device, dt); Delta = Delta_L.to(device, dt)
        havg = h_avg.to(device, dt)
        last = h[:, -1, :].float()  # (B, D)
        centered = last - g.float()
        proj = (centered * dhat.float()).sum(dim=-1, keepdim=True)  # (B, 1)
        if mode == "amp_2x":
            last = last + proj * dhat.float()  # add one more projection → 2x on-axis
        elif mode == "clean_sc":
            # remove own projection, add SC-magnitude on canonical axis
            last = last - proj * dhat.float() + mag_SC * dhat.float()
        elif mode == "add_canon":
            last = last + mag_SC * dhat.float()
        elif mode == "on_axis":
            last = g.float() + proj * dhat.float()
        elif mode == "full_rep":
            last = havg.float()
        elif mode == "remove_own":
            last = last - proj * dhat.float()
        h = h.clone()
        h[:, -1, :] = last.to(dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    return hook


@torch.no_grad()
def run(model, tokenizer, image_processor, conv_template, questions,
        OP_stats, SC_stats):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        "mech", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())
    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    conditions = [
        # (name, layer, mode)
        ("no_swap", None, None),
        ("L21_amp_2x",   21, "amp_2x"),
        ("L21_clean_sc", 21, "clean_sc"),
        ("L21_add_canon",21, "add_canon"),
        ("L21_on_axis",  21, "on_axis"),
        ("L21_remove_own",21,"remove_own"),
        ("L21_full_rep", 21, "full_rep"),
        ("L18_clean_sc", 18, "clean_sc"),
        ("L16_clean_sc", 16, "clean_sc"),
        ("L14_clean_sc", 14, "clean_sc"),
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
        try:
            for cname, L, mode in conditions:
                if mode is None:
                    pred = fwd(input_ids, image_tensor, image_sizes, modality)
                else:
                    SC_mag = float(SC_stats["mag"][d][L])
                    h_avg_OP = OP_stats["h_avg_d"][d][L]
                    h = decoder_layers[L].register_forward_hook(
                        make_hook(mode, d, L, OP_stats, SC_mag, h_avg_OP))
                    try:
                        pred = fwd(input_ids, image_tensor, image_sizes, modality)
                    finally:
                        h.remove()
                stats[cname]["n"] += 1
                if pred == expected: stats[cname]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="obj_place")
    ap.add_argument("--n_target", type=int, default=500)
    ap.add_argument("--target_offset", type=int, default=0)
    args = ap.parse_args()

    print(f"[load OP factorial]")
    OP_data = load_factorial(args.condition)
    OP_stats = compute_stats(OP_data)

    print(f"[load SC factorial]")
    SC_data = load_factorial("shape_color")
    SC_stats = compute_stats(SC_data)

    # Magnitudes log
    print("\nDirection magnitude per layer (mean over 4 directions):")
    for L in [14, 16, 18, 21]:
        op_m = np.mean([OP_stats["mag"][d][L] for d in OP_stats["dirs"]])
        sc_m = np.mean([SC_stats["mag"][d][L] for d in SC_stats["dirs"]])
        print(f"  L{L}: OP={op_m:.3f}  SC={sc_m:.3f}  ratio SC/OP={sc_m/op_m:.2f}x")

    qa_all = json.load(open(os.path.join(JSON_ROOT, f"{args.condition}_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[args.target_offset:args.target_offset + args.n_target]
    print(f"\n[target] {len(qa_target)} samples variant_id=0")
    questions = build_questions(qa_target)

    tokenizer, model, image_processor, _, _, conv_template = load_model_lora()
    model.eval()

    stats = run(model, tokenizer, image_processor, conv_template, questions,
                 OP_stats, SC_stats)

    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"\n=== Mechanism diagnosis: {args.condition} (baseline={base:.2f}%) ===")
    for c, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>18s}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"condition": args.condition, "n_target": len(qa_target), "stats": stats},
              open(os.path.join(OUT_ROOT, f"mech_{args.condition}.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
