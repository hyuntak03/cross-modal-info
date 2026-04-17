"""
Propagation test — does a magnitude boost at layer L propagate downstream?

For each intervention layer L ∈ {14, 16, 18, 20}:
  1. Clean_sc at L on own local axis (sets L on-axis magnitude to SC's L magnitude)
  2. Capture downstream hidden at L+2, L+4, L21
  3. Measure on-axis magnitude at each capture (on local axis + on L21 canonical axis)
  4. Measure final MCQ

This disambiguates:
  - L14 clean_sc = 0pp MCQ. Is it because:
    (a) boost doesn't reach L16 (binding) - check L16 magnitude
    (b) reaches L16 but binding circuit doesn't respond - check L16 letter-readable
    (c) reaches L16 and binding responds but doesn't reach L21 canonical - check L21 magnitude
    (d) reaches L21 but lm_head doesn't read - unlikely given final MCQ 0pp and
        L21 canonical = 1.00 self-cos
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

INTERVENTION_LAYERS = [14, 16, 18, 20]
CAPTURE_LAYERS = [14, 16, 18, 20, 21, 24]


def load_factorial(cond):
    arr = {"hiddens": [], "directions": []}
    for f in sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz")):
        d = np.load(f, allow_pickle=True)
        for k in arr: arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


def compute_stats_all_layers(data):
    H = data["hiddens"].astype(np.float32)  # (N, 28, D)
    dirs = data["directions"]
    g = H.mean(0)
    out = {}
    for dn in ["up", "right", "down", "left"]:
        h_avg = H[dirs == dn].mean(0)
        Delta = h_avg - g  # (28, D)
        mag = np.linalg.norm(Delta, axis=1)  # (28,)
        hat = Delta / (mag[:, None] + 1e-9)  # (28, D)
        out[dn] = {"Delta": Delta, "hat": hat, "mag": mag}
    return g, out


def load_model():
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}")
    return load_model_from_args(a)


def get_letter_ids(tok):
    ids = {}
    for ltr in ["A","B","C","D"]:
        for c in [ltr, " " + ltr]:
            tids = tok.encode(c, add_special_tokens=False)
            if len(tids) == 1: ids[ltr] = tids[0]; break
    return ids


def build_q(q):
    t = q["question"] + "\n"
    for i, opt in enumerate(q["candidates"]):
        t += f"{chr(ord('A')+i)}. {opt}\n"
    return t + "Answer with the option letter only."


def build_questions(qa):
    out = []
    for q in qa:
        vp = q["video"]
        if vp.startswith(VIDEO_FOLDER): vp = vp[len(VIDEO_FOLDER):]
        out.append({"q_id": f"{q['id']}_v{q.get('variant_id',0)}",
                    "question": build_q(q), "answer": q["answer"],
                    "direction": q["direction"], "video": vp})
    return out


def make_intervention_hook(L_int, d, g_op, stats_op, stats_sc):
    """Hook at layer L_int: clean_sc on own local axis, set to SC's L_int magnitude."""
    g_L = torch.from_numpy(g_op[L_int]).float()
    hat = torch.from_numpy(stats_op[d]["hat"][L_int]).float()
    mag_sc = float(stats_sc[d]["mag"][L_int])

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        dev, dt = h.device, h.dtype
        last = h[:, -1, :].float()
        gl = g_L.to(dev); hl = hat.to(dev)
        proj = ((last - gl) * hl).sum(dim=-1, keepdim=True)
        last_new = last - proj * hl + mag_sc * hl
        h = h.clone()
        h[:, -1, :] = last_new.to(dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_capture_hook(L_cap, storage):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        storage[L_cap] = h[:, -1, :].detach().cpu().float().numpy()[0]  # (D,)
        return output
    return hook


@torch.no_grad()
def run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(questions, "", 1, 4, tok, ip, model.config,
                             "prop", ct, video_folder=VIDEO_FOLDER, video_fps=1,
                             frames_upbound=8, force_sample=True)
    lid = get_letter_ids(tok)
    id2l = {v: k for k, v in lid.items()}
    ltids = list(lid.values())
    decoder = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    # Conditions: no_swap + 4 intervention layers
    conditions = [("no_swap", None)] + [(f"L{L}_clean", L) for L in INTERVENTION_LAYERS]

    # Results: per condition, per capture layer, collect magnitudes
    results = {c[0]: {f"L{L}_local_mag": [] for L in CAPTURE_LAYERS} for c in conditions}
    for c in conditions:
        results[c[0]].update({f"L{L}_canon_mag": [] for L in CAPTURE_LAYERS})
    mcq = {c[0]: {"n":0, "correct":0} for c in conditions}

    def fwd(input_ids, image_tensor, image_sizes, modality):
        (_, pos, am, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, image_tensor,
            modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=emb, attention_mask=am, position_ids=pos, return_dict=True)
        return out.logits[0, -1, :]

    for batch, line in tqdm(zip(dl, questions), total=len(questions)):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d = line["direction"]
        expected = line["answer"]
        try:
            for cname, L_int in conditions:
                capture = {}
                hooks = []
                # Register capture hooks
                for L_cap in CAPTURE_LAYERS:
                    hooks.append(decoder[L_cap].register_forward_hook(make_capture_hook(L_cap, capture)))
                # Register intervention hook
                if L_int is not None:
                    hooks.append(decoder[L_int].register_forward_hook(
                        make_intervention_hook(L_int, d, g_op, stats_op, stats_sc)))
                try:
                    logits = fwd(input_ids, image_tensor, image_sizes, modality)
                finally:
                    for h in hooks: h.remove()

                # Measure pred
                pred = id2l[ltids[int(logits[ltids].argmax())]]
                mcq[cname]["n"] += 1
                if pred == expected: mcq[cname]["correct"] += 1

                # Measure magnitudes at each capture
                for L_cap in CAPTURE_LAYERS:
                    h_vec = capture[L_cap]  # (D,)
                    # Local axis magnitude at L_cap
                    hat_local = stats_op[d]["hat"][L_cap]
                    g_local = g_op[L_cap]
                    proj_local = float((h_vec - g_local) @ hat_local)
                    # Canonical (L21) axis magnitude
                    hat_canon = stats_op[d]["hat"][21]
                    proj_canon = float((h_vec - g_op[21]) @ hat_canon)
                    results[cname][f"L{L_cap}_local_mag"].append(proj_local)
                    results[cname][f"L{L_cap}_canon_mag"].append(proj_canon)
        except Exception as e:
            print(f"[ERR] {line['q_id']}: {e}")

    return mcq, results


def main():
    print("[load factorial OP]")
    OP = load_factorial("obj_place")
    g_op, stats_op = compute_stats_all_layers(OP)

    print("[load factorial SC]")
    SC = load_factorial("shape_color")
    _, stats_sc = compute_stats_all_layers(SC)

    print(f"\nSC mean magnitudes at intervention layers:")
    for L in INTERVENTION_LAYERS:
        print(f"  L{L}: SC_mag = {np.mean([stats_sc[d]['mag'][L] for d in ['up','right','down','left']]):.3f}  "
              f"OP_mag = {np.mean([stats_op[d]['mag'][L] for d in ['up','right','down','left']]):.3f}")

    qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[:200]  # smaller for speed
    questions = build_questions(qa_target)

    tok, model, ip, _, _, ct = load_model()
    model.eval()

    mcq, res = run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc)

    print(f"\n=== MCQ acc per condition ===")
    base = mcq["no_swap"]["correct"]/max(mcq["no_swap"]["n"],1)*100
    for c, s in mcq.items():
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>14}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    print(f"\n=== LOCAL axis magnitude propagation (mean over samples) ===")
    print(f"{'Cond':>14} | " + " | ".join(f"L{L}_loc" for L in CAPTURE_LAYERS))
    for c in mcq:
        row = f"{c:>14} |"
        for L in CAPTURE_LAYERS:
            vals = res[c][f"L{L}_local_mag"]
            row += f" {np.mean(vals):>6.2f} |"
        print(row)

    print(f"\n=== CANONICAL (L21) axis magnitude propagation (mean over samples) ===")
    print(f"{'Cond':>14} | " + " | ".join(f"L{L}_can" for L in CAPTURE_LAYERS))
    for c in mcq:
        row = f"{c:>14} |"
        for L in CAPTURE_LAYERS:
            vals = res[c][f"L{L}_canon_mag"]
            row += f" {np.mean(vals):>6.2f} |"
        print(row)

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"mcq": mcq, "magnitudes": {c: {k: [float(x) for x in v] for k,v in r.items()} for c,r in res.items()}},
              open(os.path.join(OUT_ROOT, "propagation.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
