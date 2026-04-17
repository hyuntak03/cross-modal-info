"""
Combined BMS + REF test with fast GPU probe.

Tests both:
  (1) Binding-stage intervention: L14/L15/L16/L17 clean_sc → L16 letter probe
  (2) Refinement-stage intervention: L21 clean_sc/clean_2x_sc → L22-L27 letter probe
  (3) Additive: L14+L21 combined

Saves hiddens first (recoverable), then fast GPU probe.
"""
import os, sys, json, gc, glob
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_probe import gpu_probe

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
HID_CACHE = "/local_datasets/vlm_direction/combined_intervention_hiddens"

CAPTURE_LAYERS = [14, 15, 16, 17, 18, 21, 22, 24, 26, 27]

# Conditions: (name, list of (layer, magnitude_multiplier))
CONDITIONS = [
    ("no_swap", []),
    ("L14", [(14, 1.0)]),
    ("L15", [(15, 1.0)]),
    ("L16", [(16, 1.0)]),
    ("L17", [(17, 1.0)]),
    ("L14_15_16", [(14, 1.0), (15, 1.0), (16, 1.0)]),
    ("L21_sc", [(21, 1.0)]),
    ("L21_2xsc", [(21, 2.0)]),
    ("L14_plus_L21_2xsc", [(14, 1.0), (21, 2.0)]),
]


def load_factorial(cond):
    arr = {"hiddens": [], "directions": []}
    for f in sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz")):
        d = np.load(f, allow_pickle=True)
        for k in arr: arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


def compute_stats(data):
    H = data["hiddens"].astype(np.float32)
    dirs = data["directions"]
    g = H.mean(0)
    out = {}
    for dn in ["up", "right", "down", "left"]:
        h_avg = H[dirs == dn].mean(0)
        Delta = h_avg - g
        mag = np.linalg.norm(Delta, axis=1)
        hat = Delta / (mag[:, None] + 1e-9)
        out[dn] = {"hat": hat, "mag": mag}
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


def build_prompt(q):
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
                    "question": build_prompt(q), "answer": q["answer"],
                    "direction": q["direction"],
                    "variant_id": q.get("variant_id", 0), "video": vp})
    return out


def make_int_hook(L_int, d, g_op, stats_op, stats_sc, mag_mult=1.0):
    g_L = torch.from_numpy(g_op[L_int]).float()
    hat = torch.from_numpy(stats_op[d]["hat"][L_int]).float()
    mag_target = float(stats_sc[d]["mag"][L_int]) * mag_mult
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        dev, dt = h.device, h.dtype
        last = h[:, -1, :].float()
        gl = g_L.to(dev); hl = hat.to(dev)
        proj = ((last - gl) * hl).sum(dim=-1, keepdim=True)
        last_new = last - proj * hl + mag_target * hl
        h = h.clone()
        h[:, -1, :] = last_new.to(dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_cap_hook(L, storage):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        storage[L] = h[:, -1, :].detach().cpu().float().numpy()[0]
        return output
    return hook


@torch.no_grad()
def run_forward(model, tok, ip, ct, questions, g_op, stats_op, stats_sc):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(questions, "", 1, 4, tok, ip, model.config,
                             "comb", ct, video_folder=VIDEO_FOLDER, video_fps=1,
                             frames_upbound=8, force_sample=True)
    lid = get_letter_ids(tok); id2l = {v:k for k,v in lid.items()}
    ltids = list(lid.values())
    decoder = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    state = {c[0]: {"hid": {L: [] for L in CAPTURE_LAYERS},
                     "letter": [], "dir": [], "pred": []} for c in CONDITIONS}

    def fwd(input_ids, image_tensor, image_sizes, modality):
        (_, pos, am, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, image_tensor,
            modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=emb, attention_mask=am, position_ids=pos, return_dict=True)
        return out.logits[0, -1, :]

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc="forward"):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d = line["direction"]; expected = line["answer"]

        for cname, int_list in CONDITIONS:
            capture = {}; hooks = []
            for L_cap in CAPTURE_LAYERS:
                hooks.append(decoder[L_cap].register_forward_hook(make_cap_hook(L_cap, capture)))
            for L_int, mag_mult in int_list:
                hooks.append(decoder[L_int].register_forward_hook(
                    make_int_hook(L_int, d, g_op, stats_op, stats_sc, mag_mult)))
            try:
                logits = fwd(input_ids, image_tensor, image_sizes, modality)
            finally:
                for h in hooks: h.remove()
            for L_cap in CAPTURE_LAYERS:
                state[cname]["hid"][L_cap].append(capture[L_cap])
            state[cname]["letter"].append(expected)
            state[cname]["dir"].append(d)
            state[cname]["pred"].append(id2l[ltids[int(logits[ltids].argmax())]])
    return state


def save_hiddens(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save as compressed npz. Keys: {cname}_{L} for hiddens, {cname}_letter, etc.
    data = {}
    for cname in state:
        for L in CAPTURE_LAYERS:
            data[f"{cname}_L{L}"] = np.stack(state[cname]["hid"][L])
        data[f"{cname}_letter"] = np.array(state[cname]["letter"])
        data[f"{cname}_dir"] = np.array(state[cname]["dir"])
        data[f"{cname}_pred"] = np.array(state[cname]["pred"])
    np.savez_compressed(path, **data)


def probe_and_report(state_or_path):
    if isinstance(state_or_path, str):
        d = np.load(state_or_path)
        state = {}
        cnames = [c[0] for c in CONDITIONS]
        for cname in cnames:
            state[cname] = {"hid": {L: d[f"{cname}_L{L}"] for L in CAPTURE_LAYERS},
                             "letter": d[f"{cname}_letter"].tolist(),
                             "dir": d[f"{cname}_dir"].tolist(),
                             "pred": d[f"{cname}_pred"].tolist()}
    else:
        state = state_or_path
        # ensure stacked
        for cname in state:
            for L in CAPTURE_LAYERS:
                if isinstance(state[cname]["hid"][L], list):
                    state[cname]["hid"][L] = np.stack(state[cname]["hid"][L])

    le_letter = {"A":0,"B":1,"C":2,"D":3}
    le_dir = {"up":0,"right":1,"down":2,"left":3}

    # MCQ
    print("\n=== MCQ acc ===")
    mcq = {}
    for c in state:
        correct = sum(1 for p,e in zip(state[c]["pred"], state[c]["letter"]) if p==e)
        n = len(state[c]["pred"])
        mcq[c] = {"correct": correct, "n": n, "acc": correct/n*100}
        print(f"  {c:>22}: {mcq[c]['acc']:6.2f}%  ({correct}/{n})")

    print("\n=== LETTER PROBE (GPU, 4-class A/B/C/D) ===")
    letter_results = {c: {} for c in state}
    print(f"{'L':>4} | " + " | ".join(f"{c:>16}" for c in state.keys()))
    for L in CAPTURE_LAYERS:
        row = f" L{L:<2} |"
        for c in state:
            X = state[c]["hid"][L]
            y = np.array([le_letter[x] for x in state[c]["letter"]])
            acc = gpu_probe(X, y, 4)
            letter_results[c][f"L{L}"] = acc
            row += f" {acc*100:>15.2f}%"
        print(row)

    print("\n=== DIRECTION PROBE (sanity) ===")
    dir_results = {c: {} for c in state}
    print(f"{'L':>4} | " + " | ".join(f"{c:>16}" for c in state.keys()))
    for L in CAPTURE_LAYERS:
        row = f" L{L:<2} |"
        for c in state:
            X = state[c]["hid"][L]
            y = np.array([le_dir[x] for x in state[c]["dir"]])
            acc = gpu_probe(X, y, 4)
            dir_results[c][f"L{L}"] = acc
            row += f" {acc*100:>15.2f}%"
        print(row)

    return {"mcq": mcq, "letter_probe": letter_results, "direction_probe": dir_results}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_only", action="store_true", help="Skip forward, load hiddens")
    args = ap.parse_args()

    hid_path = os.path.join(HID_CACHE, "comb_op.npz")

    if not args.probe_only:
        print("[load factorial]")
        OP = load_factorial("obj_place")
        g_op, stats_op = compute_stats(OP)
        SC = load_factorial("shape_color")
        _, stats_sc = compute_stats(SC)

        print(f"\nMagnitudes at key layers (OP vs SC):")
        for L in [14, 15, 16, 17, 21]:
            op = np.mean([stats_op[d]['mag'][L] for d in ['up','right','down','left']])
            sc = np.mean([stats_sc[d]['mag'][L] for d in ['up','right','down','left']])
            print(f"  L{L}: OP={op:.2f}  SC={sc:.2f}  ratio={sc/op:.2f}x")

        qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
        by_sid = defaultdict(list)
        for q in qa_all:
            by_sid[q["id"]].append(q)
        selected = []
        for sid in sorted(by_sid.keys())[:75]:
            for q in sorted(by_sid[sid], key=lambda x: x.get("variant_id", 0)):
                selected.append(q)
        questions = build_questions(selected)
        print(f"[target] {len(questions)} samples (across 4 variants)")

        tok, model, ip, _, _, ct = load_model()
        model.eval()

        state = run_forward(model, tok, ip, ct, questions, g_op, stats_op, stats_sc)
        save_hiddens(state, hid_path)
        print(f"[SAVED hiddens] {hid_path}")

        del model; gc.collect(); torch.cuda.empty_cache()
    else:
        state = hid_path  # will be loaded

    results = probe_and_report(state)

    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, "combined_intervention.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
