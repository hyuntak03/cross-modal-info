"""
Refinement hypothesis test — does L21 direction boost improve L22-L27 letter probe?

Hypothesis: L22-L27 performs "letter refinement" using direction signal at L21.
  - L21 letter probe OP = 72% (not saturated)
  - L27 letter probe OP = 79% (refinement +7pp)
  - L21 boost MCQ = 86% > L27 probe 79% → boost activates refinement beyond baseline

Conditions (5):
  no_swap
  L14_clean_sc      (pre-binding, control — didn't work for MCQ)
  L21_clean_sc      (post-binding, SC magnitude)
  L21_clean_2x_sc   (full recovery magnitude)
  L14_plus_L21      (combined: fix binding input AND refinement fuel)

Captures: L16 (binding), L18, L21, L22, L24, L26, L27

Letter probe + direction probe at each capture (4-variant data).

Interpretation:
  - If L21 clean_2x_sc → L27 letter probe jumps 79% → 95%+ → refinement hypothesis confirmed
  - If L27 letter probe unchanged but MCQ jumps → lm_head non-linear effect
  - If L14_plus_L21 > L21_alone → binding + refinement both needed (additive)
"""
import os, sys, json, gc, glob
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

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

CAPTURE_LAYERS = [16, 18, 21, 22, 24, 26, 27]


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


def make_int_hook(L_int, d, g_op, stats_op, stats_sc, magnitude_mult=1.0):
    """clean_sc with customizable SC mag multiplier."""
    g_L = torch.from_numpy(g_op[L_int]).float()
    hat = torch.from_numpy(stats_op[d]["hat"][L_int]).float()
    mag_target = float(stats_sc[d]["mag"][L_int]) * magnitude_mult
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
def run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(questions, "", 1, 4, tok, ip, model.config,
                             "ref", ct, video_folder=VIDEO_FOLDER, video_fps=1,
                             frames_upbound=8, force_sample=True)
    lid = get_letter_ids(tok); id2l = {v:k for k,v in lid.items()}
    ltids = list(lid.values())
    decoder = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    # conditions: (name, list of (layer, magnitude_multiplier))
    conditions = [
        ("no_swap", []),
        ("L14_clean", [(14, 1.0)]),
        ("L21_clean_sc", [(21, 1.0)]),
        ("L21_clean_2x_sc", [(21, 2.0)]),
        ("L14_plus_L21", [(14, 1.0), (21, 2.0)]),
    ]

    state = {c[0]: {"hid": {L: [] for L in CAPTURE_LAYERS},
                     "letter": [], "dir": [], "pred": []} for c in conditions}

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
        d = line["direction"]; expected = line["answer"]

        for cname, int_list in conditions:
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


def eval_probe(X_list, labels, le):
    X = np.stack(X_list).astype(np.float32)
    y = np.array([le[lbl] for lbl in labels])
    np.random.seed(42)
    idx = np.random.permutation(len(y))
    n_tr = int(0.7 * len(idx))
    tr, te = idx[:n_tr], idx[n_tr:]
    s = StandardScaler().fit(X[tr])
    Xt, Xv = s.transform(X[tr]), s.transform(X[te])
    # Use simpler/faster solver
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs').fit(Xt, y[tr])
    return float(clf.score(Xv, y[te]))


def main():
    print("[load factorial OP/SC]")
    OP = load_factorial("obj_place")
    g_op, stats_op = compute_stats(OP)
    SC = load_factorial("shape_color")
    _, stats_sc = compute_stats(SC)

    print(f"\nL21 magnitudes: OP={np.mean([stats_op[d]['mag'][21] for d in ['up','right','down','left']]):.2f}  "
          f"SC={np.mean([stats_sc[d]['mag'][21] for d in ['up','right','down','left']]):.2f}")

    qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
    by_sid = defaultdict(list)
    for q in qa_all:
        by_sid[q["id"]].append(q)
    selected = []
    for sid in sorted(by_sid.keys())[:75]:
        for q in sorted(by_sid[sid], key=lambda x: x.get("variant_id", 0)):
            selected.append(q)
    questions = build_questions(selected)
    print(f"[target] {len(questions)} samples")

    tok, model, ip, _, _, ct = load_model()
    model.eval()

    state = run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc)

    print("\n=== MCQ acc ===")
    for c in state:
        correct = sum(1 for p,e in zip(state[c]["pred"], state[c]["letter"]) if p==e)
        n = len(state[c]["pred"])
        print(f"  {c:>16}: {correct/n*100:6.2f}%  ({correct}/{n})")

    print("\n=== LETTER PROBE (4-class A/B/C/D) ===")
    print(f"{'L':>4} | " + " | ".join(f"{c:>14}" for c in state.keys()))
    le_letter = {"A":0,"B":1,"C":2,"D":3}
    letter_res = {c: {} for c in state}
    for L in CAPTURE_LAYERS:
        row = f" L{L:<2} |"
        for c in state:
            acc = eval_probe(state[c]["hid"][L], state[c]["letter"], le_letter)
            letter_res[c][f"L{L}"] = acc
            row += f" {acc*100:>13.2f}%"
        print(row)

    print("\n=== DIRECTION PROBE ===")
    le_dir = {"up":0,"right":1,"down":2,"left":3}
    dir_res = {c: {} for c in state}
    print(f"{'L':>4} | " + " | ".join(f"{c:>14}" for c in state.keys()))
    for L in CAPTURE_LAYERS:
        row = f" L{L:<2} |"
        for c in state:
            acc = eval_probe(state[c]["hid"][L], state[c]["dir"], le_dir)
            dir_res[c][f"L{L}"] = acc
            row += f" {acc*100:>13.2f}%"
        print(row)

    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = {
        "n_samples": len(questions),
        "mcq": {c: {"correct": sum(1 for p,e in zip(state[c]["pred"], state[c]["letter"]) if p==e),
                     "n": len(state[c]["pred"])} for c in state},
        "letter_probe": letter_res,
        "direction_probe": dir_res,
    }
    json.dump(summary, open(os.path.join(OUT_ROOT, "refinement_test.json"), "w"), indent=2)
    print(f"\n[SAVED]")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
