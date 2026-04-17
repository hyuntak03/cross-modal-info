"""
Rigorous verification of two critical claims:

A. "clean_2x_sc magnitude boost effect is specific to direction axis (Δ̂_d)"
   - Control: random unit vectors → MCQ should NOT improve
   - Control: identity LDA axis → should NOT improve
   - If random axis also improves → claim invalidated

B. "Magnitude-acc monotonic, no saturation"
   - Sweep: 0.5×, 1×, 2×, 3×, 5×, 10× SC mean
   - Negative: -1×, -2× SC mean (wrong direction)

Combined to single run for efficiency.
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
    arr = {"hiddens": [], "directions": [], "identities": []}
    for f in sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz")):
        d = np.load(f, allow_pickle=True)
        for k in arr:
            if k in d.files: arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items() if v}


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


def compute_identity_basis(data, L):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    H = data["hiddens"].astype(np.float32)[:, L, :]
    ids = data["identities"]
    lda = LinearDiscriminantAnalysis(n_components=min(4, len(np.unique(ids))-1)).fit(H, ids)
    Q, _ = np.linalg.qr(lda.scalings_[:, :4])
    return Q  # (D, k)


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
                    "direction": q["direction"], "video": vp})
    return out


def make_hook_set_mag(d, g_op, hat_override, mag_target):
    """Generic: set projection on hat_override to mag_target."""
    L = L_TARGET
    g_L = torch.from_numpy(g_op[L]).float()
    hat = torch.from_numpy(hat_override).float()
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


@torch.no_grad()
def run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc, id_basis):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(questions, "", 1, 4, tok, ip, model.config,
                             "rigor", ct, video_folder=VIDEO_FOLDER, video_fps=1,
                             frames_upbound=8, force_sample=True)
    lid = get_letter_ids(tok); id2l = {v:k for k,v in lid.items()}
    ltids = list(lid.values())
    decoder = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    # Pre-generate 3 random axes (same for all samples)
    np.random.seed(42)
    D = stats_op["up"]["hat"].shape[1]
    random_axes = []
    for ri in range(3):
        rv = np.random.randn(D).astype(np.float32)
        rv /= np.linalg.norm(rv) + 1e-9
        random_axes.append(rv)

    # Identity LDA basis direction (take first component)
    id_axis = id_basis[:, 0].astype(np.float32)
    id_axis /= np.linalg.norm(id_axis) + 1e-9

    # Conditions
    # For direction axis: use per-direction Δ̂_d (varies by sample)
    # For random axes: use same random vector for all samples
    conditions = []
    conditions.append(("no_swap", None, None, 0.0))
    # Magnitude sweep on direction axis
    for mult, name in [(0.5, "dir_0.5x"), (1.0, "dir_1x"), (2.0, "dir_2x"),
                       (3.0, "dir_3x"), (5.0, "dir_5x"), (10.0, "dir_10x"),
                       (-1.0, "dir_neg1x"), (-2.0, "dir_neg2x")]:
        conditions.append((f"dir_mag_{mult}x_sc", "direction", None, mult))
    # Random axis controls (at SC mag and 2×SC mag)
    for ri in range(3):
        conditions.append((f"rand_axis_{ri}_1x_sc", "random", random_axes[ri], 1.0))
        conditions.append((f"rand_axis_{ri}_2x_sc", "random", random_axes[ri], 2.0))
    # Identity axis controls
    conditions.append(("id_axis_1x_sc", "identity", id_axis, 1.0))
    conditions.append(("id_axis_2x_sc", "identity", id_axis, 2.0))

    stats_out = {c[0]: {"n": 0, "correct": 0} for c in conditions}

    def fwd(input_ids, image_tensor, image_sizes, modality):
        (_, pos, am, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, image_tensor,
            modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=emb, attention_mask=am, position_ids=pos, return_dict=True)
        return out.logits[0, -1, :]

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc="rigor"):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d = line["direction"]; expected = line["answer"]

        # Reference SC magnitude for this direction at L21
        mag_sc_d = float(stats_sc[d]["mag"][L_TARGET])
        # OP own magnitude (for reference)
        mag_op_d = float(stats_op[d]["mag"][L_TARGET])

        for cname, axis_type, axis_override, multiplier in conditions:
            if axis_type is None:
                hook = None
            elif axis_type == "direction":
                # Use direction axis of this sample's direction
                hat = stats_op[d]["hat"][L_TARGET]
                target_mag = mag_sc_d * multiplier
                hook = decoder[L_TARGET].register_forward_hook(
                    make_hook_set_mag(d, g_op, hat, target_mag))
            else:
                # Random or identity axis — fixed for all samples
                hat = axis_override
                target_mag = mag_sc_d * multiplier
                hook = decoder[L_TARGET].register_forward_hook(
                    make_hook_set_mag(d, g_op, hat, target_mag))
            try:
                logits = fwd(input_ids, image_tensor, image_sizes, modality)
            finally:
                if hook is not None: hook.remove()

            pred = id2l[ltids[int(logits[ltids].argmax())]]
            stats_out[cname]["n"] += 1
            if pred == expected: stats_out[cname]["correct"] += 1

    return stats_out


def main():
    print("[load factorial]")
    OP = load_factorial("obj_place")
    g_op, stats_op = compute_stats(OP)
    SC = load_factorial("shape_color")
    _, stats_sc = compute_stats(SC)

    print("[compute identity LDA basis at L21]")
    id_basis = compute_identity_basis(OP, L_TARGET)

    # Verify: cos between identity basis and direction axis
    print("\nSanity: cos(Δ̂_d, id_basis[:, 0])")
    for d in ["up", "right", "down", "left"]:
        c = float(stats_op[d]["hat"][L_TARGET] @ id_basis[:, 0])
        print(f"  {d}: {c:.4f}  (should be ~0)")

    # Reference magnitudes
    print(f"\nReference magnitudes at L{L_TARGET}:")
    for d in ["up", "right", "down", "left"]:
        op = stats_op[d]["mag"][L_TARGET]
        sc = stats_sc[d]["mag"][L_TARGET]
        print(f"  {d}: OP={op:.2f}  SC={sc:.2f}  ratio={sc/op:.2f}x")

    qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[:300]
    questions = build_questions(qa_target)

    tok, model, ip, _, _, ct = load_model()
    model.eval()

    stats = run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc, id_basis)

    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"\n{'='*80}")
    print(f"RIGOROUS VERIFICATION — obj_place (base={base:.2f}%)")
    print(f"{'='*80}")

    # Group A: direction axis sweep
    print(f"\n--- Direction axis magnitude sweep ---")
    for c in ["dir_mag_0.5x_sc", "dir_mag_1.0x_sc", "dir_mag_2.0x_sc",
              "dir_mag_3.0x_sc", "dir_mag_5.0x_sc", "dir_mag_10.0x_sc",
              "dir_mag_-1.0x_sc", "dir_mag_-2.0x_sc"]:
        s = stats[c]
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>22}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    # Group B: random axis controls
    print(f"\n--- Random axis controls ---")
    for c in ["rand_axis_0_1x_sc", "rand_axis_0_2x_sc",
              "rand_axis_1_1x_sc", "rand_axis_1_2x_sc",
              "rand_axis_2_1x_sc", "rand_axis_2_2x_sc"]:
        s = stats[c]
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>22}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    # Group C: identity axis controls
    print(f"\n--- Identity LDA axis controls ---")
    for c in ["id_axis_1x_sc", "id_axis_2x_sc"]:
        s = stats[c]
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>22}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"stats": stats, "base": base},
              open(os.path.join(OUT_ROOT, "rigorous_verification.json"), "w"), indent=2)
    print(f"\n[SAVED]")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
