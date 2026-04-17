"""
Identity isolation test — does identity CAUSALLY contribute to binding gap?

Previous Section M showed (b)=(e) within 0.4pp — suggesting identity role minor.
But all conditions (b)-(e) replace h with averaged prototype (cleanup direction).
This doesn't isolate "identity effect without direction change".

This script adds CLEAN identity-only interventions:
  (f) id_remove_lda: project out identity subspace (LDA-based, 4D for 5 obj classes)
      while preserving direction axis component
  (g) bg_dir_L21: replicate Section M cond-d at L21 (average over obj, keep bg+dir)
  (h) id_dir_L21: replicate Section M cond-c at L21 (average over bg, keep obj+dir)

Compare with:
  no_swap, clean_sc_mean (L21 on-axis = SC mag)

Key question:
  If id_remove_lda ≈ clean_sc_mean → identity projection doesn't add beyond magnitude
  If id_remove_lda > clean_sc_mean → removing identity HELPS (identity = noise)
  If id_remove_lda < clean_sc_mean → identity provides useful info
"""
import os, sys, json, gc, glob
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
L_TARGET = 21


def load_factorial(cond):
    arr = {"hiddens": [], "directions": [], "identities": [], "bgs": []}
    for f in sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz")):
        d = np.load(f, allow_pickle=True)
        for k in arr:
            if k in d.files:
                arr[k].append(d[k])
    return {k: np.concatenate(v) if v else None for k, v in arr.items()}


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
        out[dn] = {"hat": hat, "mag": mag, "h_avg_d": h_avg}
    return g, out


def compute_identity_subspace(data, L):
    """Compute identity (obj) subspace at layer L using LDA (returns orthonormal basis, D, k)."""
    H = data["hiddens"].astype(np.float32)[:, L, :]  # (N, D)
    ids = data["identities"]
    lda = LinearDiscriminantAnalysis(n_components=4)  # 5 classes → 4 components
    lda.fit(H, ids)
    # LDA scalings_ are not orthonormal; orthogonalize via QR
    Q, _ = np.linalg.qr(lda.scalings_[:, :4])
    return Q  # (D, 4) orthonormal


def compute_obj_dir_avgs(data, L):
    """Compute (bg, dir) → average and (id, dir) → average at L."""
    H = data["hiddens"].astype(np.float32)
    dirs = data["directions"]
    ids = data["identities"]
    bgs = data["bgs"] if data.get("bgs") is not None else np.full_like(dirs, "any")

    bg_dir_avg = {}  # key: (bg, dir) → (D,)
    id_dir_avg = {}  # key: (id, dir) → (D,)
    for bg in np.unique(bgs):
        for d in np.unique(dirs):
            mask = (bgs == bg) & (dirs == d)
            if mask.sum() > 0:
                bg_dir_avg[(str(bg), str(d))] = H[mask].mean(0)[L]
    for id_ in np.unique(ids):
        for d in np.unique(dirs):
            mask = (ids == id_) & (dirs == d)
            if mask.sum() > 0:
                id_dir_avg[(str(id_), str(d))] = H[mask].mean(0)[L]
    return bg_dir_avg, id_dir_avg


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
                    "direction": q["direction"], "identity": q.get("identity"),
                    "bg": q.get("bg"), "variant_id": q.get("variant_id", 0),
                    "video": vp})
    return out


def make_hook_clean_sc(d, g_op, stats_op, stats_sc, mag_mult=1.0):
    L = L_TARGET
    g_L = torch.from_numpy(g_op[L]).float()
    hat = torch.from_numpy(stats_op[d]["hat"][L]).float()
    mag_target = float(stats_sc[d]["mag"][L]) * mag_mult
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


def make_hook_id_remove(d, g_op, stats_op, id_basis):
    """Remove identity subspace projection, preserve direction axis component."""
    L = L_TARGET
    g_L = torch.from_numpy(g_op[L]).float()
    hat = torch.from_numpy(stats_op[d]["hat"][L]).float()  # (D,)
    Q = torch.from_numpy(id_basis).float()  # (D, k)
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        dev, dt = h.device, h.dtype
        last = h[:, -1, :].float()  # (B, D)
        gl = g_L.to(dev); hl = hat.to(dev); Ql = Q.to(dev)
        # 1. Identity subspace projection
        centered = last - gl
        id_coef = centered @ Ql  # (B, k)
        id_proj = id_coef @ Ql.T  # (B, D)
        # 2. Save direction axis component
        dir_coef = (centered * hl).sum(dim=-1, keepdim=True)  # (B, 1)
        # 3. Remove identity proj
        last_new = last - id_proj
        # 4. Restore direction axis if lost
        new_dir_coef = ((last_new - gl) * hl).sum(dim=-1, keepdim=True)
        correction = (dir_coef - new_dir_coef) * hl
        last_new = last_new + correction
        h = h.clone()
        h[:, -1, :] = last_new.to(dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_hook_replace_avg(target_vec):
    """Replace last token with precomputed target vector."""
    tv = torch.from_numpy(target_vec).float()
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        dev, dt = h.device, h.dtype
        h = h.clone()
        h[:, -1, :] = tv.to(dev, dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


@torch.no_grad()
def run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc, id_basis,
        bg_dir_avg, id_dir_avg):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(questions, "", 1, 4, tok, ip, model.config,
                             "idtest", ct, video_folder=VIDEO_FOLDER, video_fps=1,
                             frames_upbound=8, force_sample=True)
    lid = get_letter_ids(tok); id2l = {v:k for k,v in lid.items()}
    ltids = list(lid.values())
    decoder = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    conditions = ["no_swap", "clean_sc_mean", "clean_2x_sc",
                   "id_remove_lda", "bg_dir_L21", "id_dir_L21", "full_dir_L21"]
    stats_out = {c: {"n": 0, "correct": 0} for c in conditions}

    def fwd(input_ids, image_tensor, image_sizes, modality):
        (_, pos, am, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, image_tensor,
            modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=emb, attention_mask=am, position_ids=pos, return_dict=True)
        return out.logits[0, -1, :]

    for batch, line in tqdm(zip(dl, questions), total=len(questions), desc="identity_test"):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d = line["direction"]; expected = line["answer"]
        id_ = line["identity"]; bg = line["bg"]

        for cname in conditions:
            hooks = []
            if cname == "no_swap":
                pass
            elif cname == "clean_sc_mean":
                hooks.append(decoder[L_TARGET].register_forward_hook(
                    make_hook_clean_sc(d, g_op, stats_op, stats_sc, 1.0)))
            elif cname == "clean_2x_sc":
                hooks.append(decoder[L_TARGET].register_forward_hook(
                    make_hook_clean_sc(d, g_op, stats_op, stats_sc, 2.0)))
            elif cname == "id_remove_lda":
                hooks.append(decoder[L_TARGET].register_forward_hook(
                    make_hook_id_remove(d, g_op, stats_op, id_basis)))
            elif cname == "bg_dir_L21":
                # replace with (bg, dir) average
                key = (str(bg), d)
                if key not in bg_dir_avg: continue
                hooks.append(decoder[L_TARGET].register_forward_hook(
                    make_hook_replace_avg(bg_dir_avg[key])))
            elif cname == "id_dir_L21":
                key = (str(id_), d)
                if key not in id_dir_avg: continue
                hooks.append(decoder[L_TARGET].register_forward_hook(
                    make_hook_replace_avg(id_dir_avg[key])))
            elif cname == "full_dir_L21":
                hooks.append(decoder[L_TARGET].register_forward_hook(
                    make_hook_replace_avg(stats_op[d]["h_avg_d"][L_TARGET])))

            try:
                logits = fwd(input_ids, image_tensor, image_sizes, modality)
            finally:
                for h in hooks: h.remove()

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

    print("[compute identity subspace via LDA]")
    id_basis = compute_identity_subspace(OP, L_TARGET)
    print(f"  Identity subspace shape: {id_basis.shape}")
    # Check orthogonality with direction axis
    for dn in ["up", "right", "down", "left"]:
        cos_max = max(abs(stats_op[dn]["hat"][L_TARGET] @ id_basis[:, k]) for k in range(4))
        print(f"  max|cos(Δ̂_{dn}, id_basis)| = {cos_max:.3f}  (should be small if id ⊥ direction)")

    print("[compute (bg, dir) and (id, dir) averages at L21]")
    bg_dir_avg, id_dir_avg = compute_obj_dir_avgs(OP, L_TARGET)
    print(f"  (bg, dir) keys: {len(bg_dir_avg)}")
    print(f"  (id, dir) keys: {len(id_dir_avg)}")

    qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[:500]
    questions = build_questions(qa_target)

    tok, model, ip, _, _, ct = load_model()
    model.eval()

    stats = run(model, tok, ip, ct, questions, g_op, stats_op, stats_sc,
                 id_basis, bg_dir_avg, id_dir_avg)

    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"\n=== Identity isolation: obj_place (base={base:.2f}%) ===")
    for c, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>18}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp  (n={s['n']})")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"stats": stats}, open(os.path.join(OUT_ROOT, "identity_test.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
