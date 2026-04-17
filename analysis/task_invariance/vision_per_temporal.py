"""
Per-temporal vision intervention test.

Previous vision amp experiments failed because Δ̂_d_vision was mean-pooled
over all (T, S) positions and applied uniformly — destroying per-position structure.

This test: compute per-temporal direction axis Δ̂_d^t (t=1..8), apply each Δ̂_d^t
only to the corresponding temporal frame's vision tokens at projector output.

Conditions:
  no_swap
  per_t_amp_1.0     (per-temporal on-axis amp, scale 2x = add 1x proj)
  per_t_amp_2.0
  per_t_amp_5.0
  per_t_clean_sc    (per-temporal on-axis set to SC-t magnitude)
  per_t_clean_2x_sc
  baseline_mean_amp_5.0  (old mean-pooled for comparison, should still destroy)

Measures: MCQ acc
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
R2R_ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline"
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/mech_results")

T_FRAMES = 8


def compute_per_t_axes(model_task):
    """Compute per-temporal direction axis from cached after_projector features.

    Returns: dict d → {"hat": (T, D), "mag": (T,), "Delta": (T, D)}
    """
    base = f"{R2R_ROOT}/after_projector/vlm_direction_testbed_R2R_4way_1500_{model_task}"
    h = np.load(f"{base}/features.npy", mmap_mode="r").astype(np.float32)  # (N, T*D)
    y = np.load(f"{base}/labels.npy")
    # 0=Down, 1=Left, 2=Right, 3=Up
    dir_names = ["down", "left", "right", "up"]
    D = h.shape[1] // T_FRAMES
    H = h.reshape(h.shape[0], T_FRAMES, D)  # (N, T, D)

    g = H.mean(0)  # (T, D)
    out = {}
    for di, dn in enumerate(dir_names):
        h_avg = H[y == di].mean(0)  # (T, D)
        Delta = h_avg - g  # (T, D)
        mag = np.linalg.norm(Delta, axis=1)  # (T,)
        hat = Delta / (mag[:, None] + 1e-9)
        out[dn] = {"hat": hat, "mag": mag, "Delta": Delta}
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
                    "direction": q["direction"], "video": vp})
    return out


def make_hook(mode, d, stats_op, stats_sc, g_op, alpha=1.0, verbose_once=True):
    """Hook at mm_projector output. Applies per-temporal intervention.

    Assumes output shape (B, n_vision, D) with n_vision = T × S_spatial (temporal-major).
    If n_vision % T != 0, abort with error.
    """
    hat_op = torch.from_numpy(stats_op[d]["hat"]).float()  # (T, D)
    mag_op = torch.from_numpy(stats_op[d]["mag"]).float()  # (T,)
    mag_sc = torch.from_numpy(stats_sc[d]["mag"]).float()  # (T,)
    # For baseline_mean test — use mean over T
    hat_mean = torch.from_numpy(stats_op[d]["hat"].mean(0)).float()  # (D,)
    mag_mean = float(stats_op[d]["mag"].mean())
    printed = [False]

    def hook(module, inputs, output):
        if isinstance(output, (list, tuple)):
            out_t = output[0]
        else:
            out_t = output
        if out_t is None or out_t.dim() < 2:
            return output
        dev, dtype = out_t.device, out_t.dtype
        t32 = out_t.float()
        # LLaVA-Video projector output: (T_frames, S_spatial, D) where T is batch dim
        if t32.dim() == 3 and t32.shape[0] == T_FRAMES:
            # (T, S, D) — one frame per batch item
            T, S, D = t32.shape
            if not printed[0] and verbose_once:
                print(f"[hook] projector output shape: ({T}, {S}, {D}) — per-frame batched")
                printed[0] = True

            hop = hat_op.to(dev)  # (T, D)
            msc = mag_sc.to(dev)  # (T,)

            if mode == "per_t_amp":
                # Per-t: scale each frame's on-axis projection
                # proj[t, s] = v[t, s] · hop[t]
                proj = (t32 * hop[:, None, :]).sum(dim=-1, keepdim=True)  # (T, S, 1)
                t32 = t32 + (alpha - 1.0) * proj * hop[:, None, :]
            elif mode == "per_t_clean_sc":
                proj = (t32 * hop[:, None, :]).sum(dim=-1, keepdim=True)
                target_mag = (alpha * msc).view(T, 1, 1)
                t32 = t32 - proj * hop[:, None, :] + target_mag * hop[:, None, :]
            elif mode == "baseline_mean_amp":
                hop_mean = hat_mean.to(dev)  # (D,)
                proj = (t32 * hop_mean[None, None, :]).sum(dim=-1, keepdim=True)
                t32 = t32 + (alpha - 1.0) * proj * hop_mean[None, None, :]
            t32 = t32.to(dtype)
        else:
            # Unexpected shape — skip
            if not printed[0] and verbose_once:
                print(f"[hook] unexpected shape {t32.shape} — skipping")
                printed[0] = True
            return output
        if isinstance(output, (list, tuple)):
            out_list = list(output)
            out_list[0] = t32
            return type(output)(out_list)
        return t32
    return hook


@torch.no_grad()
def run(model, tok, ip, ct, questions, stats_op, stats_sc, g_op):
    from core.data_pipeline import create_data_loader
    dl = create_data_loader(questions, "", 1, 4, tok, ip, model.config,
                             "pt", ct, video_folder=VIDEO_FOLDER, video_fps=1,
                             frames_upbound=8, force_sample=True)
    lid = get_letter_ids(tok); id2l = {v:k for k,v in lid.items()}
    ltids = list(lid.values())
    mm_proj = model.model.mm_projector if hasattr(model.model, "mm_projector") else model.get_model().mm_projector

    conditions = [
        ("no_swap", None, 1.0),
        ("per_t_amp_2x", "per_t_amp", 2.0),
        ("per_t_amp_5x", "per_t_amp", 5.0),
        ("per_t_clean_sc", "per_t_clean_sc", 1.0),
        ("per_t_clean_2x_sc", "per_t_clean_sc", 2.0),
        ("baseline_mean_amp_5x", "baseline_mean_amp", 5.0),
    ]
    stats = {c[0]: {"n":0, "correct":0} for c in conditions}

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
        for cname, mode, alpha in conditions:
            if mode is None:
                hook = None
            else:
                hook = mm_proj.register_forward_hook(
                    make_hook(mode, d, stats_op, stats_sc, g_op, alpha,
                               verbose_once=(stats[cname]["n"] == 0)))
            try:
                logits = fwd(input_ids, image_tensor, image_sizes, modality)
            finally:
                if hook is not None: hook.remove()
            pred = id2l[ltids[int(logits[ltids].argmax())]]
            stats[cname]["n"] += 1
            if pred == expected: stats[cname]["correct"] += 1

    return stats


def main():
    print("[compute per-temporal axes from R2R 1500 Baseline projector features]")
    g_op, stats_op = compute_per_t_axes("obj_place")
    _, stats_sc = compute_per_t_axes("shape_color")

    print("\nPer-temporal magnitude (OP):")
    for d in ["up", "right", "down", "left"]:
        print(f"  {d}: mag per frame = {[f'{m:.2f}' for m in stats_op[d]['mag']]}, mean {stats_op[d]['mag'].mean():.2f}")
    print("\nPer-temporal magnitude (SC):")
    for d in ["up", "right", "down", "left"]:
        print(f"  {d}: mag per frame = {[f'{m:.2f}' for m in stats_sc[d]['mag']]}, mean {stats_sc[d]['mag'].mean():.2f}")

    # Target samples
    qa_all = json.load(open(os.path.join(JSON_ROOT, "obj_place_4variants.json")))
    qa_v0 = [q for q in qa_all if q.get("variant_id", 0) == 0]
    qa_target = qa_v0[:300]
    questions = build_questions(qa_target)

    tok, model, ip, _, _, ct = load_model()
    model.eval()

    stats = run(model, tok, ip, ct, questions, stats_op, stats_sc, g_op)

    base = stats["no_swap"]["correct"]/max(stats["no_swap"]["n"],1)*100
    print(f"\n=== Per-temporal vision intervention: obj_place (base={base:.2f}%) ===")
    for c, s in stats.items():
        acc = s["correct"]/max(s["n"],1)*100
        print(f"  {c:>22}: {acc:6.2f}%  Δ={acc-base:+5.2f}pp  (n={s['n']})")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"stats": stats}, open(os.path.join(OUT_ROOT, "vision_per_temporal.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
