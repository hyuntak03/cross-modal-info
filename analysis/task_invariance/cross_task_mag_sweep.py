"""
Cross-task magnitude sweep at L21 — check if binding gap = magnitude deficit
is universal across all OOD levels.

For each target task (obj_color, shape_place, obj_place):
  5 conditions:
    no_swap
    clean_op_half  (OP_own mean × 0.5)
    clean_op_mean  (OP_own mean)
    clean_sc_mean  (SC mean, known +10pp for obj_place)
    clean_2x_sc    (2 × SC mean, known +17.6pp for obj_place)

Uses factorial OP data for obj_place but R2R 1500 cached for obj_color/shape_place
(factorial only has obj_place + shape_color).
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
R2R_HIDDEN_ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline/answer_token"
OUT_ROOT = os.path.join(_PROJECT_ROOT, "analysis/task_invariance/mech_results")
L_TARGET = 21


def compute_axes_from_r2r(task):
    """Compute Δ̂_d, mag per direction from R2R 1500 cached last-token hidden at L21."""
    p = f"{R2R_HIDDEN_ROOT}/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{p}/features_layer_{L_TARGET}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{p}/labels.npy")
    g = h.mean(0)
    # labels: 0=Down, 1=Left, 2=Right, 3=Up
    DIR_MAP = {"down":0, "left":1, "right":2, "up":3}
    stats = {}
    for dn, dl in DIR_MAP.items():
        avg = h[y == dl].mean(0)
        Delta = avg - g
        mag = np.linalg.norm(Delta)
        stats[dn] = {"Delta_hat": Delta / (mag + 1e-9), "mag": float(mag), "h_avg": avg}
    return g, stats


def load_model():
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}")
    return load_model_from_args(a)


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A","B","C","D"]:
        for cand in [ltr, " "+ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]; break
    return ids


def make_hook(mode, d, g_L, stats_task, mag_SC_d):
    g = torch.from_numpy(g_L).float()
    dhat = torch.from_numpy(stats_task[d]["Delta_hat"]).float()
    mag_own = stats_task[d]["mag"]

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        dev, dt = h.device, h.dtype
        last = h[:, -1, :].float()
        centered = last - g.to(dev).float()
        proj = (centered * dhat.to(dev).float()).sum(dim=-1, keepdim=True)
        if mode == "clean_op_half":
            new_mag = mag_own * 0.5
        elif mode == "clean_op_mean":
            new_mag = mag_own
        elif mode == "clean_sc_mean":
            new_mag = mag_SC_d
        elif mode == "clean_2x_sc":
            new_mag = mag_SC_d * 2.0
        last = last - proj * dhat.to(dev).float() + new_mag * dhat.to(dev).float()
        h = h.clone()
        h[:, -1, :] = last.to(dt)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


@torch.no_grad()
def run(task, model, tokenizer, image_processor, conv_template,
        g_L, stats_task, stats_SC):
    # Load R2R questions via lmms-eval style dataset
    import importlib.util
    def _imp(n, p):
        s = importlib.util.spec_from_file_location(n, p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    dl_mod = _imp("core.dataset_loader", f"{_PROJECT_ROOT}/core/dataset_loader.py")
    dp_mod = _imp("core.data_pipeline", f"{_PROJECT_ROOT}/core/data_pipeline.py")

    questions, _ = dl_mod.load_dataset_as_questions(
        task_name=f"vlm_direction_testbed_R2R_4way_1500_{task}",
        video_folder=VIDEO_FOLDER,
        image_folder="",
        hf_cache_dir=os.environ.get("HF_HOME"),
        limit=400,  # 400 samples per task
    )
    data_loader = dp_mod.create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        f"mag_{task}", conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())
    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers

    conditions = [
        ("no_swap", None),
        ("clean_op_half", "clean_op_half"),
        ("clean_op_mean", "clean_op_mean"),
        ("clean_sc_mean", "clean_sc_mean"),
        ("clean_2x_sc", "clean_2x_sc"),
    ]
    stats_out = {c[0]: {"n": 0, "correct": 0} for c in conditions}

    # Direction name to match questions: R2R labels are "Up", "Down", "Left", "Right" capitalized
    DIR_MAP = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}

    def fwd(input_ids, image_tensor, image_sizes, modality):
        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor,
                modalities=[modality], image_sizes=image_sizes)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                      position_ids=position_ids, return_dict=True)
        logits = out.logits[0, -1, :]
        return id_to_letter[letter_tok_ids[int(logits[letter_tok_ids].argmax())]]

    for batch, line in tqdm(zip(data_loader, questions), total=len(questions), desc=task):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        d_raw = str(line.get("direction") or line.get("answer_direction") or "")
        d = DIR_MAP.get(d_raw, d_raw.lower())
        if d not in stats_task: continue
        expected = str(line["answer"]).strip()
        # resolve letter
        if len(expected) != 1:
            cands = line.get("candidates", [])
            if isinstance(cands, str):
                import ast
                try: cands = ast.literal_eval(cands)
                except: cands = []
            for ci, c in enumerate(cands):
                if str(c).strip() == expected:
                    expected = chr(65 + ci); break
        try:
            for cname, mode in conditions:
                if mode is None:
                    pred = fwd(input_ids, image_tensor, image_sizes, modality)
                else:
                    mag_SC = stats_SC[d]["mag"]
                    hh = decoder_layers[L_TARGET].register_forward_hook(
                        make_hook(mode, d, g_L, stats_task, mag_SC))
                    try:
                        pred = fwd(input_ids, image_tensor, image_sizes, modality)
                    finally:
                        hh.remove()
                stats_out[cname]["n"] += 1
                if pred == expected: stats_out[cname]["correct"] += 1
        except Exception as e:
            print(f"[ERR] {line.get('q_id','?')}: {e}")
    return stats_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["obj_color", "shape_place", "obj_place"])
    args = ap.parse_args()

    print(f"[load axes] SC (reference)")
    g_SC, stats_SC = compute_axes_from_r2r("shape_color")
    print(f"[load axes] target: {args.task}")
    g_task, stats_task = compute_axes_from_r2r(args.task)

    print(f"\nMagnitudes at L{L_TARGET}:")
    for d in stats_task:
        own = stats_task[d]["mag"]; sc = stats_SC[d]["mag"]
        print(f"  {d}: own={own:.2f}  SC={sc:.2f}  ratio own/SC={own/sc:.2f}")

    tokenizer, model, image_processor, _, _, conv_template = load_model()
    model.eval()

    stats = run(args.task, model, tokenizer, image_processor, conv_template,
                 g_task, stats_task, stats_SC)

    base = stats["no_swap"]["correct"] / max(stats["no_swap"]["n"], 1) * 100
    print(f"\n=== Magnitude sweep: {args.task} (base={base:.2f}%) ===")
    for c, s in stats.items():
        acc = s["correct"] / max(s["n"], 1) * 100
        print(f"  {c:>18s}: {acc:6.2f}%   Δ={acc-base:+5.2f}pp")

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump({"task": args.task, "stats": stats, "L": L_TARGET},
              open(os.path.join(OUT_ROOT, f"mag_sweep_{args.task}.json"), "w"), indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
