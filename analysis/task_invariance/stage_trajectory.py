"""
Stage-wise on-axis / (on+off) energy ratio trajectory.

For each stage and task, compute:
  1. Δ_d = h_avg(d) - grand_mean, Δ̂_d = Δ_d / ||Δ_d||
  2. Per sample: on_energy = ((h-g)·Δ̂_{d_i})^2, off = ||(h-g) - on·Δ̂||^2
  3. ratio = on / (on + off)

Stages:
  - vision_encoder (D=9216, flattened 8×1152)
  - after_projector (D=28672, flattened 8×3584)
  - vision_token @ L=0,3,7,10,14,18,21,24,27 (D=28672)
  - answer_token @ same layers (D=3584)

If OOD ratio ≈ IN-domain ratio at vision encoder but diverges mid-LLM → mechanism is in LLM.
If OOD lower from vision encoder → upstream perceptual problem.
"""
import os, json, numpy as np

ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500"
MODELS = {
    "vanilla":  "llava-video-7b",
    "baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "delta":    "llava-video-7b_lora_4combo_v2_delta",
}
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
LLM_LAYERS = [0, 3, 7, 10, 14, 18, 21, 24, 27]
OUT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"


def load_stage(model, task, stage, layer=None):
    base = f"{ROOT}/{MODELS[model]}/{stage}/vlm_direction_testbed_R2R_4way_1500_{task}"
    if layer is None:
        h = np.load(f"{base}/features.npy", mmap_mode="r").astype(np.float32)
    else:
        h = np.load(f"{base}/features_layer_{layer}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{base}/labels.npy")
    return h, y


def on_off_ratio(h, y):
    g = h.mean(0)
    proto = np.stack([h[y == d].mean(0) for d in range(4)])
    Delta = proto - g[None, :]
    norms = np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9
    Delta_hat = Delta / norms
    centered = h - g[None, :]
    on = (centered * Delta_hat[y]).sum(axis=1)
    on_e = (on ** 2).mean()
    total_e = (centered ** 2).sum(axis=1).mean()
    off_e = total_e - on_e
    ratio = float(on_e / (on_e + off_e + 1e-9))
    return {
        "ratio": ratio,
        "on_mean_magnitude": float(np.abs(on).mean()),
        "on_std": float(on.std()),
        "total_norm_mean": float(np.sqrt(total_e)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}

    stages = []
    stages.append(("vision_encoder", None))
    stages.append(("after_projector", None))
    for L in LLM_LAYERS:
        stages.append(("vision_token", L))
    for L in LLM_LAYERS:
        stages.append(("answer_token", L))

    for model in MODELS:
        print(f"[{model}]", flush=True)
        out[model] = {}
        for (stage, L) in stages:
            key = stage if L is None else f"{stage}_L{L}"
            out[model][key] = {}
            for task in TASKS:
                h, y = load_stage(model, task, stage, L)
                out[model][key][task] = on_off_ratio(h, y)
            row = " ".join(f"{task[:2]}={out[model][key][task]['ratio']:.3f}" for task in TASKS)
            print(f"  {key:>22s}: {row}", flush=True)

    out_path = f"{OUT}/stage_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_path}\n")

    # Summary tables
    print("=" * 104)
    print("ON-AXIS / TOTAL ENERGY RATIO — stage-wise trajectory (direction info concentration)")
    print("=" * 104)
    stage_order = [("vision_encoder", None), ("after_projector", None)] \
                  + [("vision_token", L) for L in LLM_LAYERS] \
                  + [("answer_token", L) for L in LLM_LAYERS]
    labels = []
    for s, L in stage_order:
        labels.append(s if L is None else f"{s[:2]}_L{L}")

    for model in MODELS:
        print(f"\n--- {model} ---")
        header = f"{'Stage':>22s} | {'SC':>7} | {'OC':>7} | {'SP':>7} | {'OP':>7} | OOD gap (SC-OP)"
        print(header)
        print("-" * len(header))
        for (stage, L), lab in zip(stage_order, labels):
            key = stage if L is None else f"{stage}_L{L}"
            rates = [out[model][key][t]["ratio"] for t in TASKS]
            gap = rates[0] - rates[3]
            print(f"{lab:>22s} | {rates[0]:>7.4f} | {rates[1]:>7.4f} | {rates[2]:>7.4f} | {rates[3]:>7.4f} | {gap:>+.4f}")


if __name__ == "__main__":
    main()
