"""
Per-stage Δ magnitude trajectory: where does OOD magnitude deficit originate?

Measure ‖Δ_d‖ at:
  - vision_encoder (SigLIP mean-pooled output)
  - after_projector (mm_projector output mean-pooled)
  - vision_token L0..L27 (LLM vision token slice)
  - answer_token L0..L27 (LLM last token)

For 4 tasks, 3 models. Show the cascade.
"""
import numpy as np, json, os

ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500"
MODELS = {
    "vanilla":  "llava-video-7b",
    "baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "delta":    "llava-video-7b_lora_4combo_v2_delta",
}
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
LLM_LAYERS = [0, 7, 14, 18, 21, 24, 27]
DIR_KEY = ["down", "left", "right", "up"]


def load(model, task, stage, L=None):
    base = f"{ROOT}/{MODELS[model]}/{stage}/vlm_direction_testbed_R2R_4way_1500_{task}"
    if L is None:
        h = np.load(f"{base}/features.npy", mmap_mode="r").astype(np.float32)
    else:
        h = np.load(f"{base}/features_layer_{L}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{base}/labels.npy")
    # Temporal-mean if needed (N, T×D) → (N, D)
    if stage in ("vision_encoder",):
        h = h.reshape(h.shape[0], 8, -1).mean(axis=1)
    elif stage in ("after_projector", "vision_token"):
        h = h.reshape(h.shape[0], 8, -1).mean(axis=1)
    return h, y


def magnitudes(h, y):
    g = h.mean(0)
    return [float(np.linalg.norm(h[y == d].mean(0) - g)) for d in range(4)]


def main():
    stages = [("vision_encoder", None), ("after_projector", None)]
    for L in LLM_LAYERS: stages.append(("vision_token", L))
    for L in LLM_LAYERS: stages.append(("answer_token", L))

    result = {}
    for model in MODELS:
        result[model] = {}
        for (stage, L) in stages:
            key = stage if L is None else f"{stage}_L{L}"
            result[model][key] = {}
            for task in TASKS:
                h, y = load(model, task, stage, L)
                mags = magnitudes(h, y)
                result[model][key][task] = {"mean": float(np.mean(mags)),
                                              "per_dir": mags}

    out = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"
    json.dump(result, open(f"{out}/magnitude_cascade.json", "w"), indent=2)

    # Tables
    print("=" * 100)
    print("‖Δ_d‖ MEAN (across 4 directions) per stage per task per model")
    print("=" * 100)

    for model in MODELS:
        print(f"\n--- {model} ---")
        print(f"{'Stage':>25} |", end="")
        for t in TASKS: print(f" {t[:8]:>8} |", end="")
        print(" ratio SC/OP")
        for (stage, L) in stages:
            key = stage if L is None else f"{stage}_L{L}"
            r = result[model][key]
            sc, op = r["shape_color"]["mean"], r["obj_place"]["mean"]
            row = f"{key:>25} |"
            for t in TASKS:
                row += f" {r[t]['mean']:>8.3f} |"
            row += f"  {sc/max(op,1e-6):>5.2f}×"
            print(row)


if __name__ == "__main__":
    main()
