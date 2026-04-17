"""
Per-layer vision token Δ-axis cross-task cosine.

Hypothesis: Binding at L16-L17 pulls direction from VISION TOKENS
(not last token). If vision token axes remain task-specific through
LLM layers while last token axes converge, this explains binding gap.

For each (model, layer):
  Δ_d_VT = mean(vision_token[y==d]) - mean(vision_token[all])  per task
  cos(Δ_d_SC_VT, Δ_d_OP_VT) per direction, averaged

Compare with cross-task answer-token axis cos from N.1.

Data: R2R 1500 cached vision_token features (N, T×D) = (6000, 8×3584).
We take T-mean to get per-sample mean-pooled vision token per layer.
"""
import numpy as np
import json
import os

ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500"
MODELS = {
    "vanilla":  "llava-video-7b",
    "baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "delta":    "llava-video-7b_lora_4combo_v2_delta",
}
TASKS = ["shape_color", "obj_place"]
LAYERS = [0, 3, 7, 10, 14, 16, 18, 21, 24, 27]  # includes L16 (binding layer)
DIR_NAMES = ["Down", "Left", "Right", "Up"]
OUT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"


def load_vision_token(model, task, L):
    p = f"{ROOT}/{MODELS[model]}/vision_token/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{p}/features_layer_{L}.npy", mmap_mode="r").astype(np.float32)  # (N, T*D)
    y = np.load(f"{p}/labels.npy")
    # T-mean to (N, D)
    h = h.reshape(h.shape[0], 8, -1).mean(axis=1)
    return h, y


def load_answer(model, task, L):
    p = f"{ROOT}/{MODELS[model]}/answer_token/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{p}/features_layer_{L}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{p}/labels.npy")
    return h, y


def axes_per_dir(h, y):
    g = h.mean(0)
    return np.stack([h[y == d].mean(0) - g for d in range(4)])


def cos_per_dir(A, B):
    A_hat = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_hat = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A_hat * B_hat).sum(axis=1)


def main():
    results = {}
    for model in MODELS:
        results[model] = {"vt": {}, "an": {}, "vt_norms": {}}
        print(f"\n=== {model} ===")
        print(f"{'Layer':>6} | {'VT cos(SC,OP)':>14} | {'AT cos(SC,OP)':>14} | {'VT OP ||Δ||':>12} | {'VT SC ||Δ||':>12}")
        print("-" * 80)
        for L in LAYERS:
            # Vision token
            h_vt_sc, y_sc = load_vision_token(model, "shape_color", L)
            h_vt_op, y_op = load_vision_token(model, "obj_place", L)
            A_vt_sc = axes_per_dir(h_vt_sc, y_sc)
            A_vt_op = axes_per_dir(h_vt_op, y_op)
            cos_vt = cos_per_dir(A_vt_sc, A_vt_op).mean()
            norms_sc = np.linalg.norm(A_vt_sc, axis=1).mean()
            norms_op = np.linalg.norm(A_vt_op, axis=1).mean()

            # Answer token
            h_an_sc, y_sc2 = load_answer(model, "shape_color", L)
            h_an_op, y_op2 = load_answer(model, "obj_place", L)
            A_an_sc = axes_per_dir(h_an_sc, y_sc2)
            A_an_op = axes_per_dir(h_an_op, y_op2)
            cos_an = cos_per_dir(A_an_sc, A_an_op).mean()

            results[model]["vt"][f"L{L}"] = float(cos_vt)
            results[model]["an"][f"L{L}"] = float(cos_an)
            results[model]["vt_norms"][f"L{L}"] = {"SC": float(norms_sc), "OP": float(norms_op)}

            print(f"  L{L:<3} | {cos_vt:>13.3f}  | {cos_an:>13.3f}  | {norms_op:>12.2f} | {norms_sc:>12.2f}")

    with open(f"{OUT}/vision_token_axis_per_layer.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== SUMMARY — Vision Token vs Answer Token cross-task axis cos ===")
    print(f"{'Layer':>6} |", end="")
    for m in MODELS: print(f" VT_{m[:2]}  AT_{m[:2]} |", end="")
    print()
    print("-" * 80)
    for L in LAYERS:
        print(f"  L{L:<3} |", end="")
        for m in MODELS:
            vt = results[m]["vt"][f"L{L}"]
            an = results[m]["an"][f"L{L}"]
            print(f" {vt:>5.2f}  {an:>5.2f}  |", end="")
        print()


if __name__ == "__main__":
    main()
