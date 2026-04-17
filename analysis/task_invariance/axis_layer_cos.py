"""
Within-task Δ_d axis cosine across layers.

Q: Does the direction axis rotate through layers, or is it the same axis?
- If cos(Δ_L14, Δ_L21) ≈ 1 → same axis, L19 amplifies magnitude on stable axis (interpretation A)
- If cos(Δ_L14, Δ_L21) low → axis rotates, L19 performs alignment (interpretation B)
"""
import numpy as np

ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500"
MODELS = {
    "vanilla":  "llava-video-7b",
    "baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "delta":    "llava-video-7b_lora_4combo_v2_delta",
}
TASKS = ["shape_color", "obj_place"]
LAYERS = [3, 7, 10, 14, 17, 18, 19, 20, 21, 24, 27]
LABELS = ["Down", "Left", "Right", "Up"]


def load(model, task, L):
    p = f"{ROOT}/{MODELS[model]}/answer_token/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{p}/features_layer_{L}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{p}/labels.npy")
    return h, y


def deltas(h, y):
    g = h.mean(0)
    return np.stack([h[y == d].mean(0) - g for d in range(4)])  # (4, D)


def main():
    REF = 21
    for model in ["vanilla", "baseline", "delta"]:
        print(f"\n{'='*80}\n{model}\n{'='*80}")
        for task in TASKS:
            d_ref = deltas(*load(model, task, REF))  # (4, D)
            d_ref_hat = d_ref / (np.linalg.norm(d_ref, axis=1, keepdims=True) + 1e-9)
            print(f"\n--- {task}, reference = L{REF} ---")
            print(f"{'Layer':>6} | {'Down':>6} | {'Left':>6} | {'Right':>6} | {'Up':>6} | {'mean':>6}")
            for L in LAYERS:
                d_L = deltas(*load(model, task, L))
                d_L_hat = d_L / (np.linalg.norm(d_L, axis=1, keepdims=True) + 1e-9)
                cos_per = [float(d_L_hat[i] @ d_ref_hat[i]) for i in range(4)]
                row = f"  L{L:<3} |"
                for c in cos_per: row += f" {c:>6.3f} |"
                row += f" {np.mean(cos_per):>6.3f}"
                print(row)

        # Also task-invariance at L21 for reference
        print(f"\n[{model}] cross-task Δ_L{REF} cosine (sanity): SC vs OP")
        d_sc = deltas(*load(model, "shape_color", REF))
        d_op = deltas(*load(model, "obj_place", REF))
        d_sc_hat = d_sc / (np.linalg.norm(d_sc, axis=1, keepdims=True) + 1e-9)
        d_op_hat = d_op / (np.linalg.norm(d_op, axis=1, keepdims=True) + 1e-9)
        cos_cross = [float(d_sc_hat[i] @ d_op_hat[i]) for i in range(4)]
        print(f"  per direction: {[f'{c:.3f}' for c in cos_cross]}, mean={np.mean(cos_cross):.3f}")


if __name__ == "__main__":
    main()
