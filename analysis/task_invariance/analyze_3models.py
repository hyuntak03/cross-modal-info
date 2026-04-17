"""
3-model × 2-task attn/mlp contribution comparison at L10-L25.

Projects each layer's attn_out/mlp_out at last token onto Δ̂_d
(direction axis from that model's L=21 hidden).
"""
import os, json
import numpy as np

CONTRIB_ROOT = "/local_datasets/vlm_direction/attn_mlp_contrib"
HIDDEN_ROOTS = {
    "vanilla":  "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b/answer_token",
    "baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline/answer_token",
    "delta":    "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta/answer_token",
}
OUT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"
MODELS = ["vanilla", "baseline", "delta"]
TASKS = ["shape_color", "obj_place"]
DIR_TO_LABEL = {"Down": 0, "Left": 1, "Right": 2, "Up": 3,
                "down": 0, "left": 1, "right": 2, "up": 3}
L_AXIS = 21


def load_axes(model, task):
    base = f"{HIDDEN_ROOTS[model]}/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{base}/features_layer_{L_AXIS}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{base}/labels.npy")
    g = h.mean(0)
    proto = np.stack([h[y == d].mean(0) for d in range(4)])
    Delta = proto - g[None, :]
    return Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)


def analyze(model, task):
    path = f"{CONTRIB_ROOT}/{model}_{task}_off0_lim2000.npz"
    d = np.load(path, allow_pickle=True)
    attn = d["attn_contribs"].astype(np.float32)
    mlp = d["mlp_contribs"].astype(np.float32)
    dirs = d["directions"]
    layers = d["layers"]
    corrects = d["corrects"]
    dir_idx = np.array([DIR_TO_LABEL.get(str(x), -1) for x in dirs])
    valid = dir_idx >= 0
    attn, mlp, dir_idx, corrects = attn[valid], mlp[valid], dir_idx[valid], corrects[valid]
    Delta_hat = load_axes(model, task)
    axis = Delta_hat[dir_idx]
    attn_proj = np.einsum("nld,nd->nl", attn, axis).mean(axis=0)
    mlp_proj = np.einsum("nld,nd->nl", mlp, axis).mean(axis=0)
    attn_norm = np.linalg.norm(attn, axis=2).mean(axis=0)
    mlp_norm = np.linalg.norm(mlp, axis=2).mean(axis=0)
    return {
        "layers": layers.tolist(),
        "attn_proj": attn_proj.tolist(),
        "mlp_proj": mlp_proj.tolist(),
        "attn_norm": attn_norm.tolist(),
        "mlp_norm": mlp_norm.tolist(),
        "acc": float(corrects.mean() * 100),
    }


def main():
    res = {}
    for m in MODELS:
        res[m] = {}
        for t in TASKS:
            res[m][t] = analyze(m, t)
            print(f"[{m}/{t}] acc={res[m][t]['acc']:.2f}%")

    with open(f"{OUT}/attn_mlp_3models.json", "w") as f:
        json.dump(res, f, indent=2)

    layers = res["baseline"]["shape_color"]["layers"]

    print("\n" + "=" * 110)
    print("ATTN direction projection @ each layer (⟨attn_out[-1], Δ̂_d⟩ averaged)")
    print("=" * 110)
    header = f"{'L':>4} |"
    for t in TASKS:
        for m in MODELS: header += f" {m[:2].upper()}_{t[:2]}  |"
    print(header)
    for i, L in enumerate(layers):
        row = f" {L:<3} |"
        for t in TASKS:
            for m in MODELS:
                row += f" {res[m][t]['attn_proj'][i]:>7.3f} |"
        print(row)

    print("\n" + "=" * 110)
    print("MLP direction projection @ each layer")
    print("=" * 110)
    print(header)
    for i, L in enumerate(layers):
        row = f" {L:<3} |"
        for t in TASKS:
            for m in MODELS:
                row += f" {res[m][t]['mlp_proj'][i]:>7.3f} |"
        print(row)

    print("\n" + "=" * 110)
    print("TOTAL (attn + mlp) direction projection")
    print("=" * 110)
    print(header)
    for i, L in enumerate(layers):
        row = f" {L:<3} |"
        for t in TASKS:
            for m in MODELS:
                tot = res[m][t]['attn_proj'][i] + res[m][t]['mlp_proj'][i]
                row += f" {tot:>7.3f} |"
        print(row)

    print("\n" + "=" * 110)
    print("L19 FOCUS (the amplifier layer)")
    print("=" * 110)
    L19_i = layers.index(19)
    L20_i = layers.index(20)
    print(f"{'Model':>10} | {'Task':>12} | {'L19_attn':>10} | {'L19_mlp':>10} | {'L19_tot':>10} | {'L20_attn':>10} | {'L20_mlp':>10} | {'L20_tot':>10} | acc")
    for m in MODELS:
        for t in TASKS:
            r = res[m][t]
            l19t = r['attn_proj'][L19_i] + r['mlp_proj'][L19_i]
            l20t = r['attn_proj'][L20_i] + r['mlp_proj'][L20_i]
            print(f"{m:>10} | {t:>12} | {r['attn_proj'][L19_i]:>10.3f} | {r['mlp_proj'][L19_i]:>10.3f} | "
                  f"{l19t:>10.3f} | {r['attn_proj'][L20_i]:>10.3f} | {r['mlp_proj'][L20_i]:>10.3f} | "
                  f"{l20t:>10.3f} | {r['acc']:.1f}")


if __name__ == "__main__":
    main()
