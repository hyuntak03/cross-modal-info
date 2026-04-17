"""
Layer x module direction amplification map.

For each sample i with direction d_i:
  attn_proj[i, L] = <attn_out_L[-1][i], Δ̂_d_i>
  mlp_proj[i, L]  = <mlp_out_L[-1][i],  Δ̂_d_i>

Δ̂_d is computed from the cached R2R 1500 last-token hiddens (already extracted).
We use the Baseline final layer (L=27) Δ axis as canonical "direction axis",
but since amplification is per-layer, it's more informative to use the
L=27 axis (readout target) OR per-layer Δ̂ from the layer's own hidden.

Simpler: use the SAME axis (L=21 hidden's Δ̂_d) for all layer projections.
This answers: "how much does each layer's attn/mlp push the last token
toward the direction axis the readout uses?"
"""
import os, json, glob
import numpy as np

CONTRIB_ROOT = "/local_datasets/vlm_direction/attn_mlp_contrib"
HIDDEN_ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline/answer_token"
OUT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"

TASKS = ["shape_color", "obj_place"]
LABEL_NAMES = ["Down", "Left", "Right", "Up"]  # label idx order
DIR_TO_LABEL = {"Down": 0, "Left": 1, "Right": 2, "Up": 3,
                "down": 0, "left": 1, "right": 2, "up": 3}
L_AXIS = 21  # canonical direction axis layer (peak task-invariance)


def load_direction_axes(task):
    """Δ̂_d (4, D) from L=21 baseline hidden."""
    base = f"{HIDDEN_ROOT}/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{base}/features_layer_{L_AXIS}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{base}/labels.npy")
    g = h.mean(0)
    proto = np.stack([h[y == d].mean(0) for d in range(4)])
    Delta = proto - g[None, :]
    Delta_hat = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)
    return Delta_hat  # (4, D)


def analyze(task):
    path = f"{CONTRIB_ROOT}/baseline_{task}_off0_lim2000.npz"
    d = np.load(path, allow_pickle=True)
    attn = d["attn_contribs"].astype(np.float32)  # (N, nL, D)
    mlp = d["mlp_contribs"].astype(np.float32)
    dirs = d["directions"]
    layers = d["layers"]
    corrects = d["corrects"]

    dir_idx = np.array([DIR_TO_LABEL.get(str(x), -1) for x in dirs])
    valid = dir_idx >= 0
    attn, mlp, dir_idx, corrects = attn[valid], mlp[valid], dir_idx[valid], corrects[valid]

    Delta_hat = load_direction_axes(task)  # (4, D)

    # Per-sample per-layer projection onto its own direction axis
    axis_per_sample = Delta_hat[dir_idx]  # (N, D)
    attn_proj = np.einsum("nld,nd->nl", attn, axis_per_sample)  # (N, nL)
    mlp_proj = np.einsum("nld,nd->nl", mlp, axis_per_sample)
    total_proj = attn_proj + mlp_proj

    # Also: contribution relative to the axis magnitude (||attn_out||·cos θ)
    attn_norm = np.linalg.norm(attn, axis=2)  # (N, nL)
    mlp_norm = np.linalg.norm(mlp, axis=2)

    # Metrics per layer
    out = {"layers": layers.tolist()}
    out["attn_proj_mean"] = attn_proj.mean(axis=0).tolist()      # avg direction push
    out["mlp_proj_mean"] = mlp_proj.mean(axis=0).tolist()
    out["attn_proj_std"] = attn_proj.std(axis=0).tolist()        # variability
    out["mlp_proj_std"] = mlp_proj.std(axis=0).tolist()
    out["attn_norm_mean"] = attn_norm.mean(axis=0).tolist()      # total module update size
    out["mlp_norm_mean"] = mlp_norm.mean(axis=0).tolist()
    # Fraction of module energy aligned with direction axis
    out["attn_align_frac"] = (attn_proj.mean(axis=0) / (attn_norm.mean(axis=0) + 1e-9)).tolist()
    out["mlp_align_frac"] = (mlp_proj.mean(axis=0) / (mlp_norm.mean(axis=0) + 1e-9)).tolist()
    # Cumulative direction amplification (projection sum from L10 onwards)
    cumul_attn = attn_proj.cumsum(axis=1).mean(axis=0)
    cumul_mlp = mlp_proj.cumsum(axis=1).mean(axis=0)
    out["cumul_attn"] = cumul_attn.tolist()
    out["cumul_mlp"] = cumul_mlp.tolist()
    out["cumul_total"] = (cumul_attn + cumul_mlp).tolist()
    # Per-sample correlation of direction projection with correctness (L21)
    L21_idx = list(layers).index(21)
    out["attn_L21_corr_mean"] = float(attn_proj[corrects == 1, L21_idx].mean()) if (corrects == 1).any() else 0.0
    out["attn_L21_wrong_mean"] = float(attn_proj[corrects == 0, L21_idx].mean()) if (corrects == 0).any() else 0.0
    out["mlp_L21_corr_mean"] = float(mlp_proj[corrects == 1, L21_idx].mean()) if (corrects == 1).any() else 0.0
    out["mlp_L21_wrong_mean"] = float(mlp_proj[corrects == 0, L21_idx].mean()) if (corrects == 0).any() else 0.0
    out["acc"] = float(corrects.mean() * 100)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {task: analyze(task) for task in TASKS}
    with open(f"{OUT}/attn_mlp_contrib_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Tables
    layers = results[TASKS[0]]["layers"]
    print("=" * 100)
    print("Per-layer DIRECTION AXIS PROJECTION MEAN (⟨module_out[-1], Δ̂_d⟩, avg over samples)")
    print(f"Axis = baseline L=21 Δ̂_d. Positive = pushes toward direction axis.")
    print("=" * 100)

    for task in TASKS:
        r = results[task]
        print(f"\n--- {task} (acc={r['acc']:.2f}%) ---")
        print(f"{'L':>4} | {'attn':>10} | {'mlp':>10} | {'total':>10} | {'attn_norm':>10} | {'mlp_norm':>10} | {'attn_align':>10} | {'mlp_align':>10}")
        for i, L in enumerate(layers):
            print(f" {L:<3} | {r['attn_proj_mean'][i]:>10.4f} | {r['mlp_proj_mean'][i]:>10.4f} | "
                  f"{r['attn_proj_mean'][i]+r['mlp_proj_mean'][i]:>10.4f} | "
                  f"{r['attn_norm_mean'][i]:>10.3f} | {r['mlp_norm_mean'][i]:>10.3f} | "
                  f"{r['attn_align_frac'][i]:>10.4f} | {r['mlp_align_frac'][i]:>10.4f}")

    print("\n" + "=" * 100)
    print("CUMULATIVE DIRECTION PROJECTION (sum of per-layer projections, L10→L25)")
    print("=" * 100)
    print(f"{'L':>4} |  SC_cumul_attn  SC_cumul_mlp  SC_total  |  OP_cumul_attn  OP_cumul_mlp  OP_total")
    for i, L in enumerate(layers):
        sc = results["shape_color"]
        op = results["obj_place"]
        print(f" {L:<3} |  {sc['cumul_attn'][i]:>13.3f}  {sc['cumul_mlp'][i]:>12.3f}  {sc['cumul_total'][i]:>8.3f}  |  "
              f"{op['cumul_attn'][i]:>13.3f}  {op['cumul_mlp'][i]:>12.3f}  {op['cumul_total'][i]:>8.3f}")

    print("\n" + "=" * 100)
    print("IN vs OOD attn/mlp gap per layer (SC - OP)")
    print("=" * 100)
    print(f"{'L':>4} | {'attn_gap':>10} | {'mlp_gap':>10} | {'total_gap':>10}")
    for i, L in enumerate(layers):
        sc = results["shape_color"]
        op = results["obj_place"]
        ag = sc["attn_proj_mean"][i] - op["attn_proj_mean"][i]
        mg = sc["mlp_proj_mean"][i] - op["mlp_proj_mean"][i]
        print(f" {L:<3} | {ag:>10.4f} | {mg:>10.4f} | {ag+mg:>10.4f}")

    print("\n" + "=" * 100)
    print("L21 correct vs wrong sample comparison (direction push)")
    print("=" * 100)
    for task in TASKS:
        r = results[task]
        print(f"  {task:>13s}: attn_corr={r['attn_L21_corr_mean']:.3f} attn_wrong={r['attn_L21_wrong_mean']:.3f}  "
              f"mlp_corr={r['mlp_L21_corr_mean']:.3f} mlp_wrong={r['mlp_L21_wrong_mean']:.3f}")


if __name__ == "__main__":
    main()
