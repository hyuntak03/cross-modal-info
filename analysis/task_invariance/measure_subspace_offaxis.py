"""
Subspace alignment (principal angles) + on-axis vs off-axis energy decomposition.

Adds two missing measurements to the per-axis cosine analysis:
  (1) Subspace alignment: 4-direction Δ vectors form a (rank ≤ 3) subspace per task.
      Compare via principal angles (SVD of U_A^T U_B). All cos≈1 → subspace aligned.
  (2) Off-axis variance: for each sample, decompose (h_i - g) into on-axis + off-axis.
      on_axis_energy = (h_i - g)·Δ̂_{d_i}  squared
      off_axis_energy = ||(h_i - g) - on_axis · Δ̂_{d_i}||^2
      ratio = on / (on + off) — signal vs total.
      Lower in OOD → direction info dispersed off-canonical-axis.

  (3) PCA on within-direction residuals: top-1 PC explained variance + alignment with
      canonical axis. Reveals whether OOD residuals have structured off-axis direction
      info (sub-axis split) vs isotropic noise.

Layers: focus on binding region L18, L21, L24, L27.
"""
import os, json, numpy as np
from itertools import combinations
from numpy.linalg import svd

ROOT = "/data3/local_datasets/vlm_direction/linear_probing_1500"
MODELS = {
    "vanilla":  "llava-video-7b",
    "baseline": "llava-video-7b_lora_4combo_v2_baseline",
    "delta":    "llava-video-7b_lora_4combo_v2_delta",
}
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
LAYERS = [14, 18, 21, 24, 27]
LABEL_NAMES = ["Down", "Left", "Right", "Up"]
OUT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/task_invariance"


def load(model, task, L):
    p = f"{ROOT}/{MODELS[model]}/answer_token/vlm_direction_testbed_R2R_4way_1500_{task}"
    h = np.load(f"{p}/features_layer_{L}.npy", mmap_mode="r").astype(np.float32)
    y = np.load(f"{p}/labels.npy")
    return h, y


def task_signature(h, y):
    """Returns g (D,), Delta (4,D), Delta_hat (4,D), U_subspace (D,3) orthonormal basis of centered Δ span."""
    g = h.mean(0)
    proto = np.stack([h[y == d].mean(0) for d in range(4)])
    Delta = proto - g[None, :]
    Delta_hat = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)
    # Center the 4 Δ vectors and SVD → orthonormal basis (rank ≤ 3)
    Delta_centered = Delta - Delta.mean(0, keepdims=True)
    U, S, _ = svd(Delta_centered, full_matrices=False)  # Delta_centered: (4, D)
    # Wait — SVD of (4,D): we want column basis of D-dim space spanned by rows
    # Use right singular vectors instead
    Vt = np.linalg.svd(Delta_centered, full_matrices=False)[2]  # (3 or 4, D)
    rank = (S > 1e-6).sum()
    U_basis = Vt[:rank].T  # (D, rank), orthonormal columns
    return dict(g=g, Delta=Delta, Delta_hat=Delta_hat, U_basis=U_basis,
                singular_values=S[:rank].tolist())


def principal_angles(U_A, U_B):
    """Returns cosines of principal angles between two subspaces (orthonormal column bases)."""
    M = U_A.T @ U_B  # (rank_A, rank_B)
    s = np.linalg.svd(M, compute_uv=False)
    return s.tolist()


def on_off_axis(h, y, g, Delta_hat):
    """Per-sample decomposition: on_axis_energy, off_axis_energy."""
    centered = h - g[None, :]
    Δ_per_sample = Delta_hat[y]  # (n, D)
    on_axis = (centered * Δ_per_sample).sum(axis=1)  # scalar projection
    on_axis_energy = on_axis ** 2
    on_axis_vec = on_axis[:, None] * Δ_per_sample  # (n, D)
    off_axis_vec = centered - on_axis_vec
    off_axis_energy = (off_axis_vec ** 2).sum(axis=1)
    return on_axis_energy, off_axis_energy


def residual_pca_per_dir(h, y, g, Delta_hat, k=5):
    """For each direction, PCA on residual (h - g - on_axis * Δ̂_d).
    Returns top-k explained variance ratio + cos with canonical axis."""
    out = {}
    for d in range(4):
        mask = y == d
        H_d = h[mask]
        centered = H_d - g[None, :]
        on_axis = (centered @ Delta_hat[d])[:, None] * Delta_hat[d][None, :]
        residual = centered - on_axis  # off-axis component
        # PCA via SVD on residual
        U, S, Vt = svd(residual, full_matrices=False)
        var = S ** 2 / (residual.shape[0] - 1)
        total_var = var.sum()
        topk_ratio = (var[:k] / (total_var + 1e-9)).tolist()
        # Alignment of top PCs with canonical axis Δ_hat_d
        topk_cos_canonical = [float(abs(Vt[i] @ Delta_hat[d])) for i in range(min(k, Vt.shape[0]))]
        out[LABEL_NAMES[d]] = {
            "top_k_var_ratio": topk_ratio,
            "top_k_cos_canonical": topk_cos_canonical,
            "total_var": float(total_var),
        }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    full = {}

    for model in MODELS:
        full[model] = {}
        for L in LAYERS:
            print(f"[{model} L{L}]", flush=True)
            sigs = {}
            on_off = {}
            res_pca = {}
            for task in TASKS:
                h, y = load(model, task, L)
                sig = task_signature(h, y)
                sigs[task] = sig
                on_e, off_e = on_off_axis(h, y, sig["g"], sig["Delta_hat"])
                on_off[task] = {
                    "on_axis_energy_mean": float(on_e.mean()),
                    "off_axis_energy_mean": float(off_e.mean()),
                    "on_off_ratio": float(on_e.mean() / (on_e.mean() + off_e.mean())),
                    "on_axis_std": float(np.sqrt(on_e).std()),
                }
                res_pca[task] = residual_pca_per_dir(h, y, sig["g"], sig["Delta_hat"])

            # Subspace principal angles per task pair
            subspace = {}
            for tA, tB in combinations(TASKS, 2):
                cos_angles = principal_angles(sigs[tA]["U_basis"], sigs[tB]["U_basis"])
                subspace[f"{tA}__{tB}"] = {
                    "cos_principal_angles": cos_angles,
                    "min_cos": float(min(cos_angles)),
                    "mean_cos": float(np.mean(cos_angles)),
                }
            full[model][f"L{L}"] = {
                "subspace_alignment": subspace,
                "on_off_axis": on_off,
                "residual_pca": res_pca,
            }

    out_path = f"{OUT}/subspace_offaxis_metrics.json"
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\n[SAVED] {out_path}\n")

    # ===== Reports =====
    print("=" * 100)
    print("SUBSPACE PRINCIPAL ANGLES — cos values (1.0 = perfectly aligned subspace)")
    print("All 3 values close to 1 → 3D direction subspace is task-invariant")
    print("=" * 100)
    for model in MODELS:
        print(f"\n--- {model} ---")
        for L in LAYERS:
            sub = full[model][f"L{L}"]["subspace_alignment"]
            mean_min = np.mean([v["min_cos"] for v in sub.values()])
            mean_avg = np.mean([v["mean_cos"] for v in sub.values()])
            print(f"  L{L:<3}  mean of (min cos): {mean_min:.3f}   mean of (avg cos): {mean_avg:.3f}")

    print("\n" + "=" * 100)
    print("ON-AXIS / TOTAL ENERGY RATIO (= signal in canonical axis vs total dispersion)")
    print("Lower in OOD → direction info more dispersed off-axis")
    print("=" * 100)
    for model in MODELS:
        print(f"\n--- {model} ---  (on / (on + off))")
        print(f"{'L':>4} | {'SC':>6} | {'OC':>6} | {'SP':>6} | {'OP':>6}")
        for L in LAYERS:
            row = f" L{L:<2} |"
            for t in TASKS:
                r = full[model][f"L{L}"]["on_off_axis"][t]["on_off_ratio"]
                row += f" {r:>6.4f} |"
            print(row)

    print("\n" + "=" * 100)
    print("RESIDUAL PCA — top-1 PC explained variance ratio (avg over 4 directions) @ L21")
    print("Higher → structured off-axis dim (sub-axis hypothesis support)")
    print("Lower → isotropic noise")
    print("=" * 100)
    for model in MODELS:
        print(f"\n--- {model} L21 ---")
        for t in TASKS:
            rpca = full[model]["L21"]["residual_pca"][t]
            top1_vars = [rpca[d]["top_k_var_ratio"][0] for d in LABEL_NAMES]
            top1_cos = [rpca[d]["top_k_cos_canonical"][0] for d in LABEL_NAMES]
            print(f"  {t:>13s} : top1 var ratio = {np.mean(top1_vars):.4f}  "
                  f"|cos with canonical axis| = {np.mean(top1_cos):.4f}")


if __name__ == "__main__":
    main()
