"""
Pure direction axis validation.

Claim: Δ_d = h_avg(dir=d) - h_avg(all) is "pure direction" axis with
letter/identity/bg/instance cancelled via factorial design.

Direct tests:
  1. Variant invariance: Δ_d computed per-variant should be consistent
     → proves letter bias cancellation
  2. Identity invariance: Δ_d computed per-obj should be consistent
     → proves identity cancellation
  3. Bg invariance: Δ_d computed per-bg should be consistent
  4. Split-half consistency: random split should give same axis
     → proves direction axis is sample-stable

Test at L21 (canonical axis layer) for Baseline obj_place.
"""
import numpy as np
import glob
from itertools import combinations

HIDDENS_ROOT = "/local_datasets/vlm_direction/factorial_dataset/hiddens"


def load_factorial(cond):
    arr = {"hiddens": [], "directions": [], "identities": [], "bgs": [],
            "variant_ids": [], "instance_idxs": []}
    for f in sorted(glob.glob(f"{HIDDENS_ROOT}/baseline_{cond}_4variants*.npz")):
        d = np.load(f, allow_pickle=True)
        for k in arr:
            if k in d.files:
                arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items() if v}


def direction_axis(H_L, mask, dirs):
    """Compute Δ_d at layer L using only samples in mask."""
    H_sub = H_L[mask]
    dirs_sub = dirs[mask]
    g = H_sub.mean(0)
    out = {}
    for d in ["up", "right", "down", "left"]:
        dmask = dirs_sub == d
        if dmask.sum() == 0:
            continue
        h_avg = H_sub[dmask].mean(0)
        Delta = h_avg - g
        out[d] = Delta
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))


def main():
    print("[load factorial OP]")
    data = load_factorial("obj_place")
    H = data["hiddens"].astype(np.float32)  # (N, 28, 3584)
    dirs = data["directions"]
    ids = data["identities"]
    bgs = data["bgs"]
    variants = data["variant_ids"]
    insts = data["instance_idxs"]

    L = 21  # canonical axis layer
    H_L = H[:, L, :]

    print(f"\n=== Sample breakdown ===")
    print(f"Total samples: {len(dirs)}")
    print(f"Directions: {dict(zip(*np.unique(dirs, return_counts=True)))}")
    print(f"Identities: {dict(zip(*np.unique(ids, return_counts=True)))}")
    print(f"Bgs: {dict(zip(*np.unique(bgs, return_counts=True)))}")
    print(f"Variants: {dict(zip(*np.unique(variants, return_counts=True)))}")

    # Reference: full Δ_d
    full_mask = np.ones(len(dirs), dtype=bool)
    delta_full = direction_axis(H_L, full_mask, dirs)

    # === Test 1: Variant invariance (letter bias test) ===
    print(f"\n{'='*80}")
    print("TEST 1: Variant invariance (letter bias test)")
    print("Compute Δ_d within each variant separately. If letter cancelled,")
    print("  Δ_d^variant_i should be consistent across variants (cos → 1).")
    print(f"{'='*80}")
    delta_by_var = {}
    for v in sorted(np.unique(variants)):
        v_mask = variants == v
        delta_by_var[v] = direction_axis(H_L, v_mask, dirs)

    print(f"\nPairwise cos across variants, per direction:")
    for d in ["up", "right", "down", "left"]:
        print(f"\n  {d}:")
        cos_list = []
        for v1, v2 in combinations(sorted(delta_by_var.keys()), 2):
            c = cos(delta_by_var[v1][d], delta_by_var[v2][d])
            cos_list.append(c)
            print(f"    variant{v1} vs variant{v2}: cos = {c:.4f}")
        print(f"    mean: {np.mean(cos_list):.4f}")
    print()
    # Overall mean
    all_cos = [cos(delta_by_var[v1][d], delta_by_var[v2][d])
               for v1, v2 in combinations(sorted(delta_by_var.keys()), 2)
               for d in ["up", "right", "down", "left"]]
    print(f"  OVERALL mean cos across variants: {np.mean(all_cos):.4f}")

    # === Test 2: Identity invariance ===
    print(f"\n{'='*80}")
    print("TEST 2: Identity (obj) invariance test")
    print("Compute Δ_d within each obj class separately. If identity cancelled,")
    print("  Δ_d should be consistent across obj classes (cos → 1).")
    print(f"{'='*80}")
    delta_by_id = {}
    for i in sorted(np.unique(ids)):
        i_mask = ids == i
        delta_by_id[i] = direction_axis(H_L, i_mask, dirs)

    print(f"\nPairwise cos across identities, per direction:")
    for d in ["up", "right", "down", "left"]:
        print(f"\n  {d}:")
        cos_list = []
        for i1, i2 in combinations(sorted(delta_by_id.keys()), 2):
            c = cos(delta_by_id[i1][d], delta_by_id[i2][d])
            cos_list.append(c)
        print(f"    {len(cos_list)} pairs, mean cos: {np.mean(cos_list):.4f}  "
              f"(min {min(cos_list):.4f}, max {max(cos_list):.4f})")
    all_cos = [cos(delta_by_id[i1][d], delta_by_id[i2][d])
               for i1, i2 in combinations(sorted(delta_by_id.keys()), 2)
               for d in ["up", "right", "down", "left"]]
    print(f"\n  OVERALL mean cos across identities: {np.mean(all_cos):.4f}")

    # === Test 3: Bg invariance ===
    print(f"\n{'='*80}")
    print("TEST 3: Bg invariance test")
    print(f"{'='*80}")
    delta_by_bg = {}
    for b in sorted(np.unique(bgs)):
        b_mask = bgs == b
        delta_by_bg[b] = direction_axis(H_L, b_mask, dirs)

    print(f"\nPairwise cos across bgs, per direction:")
    for d in ["up", "right", "down", "left"]:
        cos_list = []
        for b1, b2 in combinations(sorted(delta_by_bg.keys()), 2):
            c = cos(delta_by_bg[b1][d], delta_by_bg[b2][d])
            cos_list.append(c)
        print(f"  {d}: {len(cos_list)} pairs, mean cos: {np.mean(cos_list):.4f}  "
              f"(min {min(cos_list):.4f}, max {max(cos_list):.4f})")
    all_cos = [cos(delta_by_bg[b1][d], delta_by_bg[b2][d])
               for b1, b2 in combinations(sorted(delta_by_bg.keys()), 2)
               for d in ["up", "right", "down", "left"]]
    print(f"\n  OVERALL mean cos across bgs: {np.mean(all_cos):.4f}")

    # === Test 4: Split-half consistency ===
    print(f"\n{'='*80}")
    print("TEST 4: Split-half consistency (sample stability)")
    print("Random split → Δ_d from each half → cos")
    print(f"{'='*80}")
    np.random.seed(0)
    for trial in range(5):
        idx = np.random.permutation(len(dirs))
        half = len(idx) // 2
        mask_a = np.zeros(len(dirs), dtype=bool)
        mask_a[idx[:half]] = True
        mask_b = ~mask_a
        delta_a = direction_axis(H_L, mask_a, dirs)
        delta_b = direction_axis(H_L, mask_b, dirs)
        cos_list = [cos(delta_a[d], delta_b[d]) for d in ["up", "right", "down", "left"]]
        print(f"  Trial {trial}: per-dir cos = {[f'{c:.4f}' for c in cos_list]}  "
              f"mean {np.mean(cos_list):.4f}")

    # === Test 5: Full Δ_d vs subset Δ_d (anchor check) ===
    print(f"\n{'='*80}")
    print("TEST 5: Each subset-Δ_d vs full-Δ_d")
    print("If subset Δ_d ≈ full Δ_d, subset-averaging captures the same direction.")
    print(f"{'='*80}")
    print("\nVariant subsets vs full:")
    for v in sorted(delta_by_var.keys()):
        cos_list = [cos(delta_by_var[v][d], delta_full[d]) for d in ["up", "right", "down", "left"]]
        print(f"  variant{v}: mean cos with full = {np.mean(cos_list):.4f}")
    print("\nIdentity subsets vs full:")
    for i in sorted(delta_by_id.keys()):
        cos_list = [cos(delta_by_id[i][d], delta_full[d]) for d in ["up", "right", "down", "left"]]
        print(f"  obj={i}: mean cos with full = {np.mean(cos_list):.4f}")
    print("\nBg subsets vs full:")
    for b in sorted(delta_by_bg.keys()):
        cos_list = [cos(delta_by_bg[b][d], delta_full[d]) for d in ["up", "right", "down", "left"]]
        print(f"  bg={b}: mean cos with full = {np.mean(cos_list):.4f}")

    # === Test 6: Letter-correlated bias (variant+direction → letter) ===
    # For variant=0, each direction maps to specific letter. If Δ_d picks up letter bias,
    # within-variant Δ_d should differ from across-variant Δ_d (letter signal).
    print(f"\n{'='*80}")
    print("TEST 6: Letter bias direct test")
    print("Δ_d_variant_0 vs Δ_d_all_variants")
    print(f"  Large diff → variant-specific Δ includes letter signal")
    print(f"  Small diff → Δ is letter-free")
    print(f"{'='*80}")
    for d in ["up", "right", "down", "left"]:
        # Δ_d from variant 0 only
        v0_Delta = delta_by_var[0][d]
        # Δ_d from all variants
        all_Delta = delta_full[d]
        # Difference
        diff = v0_Delta - all_Delta
        # Normalize both and compare
        v0_hat = v0_Delta / (np.linalg.norm(v0_Delta) + 1e-9)
        all_hat = all_Delta / (np.linalg.norm(all_Delta) + 1e-9)
        c = cos(v0_Delta, all_Delta)
        rel_diff = np.linalg.norm(diff) / np.linalg.norm(all_Delta)
        print(f"  {d}: cos(variant_0_Δ, full_Δ) = {c:.4f}, relative diff = {rel_diff:.4f}")


if __name__ == "__main__":
    main()
