"""
SigLIP vision tokens에 position / direction / object dim이 어떻게 분리되어 있는지 검증.

우리 데이터 (R2R_4way_1500) + 3 모델 (Vanilla/Baseline/Delta) × VE/AP에서:
  - Position R² (first frame → start_pos 회귀)
  - Direction probe acc (last-first delta → 4-class)
  - Object probe acc (mean feature → shape/obj_class)
  - Top-50 dim 식별 및 overlap 분석
  - Cross-task top-50 overlap (핵심: position dim이 task-universal인가 task-specific인가)
"""

import os, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter, defaultdict

FEAT_ROOTS = {
    "Vanilla": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b",
    "Baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta",
}

META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_video_1500"

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"

IDENTITY_KEYS = {
    "shape_color": "shape",
    "obj_color": "obj_class",
    "shape_place": "shape",
    "obj_place": "obj_class",
}

STAGES = [
    ("vision_encoder", 1152),
    ("after_projector", 3584),
]

T = 8  # num frames
TOP_K = 50


# ============================================================
#  Loading
# ============================================================

def load_features(feat_root, stage, task, D):
    """Load (N, T, D) features. qids는 vision_token 디렉토리에서 공유 로드."""
    d = os.path.join(feat_root, stage, TASK_FULL(task))
    feat = np.load(os.path.join(d, "features.npy")).astype(np.float32)
    # qids는 vision_token/에 저장되어 있음 (모든 stage 공유)
    qids_path = os.path.join(feat_root, "vision_token", TASK_FULL(task), "qids.npy")
    qids = np.load(qids_path)
    N = feat.shape[0]
    assert feat.shape[1] == T * D, f"Shape mismatch: {feat.shape}, expected (N, {T*D})"
    feat = feat.reshape(N, T, D)
    return feat, qids


def load_metadata(task):
    """Load 1500 metadata, return dict {id: metadata}."""
    p = os.path.join(META_ROOT, f"{task}_metadata.json")
    data = json.load(open(p))
    return {m["id"]: m for m in data}


def align_samples(qids, metadata, task):
    """For each qid, get (start_pos, direction, identity). Returns aligned arrays."""
    id_key = IDENTITY_KEYS[task]
    start_pos = []
    directions = []
    identities = []
    for q in qids:
        sid = int(str(q).split('_')[0])
        m = metadata[sid]
        start_pos.append(m["start_pos"])  # [x, y]
        directions.append(m["direction"])
        identities.append(m[id_key])
    return (np.array(start_pos, dtype=np.float32),
            np.array(directions),
            np.array(identities))


# ============================================================
#  Probing
# ============================================================

def position_regression(feat_frame0, start_pos, test_ratio=0.3, seed=42):
    """Ridge regression for x, y. Returns R² per axis + coefficients."""
    np.random.seed(seed)
    n = len(feat_frame0)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    tr, te = idx[n_test:], idx[:n_test]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feat_frame0[tr])
    X_te = scaler.transform(feat_frame0[te])

    # Separate regression for x and y
    r2_xy = []
    importance = np.zeros(feat_frame0.shape[1])
    for axis_idx in range(2):
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, start_pos[tr, axis_idx])
        r2 = r2_score(start_pos[te, axis_idx], reg.predict(X_te))
        r2_xy.append(float(r2))
        importance += np.abs(reg.coef_)

    top_dims = np.argsort(importance)[-TOP_K:][::-1]
    return {"R2_x": r2_xy[0], "R2_y": r2_xy[1], "top_dims": top_dims.tolist()}


def direction_probe(feat_delta, directions, test_ratio=0.3, seed=42):
    """Linear classifier for direction (4-class). Returns acc + Fisher-top dims."""
    le = LabelEncoder()
    y = le.fit_transform(directions)
    nc = len(le.classes_)

    np.random.seed(seed)
    n = len(feat_delta)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    tr, te = idx[n_test:], idx[:n_test]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feat_delta[tr])
    X_te = scaler.transform(feat_delta[te])

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                             random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y[tr])
    acc = clf.score(X_te, y[te]) * 100

    # Fisher ratio per dim (on full data for robustness)
    fisher = np.zeros(feat_delta.shape[1])
    gm = feat_delta.mean(0)
    var_b = np.zeros(feat_delta.shape[1])
    var_w = np.zeros(feat_delta.shape[1])
    for c in range(nc):
        mask = y == c
        if mask.sum() == 0: continue
        Xc = feat_delta[mask]
        cm = Xc.mean(0)
        var_b += mask.sum() * (cm - gm) ** 2
        var_w += ((Xc - cm) ** 2).sum(0)
    fisher = var_b / (var_w + 1e-8)
    top_dims = np.argsort(fisher)[-TOP_K:][::-1]
    return {"acc": float(acc), "top_dims": top_dims.tolist()}


def object_probe(feat_mean, identities, test_ratio=0.3, seed=42):
    """Linear classifier for object/shape (multi-class)."""
    le = LabelEncoder()
    y = le.fit_transform(identities)
    nc = len(le.classes_)

    np.random.seed(seed)
    n = len(feat_mean)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    tr, te = idx[n_test:], idx[:n_test]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feat_mean[tr])
    X_te = scaler.transform(feat_mean[te])

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                             random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y[tr])
    acc = clf.score(X_te, y[te]) * 100

    # Fisher per dim
    gm = feat_mean.mean(0)
    var_b = np.zeros(feat_mean.shape[1])
    var_w = np.zeros(feat_mean.shape[1])
    for c in range(nc):
        mask = y == c
        if mask.sum() == 0: continue
        Xc = feat_mean[mask]
        cm = Xc.mean(0)
        var_b += mask.sum() * (cm - gm) ** 2
        var_w += ((Xc - cm) ** 2).sum(0)
    fisher = var_b / (var_w + 1e-8)
    top_dims = np.argsort(fisher)[-TOP_K:][::-1]
    return {"acc": float(acc), "n_classes": nc, "top_dims": top_dims.tolist()}


# ============================================================
#  Overlap analysis
# ============================================================

def overlap(dims_a, dims_b):
    return len(set(dims_a) & set(dims_b))


def expected_random(d_total):
    """Expected overlap between two random top-K sets of a D-dim space."""
    return TOP_K * TOP_K / d_total


# ============================================================
#  Main
# ============================================================

def analyze_one(feat_root, stage, D, task):
    """Per (model, stage, task): compute probes and top-50 dims."""
    feat, qids = load_features(feat_root, stage, task, D)
    meta = load_metadata(task)
    start_pos, directions, identities = align_samples(qids, meta, task)

    # Feature variants
    feat_frame0 = feat[:, 0, :]
    feat_mean = feat.mean(axis=1)
    feat_delta = feat[:, -1, :] - feat[:, 0, :]

    pos = position_regression(feat_frame0, start_pos)
    dir_ = direction_probe(feat_delta, directions)
    obj = object_probe(feat_mean, identities)

    return {"position": pos, "direction": dir_, "object": obj}


def main():
    results = {}
    for model_name, feat_root in FEAT_ROOTS.items():
        if not os.path.exists(feat_root):
            continue
        results[model_name] = {}
        print(f"\n{'='*90}\n  {model_name}\n{'='*90}")

        for stage, D in STAGES:
            print(f"\n  [Stage: {stage}, D={D}]")
            print(f"  {'Task':>14} | {'Pos R²':>7} | {'Dir acc':>7} | {'Obj acc':>7} (cls) | "
                  f"{'P∩D':>4} {'P∩O':>4} {'D∩O':>4}  (rand={expected_random(D):.2f})")

            stage_results = {}
            for task in TASKS:
                r = analyze_one(feat_root, stage, D, task)
                stage_results[task] = r

                # Within-task overlaps
                pod = overlap(r["position"]["top_dims"], r["direction"]["top_dims"])
                poo = overlap(r["position"]["top_dims"], r["object"]["top_dims"])
                doo = overlap(r["direction"]["top_dims"], r["object"]["top_dims"])

                r_pos = (r["position"]["R2_x"] + r["position"]["R2_y"]) / 2
                print(f"  {task:>14} | {r_pos:>6.3f} | {r['direction']['acc']:>6.1f}% | "
                      f"{r['object']['acc']:>5.1f}% ({r['object']['n_classes']:>2}) | "
                      f"{pod:>4} {poo:>4} {doo:>4}")

            results[model_name][stage] = stage_results

            # Cross-task overlaps (3가지 feature 모두)
            print(f"\n  [Cross-task top-{TOP_K} overlap, {stage}] (random = {expected_random(D):.2f})")
            for attr in ["position", "direction", "object"]:
                print(f"    {attr}:")
                print(f"      {'':>14}", end="")
                for t2 in TASKS:
                    print(f" {t2[:10]:>10}", end="")
                print()
                for t1 in TASKS:
                    print(f"      {t1:>14}", end="")
                    for t2 in TASKS:
                        d1 = stage_results[t1][attr]["top_dims"]
                        d2 = stage_results[t2][attr]["top_dims"]
                        print(f" {overlap(d1, d2):>10}", end="")
                    print()

    os.makedirs("analysis", exist_ok=True)
    # Convert numpy types for JSON
    def convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        return o

    with open("analysis/position_direction_object_dims.json", "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n[SAVED] analysis/position_direction_object_dims.json")


if __name__ == "__main__":
    main()
