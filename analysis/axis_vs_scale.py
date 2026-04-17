"""
Direction axis alignment vs scale 분리 분석.

Class mean 기반 canonical direction axis:
  v_UD = mean(h | Up) - mean(h | Down)
  v_LR = mean(h | Left) - mean(h | Right)

이 axis는 probe와 달리 수학적으로 유일 (class centroid 차이).

Task pair (A, B)에 대해:
  - Axis alignment: cos(v̂_A, v̂_B) — 방향 비교 (normalized)
  - Scale: ||v_A|| vs ||v_B|| — centroid 간 거리 비교
"""

import os, json
import numpy as np
import torch

FEAT_ROOTS = {
    "Vanilla": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b",
    "Baseline": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "/data3/local_datasets/vlm_direction/linear_probing_1500/llava-video-7b_lora_4combo_v2_delta",
}

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"

# LabelEncoder order: ['Down', 'Left', 'Right', 'Up']
IDX = {"Down": 0, "Left": 1, "Right": 2, "Up": 3}

STAGES = [
    ("vision_encoder", None, "VE"),
    ("after_projector", None, "AP"),
    ("vision_token", 0, "vt_L0"),
    ("vision_token", 14, "vt_L14"),
    ("vision_token", 27, "vt_L27"),
    ("answer_token", 14, "at_L14"),
    ("answer_token", 27, "at_L27"),
]


def load_feat(feat_root, task, stage, layer=None):
    if stage in ("vision_encoder", "after_projector"):
        d = os.path.join(feat_root, stage, TASK_FULL(task))
        feat = np.load(os.path.join(d, "features.npy")).astype(np.float32)
    else:
        d = os.path.join(feat_root, stage, TASK_FULL(task))
        feat = np.load(os.path.join(d, f"features_layer_{layer}.npy")).astype(np.float32)
    labels = np.load(os.path.join(d, "labels.npy"))
    return feat, labels


def canonical_axes(feat, labels):
    """Canonical Up-Down and Left-Right axes from class means."""
    mean_up = feat[labels == IDX["Up"]].mean(0)
    mean_dn = feat[labels == IDX["Down"]].mean(0)
    mean_lt = feat[labels == IDX["Left"]].mean(0)
    mean_rt = feat[labels == IDX["Right"]].mean(0)
    v_UD = mean_up - mean_dn  # unnormalized
    v_LR = mean_lt - mean_rt
    return v_UD, v_LR


def analyze(feat_root, stage, layer=None):
    axes = {}
    for task in TASKS:
        feat, labels = load_feat(feat_root, task, stage, layer)
        v_UD, v_LR = canonical_axes(feat, labels)
        axes[task] = {
            "v_UD": v_UD, "v_LR": v_LR,
            "scale_UD": float(np.linalg.norm(v_UD)),
            "scale_LR": float(np.linalg.norm(v_LR)),
        }

    # Pairwise comparison
    n = len(TASKS)
    align_UD = np.zeros((n, n)); align_LR = np.zeros((n, n))
    scale_UD = np.zeros((n, n)); scale_LR = np.zeros((n, n))
    for i, t1 in enumerate(TASKS):
        for j, t2 in enumerate(TASKS):
            v1u, v2u = axes[t1]["v_UD"], axes[t2]["v_UD"]
            v1l, v2l = axes[t1]["v_LR"], axes[t2]["v_LR"]
            align_UD[i, j] = v1u @ v2u / (np.linalg.norm(v1u) * np.linalg.norm(v2u) + 1e-8)
            align_LR[i, j] = v1l @ v2l / (np.linalg.norm(v1l) * np.linalg.norm(v2l) + 1e-8)
            s1u, s2u = axes[t1]["scale_UD"], axes[t2]["scale_UD"]
            s1l, s2l = axes[t1]["scale_LR"], axes[t2]["scale_LR"]
            # Scale ratio (≥1)
            scale_UD[i, j] = max(s1u, s2u) / max(min(s1u, s2u), 1e-8)
            scale_LR[i, j] = max(s1l, s2l) / max(min(s1l, s2l), 1e-8)

    off = ~np.eye(n, dtype=bool)
    return {
        "axes": {t: {"scale_UD": axes[t]["scale_UD"], "scale_LR": axes[t]["scale_LR"]} for t in TASKS},
        "alignment_UD": align_UD.tolist(),
        "alignment_LR": align_LR.tolist(),
        "scale_ratio_UD": scale_UD.tolist(),
        "scale_ratio_LR": scale_LR.tolist(),
        "align_UD_off_mean": float(align_UD[off].mean()),
        "align_LR_off_mean": float(align_LR[off].mean()),
        "scale_UD_off_mean": float(scale_UD[off].mean()),
        "scale_LR_off_mean": float(scale_LR[off].mean()),
    }


def main():
    results = {}
    for model_name, feat_root in FEAT_ROOTS.items():
        if not os.path.exists(feat_root):
            continue
        print(f"\n{'='*80}\n  {model_name}\n{'='*80}")
        results[model_name] = {}
        print(f"  {'Stage':>10} | {'align_UD':>8} {'align_LR':>8} | "
              f"{'scale_UD':>8} {'scale_LR':>8} | {'verdict':>30}")
        for stage, layer, label in STAGES:
            try:
                r = analyze(feat_root, stage, layer)
            except FileNotFoundError:
                continue
            results[model_name][label] = r

            a_ud = r["align_UD_off_mean"]
            a_lr = r["align_LR_off_mean"]
            s_ud = r["scale_UD_off_mean"]
            s_lr = r["scale_LR_off_mean"]

            # Verdict
            avg_align = (a_ud + a_lr) / 2
            avg_scale = (s_ud + s_lr) / 2
            if avg_align > 0.7:
                if avg_scale > 1.5:
                    verdict = "same axis, DIFFERENT SCALE"
                else:
                    verdict = "same axis, similar scale"
            elif avg_align > 0.3:
                verdict = "partial axis alignment"
            else:
                verdict = "DIFFERENT AXES (rotation)"

            print(f"  {label:>10} | {a_ud:>+8.3f} {a_lr:>+8.3f} | "
                  f"{s_ud:>8.2f}× {s_lr:>8.2f}× | {verdict:>30}")

    with open("analysis/axis_vs_scale.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] analysis/axis_vs_scale.json")


if __name__ == "__main__":
    main()
