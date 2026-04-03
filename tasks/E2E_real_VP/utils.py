import matplotlib
import os
import re
import sys
import numpy as np
import datetime
from loguru import logger as eval_logger

# ============================================================
# 설정
# ============================================================
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", "/local_datasets/vlm_direction")

DIRECTION_CLASSES = ["up", "down", "left", "right"]
DIRECTION_LABELS = ["Up", "Down", "Left", "Right"]

# E2E_Colored_edge: 색상 → 방향 매핑 (영상 레이아웃 기준)
# 필요시 여기서 수정
EDGE_COLOR_TO_DIRECTION = {
    "red": "up",
    "blue": "down",
    "green": "left",
    "yellow": "right",
}
EDGE_COLOR_CLASSES = ["Red", "Blue", "Green", "Yellow"]


# ============================================================
# output_dir 추출
# ============================================================
def _get_output_dir():
    output_path = "./logs"
    model_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
        if arg == "--model_args" and i + 1 < len(sys.argv):
            match = re.search(r"pretrained=([^,]+)", sys.argv[i + 1])
            if match:
                model_name = match.group(1).replace("/", "__")
    if model_name:
        return os.path.join(output_path, model_name)
    return output_path


# ============================================================
# doc_to_visual
# ============================================================
def doc_to_visual(doc):
    video_rel = doc["video"]
    video_path = os.path.join(HF_DATASETS_CACHE, "E2E_real_VP", video_rel)
    if not os.path.exists(video_path):
        eval_logger.error(f"Video path: {video_path} does not exist")
    return [video_path]


# ============================================================
# doc_to_text
# ============================================================
def close_ended_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    candidates = doc["candidates"]
    prompt = f"{question}\n"
    for i, opt in enumerate(candidates):
        prompt += f"{chr(ord('A') + i)}. {opt}\n"
    prompt += "Answer with the option letter only."
    return prompt


# ============================================================
# 공통 헬퍼: pred_letter 추출
# ============================================================
def _extract_pred_letter(pred_raw, n_candidates):
    valid_letters = [chr(ord("A") + i) for i in range(n_candidates)]
    pattern = "[" + "".join(valid_letters) + "]"
    match = re.search(pattern, pred_raw.upper())
    return match.group(0) if match else "NONE"


def _pred_letter_to_text(pred_letter, candidates):
    """pred_letter(A/B/C/D) → 해당 candidate 텍스트. 유효하지 않으면 None."""
    if not pred_letter or len(pred_letter) != 1:
        return None
    idx = ord(pred_letter) - ord("A")
    if 0 <= idx < len(candidates):
        return candidates[idx]
    return None

# ============================================================
# process_results: E2E_VP_default (4지선다, 방향)
# ============================================================
def default_process_results(doc, results):
    pred_raw = results[0].strip()
    gold = doc["answer"]
    candidates = doc["candidates"]
    n = len(candidates)

    pred_letter = _extract_pred_letter(pred_raw, n)
    pred_text = _pred_letter_to_text(pred_letter, candidates)

    pred_direction = pred_text.lower() if pred_text else "none"
    gold_direction = doc["direction"]

    return {
        "accuracy": {
            "correct": 1.0 if pred_letter == gold else 0.0,
            "pred": pred_letter,
            "gold": gold,
            "pred_text": pred_text,
            "gold_text": doc["answer_text"],
            "pred_direction": pred_direction,
            "gold_direction": gold_direction,
            "pred_raw": pred_raw,
            "video": doc["video"],
            "direction": doc["direction"],
        }
    }

# ============================================================
# process_results: E2E_Colored_area (2지선다)
# ============================================================

# 색상 → 방향 역매핑 (축별)
AREA_COLOR_TO_DIRECTION = {
    'vertical': {'Yellow': 'up', 'Black': 'down'},      # up/down 비디오
    'horizontal': {'Yellow': 'left', 'Black': 'right'},  # left/right 비디오
}

def colored_area_process_results(doc, results):
    pred_raw = results[0].strip()
    gold = doc["answer"]
    candidates = doc["candidates"]
    n = len(candidates)

    pred_letter = _extract_pred_letter(pred_raw, n)
    pred_text = _pred_letter_to_text(pred_letter, candidates)

    gold_direction = doc["direction"]
    # 축 판별: up/down → vertical, left/right → horizontal
    axis = 'vertical' if gold_direction in ('up', 'down') else 'horizontal'
    pred_direction = AREA_COLOR_TO_DIRECTION[axis].get(pred_text, 'none') if pred_text else 'none'

    return {
        "accuracy": {
            "correct": 1.0 if pred_letter == gold else 0.0,
            "pred": pred_letter,
            "gold": gold,
            "pred_text": pred_text,
            "gold_text": doc["answer_text"],
            "pred_direction": pred_direction,
            "gold_direction": gold_direction,
            "pred_raw": pred_raw,
            "video": doc["video"],
            "direction": doc["direction"],
        }
    }


# ============================================================
# process_results: E2E_Colored_edge (4지선다, 색상)
# ============================================================
def colored_edge_process_results(doc, results):
    pred_raw = results[0].strip()
    gold = doc["answer"]
    candidates = doc["candidates"]
    n = len(candidates)

    pred_letter = _extract_pred_letter(pred_raw, n)
    pred_text = _pred_letter_to_text(pred_letter, candidates)

    # 예측 색상 → 방향 변환
    pred_direction = "none"
    if pred_text:
        pred_direction = EDGE_COLOR_TO_DIRECTION.get(pred_text.lower(), "none")

    gold_direction = doc["direction"]

    return {
        "accuracy": {
            "correct": 1.0 if pred_letter == gold else 0.0,
            "pred": pred_letter,
            "gold": gold,
            "pred_text": pred_text,
            "gold_text": doc["answer_text"],
            "pred_direction": pred_direction,
            "gold_direction": gold_direction,
            "pred_raw": pred_raw,
            "video": doc["video"],
            "direction": doc["direction"],
        }
    }


# ============================================================
# process_results: E2E_text (4지선다, 방향 텍스트)
# ============================================================
def text_process_results(doc, results):
    pred_raw = results[0].strip()
    gold = doc["answer"]
    candidates = doc["candidates"]
    n = len(candidates)

    pred_letter = _extract_pred_letter(pred_raw, n)
    pred_text = _pred_letter_to_text(pred_letter, candidates)

    # 예측 텍스트가 곧 방향
    pred_direction = pred_text.lower() if pred_text else "none"
    gold_direction = doc["direction"]

    return {
        "accuracy": {
            "correct": 1.0 if pred_letter == gold else 0.0,
            "pred": pred_letter,
            "gold": gold,
            "pred_text": pred_text,
            "gold_text": doc["answer_text"],
            "pred_direction": pred_direction,
            "gold_direction": gold_direction,
            "pred_raw": pred_raw,
            "video": doc["video"],
            "direction": doc["direction"],
        }
    }


# ============================================================
# Confusion Matrix 저장 (방향 기반)
# ============================================================
def _save_direction_confusion_matrix(results_list, task_name):
    """pred_direction vs gold_direction 기반 4×4 confusion matrix."""
    classes = DIRECTION_CLASSES
    labels = DIRECTION_LABELS
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for r in results_list:
        gi = class_to_idx.get(r["gold_direction"], -1)
        pi = class_to_idx.get(r.get("pred_direction", "none"), -1)
        if gi >= 0 and pi >= 0:
            cm[gi][pi] += 1

    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0.0

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = _get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{task_name}_confusion_matrix_{ts}.png")

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, n * 1.0)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="Ground Truth",
        title=f"{task_name}\nAccuracy: {acc:.2%} ({int(correct)}/{int(total)})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n):
        row_total = cm[i].sum()
        for j in range(n):
            pct = cm[i][j] / row_total * 100 if row_total > 0 else 0
            color = "white" if cm[i][j] > thresh else "black"
            ax.text(j, i, f"{cm[i][j]}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    eval_logger.info(f"{task_name} | Accuracy: {acc:.4f} ({int(correct)}/{int(total)}) | Saved: {save_path}")
    return acc


# ============================================================
# Per-direction accuracy 로깅
# ============================================================
def _log_per_direction_accuracy(results_list, task_name):
    dir_correct = {}
    dir_total = {}
    for r in results_list:
        d = r["direction"]
        dir_total[d] = dir_total.get(d, 0) + 1
        dir_correct[d] = dir_correct.get(d, 0) + r["correct"]

    total = len(results_list)
    correct = sum(r["correct"] for r in results_list)
    acc = correct / total if total > 0 else 0.0

    eval_logger.info(f"{'=' * 50}")
    eval_logger.info(f"{task_name} | Overall: {acc:.4f} ({int(correct)}/{total})")
    for d in ["up", "down", "left", "right"]:
        if d in dir_total:
            d_acc = dir_correct[d] / dir_total[d]
            eval_logger.info(f"  {d:>5s}: {d_acc:.4f} ({int(dir_correct[d])}/{dir_total[d]})")
    eval_logger.info(f"{'=' * 50}")

    return acc


# ============================================================
# Aggregation: E2E_Colored_area
# ============================================================
def colored_area_aggregate(results):
    _log_per_direction_accuracy(results, "e2e_vp_colored_area")
    return _log_per_direction_accuracy(results, "e2e_vp_colored_area")


# ============================================================
# Aggregation: E2E_Colored_edge (방향 confusion matrix)
# ============================================================
def colored_edge_aggregate(results):
    _log_per_direction_accuracy(results, "e2e_vp_colored_edge")
    return _log_per_direction_accuracy(results, "e2e_vp_colored_edge")


# ============================================================
# Aggregation: E2E_text (방향 confusion matrix)
# ============================================================
def text_aggregate(results):
    _log_per_direction_accuracy(results, "e2e_vp_text")
    return _log_per_direction_accuracy(results, "e2e_vp_text")

# ============================================================
# Aggregation: E2E_VP_default
# ============================================================ㅋ
def default_aggregate(results):
    _log_per_direction_accuracy(results, "e2e_vp_default")
    return _log_per_direction_accuracy(results, "e2e_vp_default")