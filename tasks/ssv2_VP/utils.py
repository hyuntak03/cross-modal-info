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
EDGE_COLOR_TO_DIRECTION = {
    "red": "up",
    "blue": "down",
    "green": "left",
    "yellow": "right",
}
EDGE_COLOR_CLASSES = ["Red", "Blue", "Green", "Yellow"]


# ============================================================
# Prompt Templates
# ============================================================
PROMPT_TEMPLATES = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Real-world — Plain (no visual prompt)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "minimal": {
        "task": "",
        "instruction": "",
    },

    "plain_light": {
        "task": (
            "[Task]\n"
            "Watch the video carefully and identify the direction "
            "of movement shown in the video.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "First, describe what is moving in the video.\n"
            "Then, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "plain_temporal": {
        "task": (
            "[Task]\n"
            "Watch the video carefully and identify the direction "
            "of movement shown in the video.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "First, describe what is moving in the video.\n"
            "Second, describe where it is at the beginning of the video.\n"
            "Third, describe where it is at the end of the video.\n"
            "Finally, based on the position change, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "plain_full": {
        "task": (
            "[Task]\n"
            "Watch the video carefully and identify the direction "
            "of movement shown in the video.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Describe what is moving in the video.\n"
            "Step 2: Describe where it is at the beginning.\n"
            "Step 3: Describe where it is in the middle.\n"
            "Step 4: Describe where it is at the end.\n"
            "Step 5: Based on the position change, "
            "determine the direction of movement.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Real-world — Colored Edge observe
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "colored_edge_observe_light": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has colored borders on each edge. "
            "Identify which colored border the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "First, describe what colors you see on each border of the video.\n"
            "Then, describe what is moving in the video.\n"
            "Finally, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "colored_edge_observe_temporal": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has colored borders on each edge. "
            "Identify which colored border the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Describe what colors you see on each border of the video.\n"
            "Step 2: Which colored border is the moving object closest to "
            "at the beginning?\n"
            "Step 3: Which colored border is the moving object closest to "
            "at the end?\n"
            "Step 4: Based on this change, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "colored_edge_observe_full": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has colored borders on each edge. "
            "Identify which colored border the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Describe what colors you see on each border of the video.\n"
            "Step 2: Describe what is moving in the video.\n"
            "Step 3: Which colored border is the moving object closest to "
            "at the beginning?\n"
            "Step 4: Which colored border is it closest to in the middle?\n"
            "Step 5: Which colored border is it closest to at the end?\n"
            "Step 6: Based on the trajectory, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Real-world — Colored Edge informed
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "colored_edge_informed_light": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has colored borders on each edge: "
            "Red on top, Blue on bottom, Green on left, Yellow on right. "
            "Identify which colored border the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "First, describe what is moving in the video.\n"
            "Then, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "colored_edge_informed_temporal": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has colored borders on each edge: "
            "Red on top, Blue on bottom, Green on left, Yellow on right. "
            "Identify which colored border the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Which colored border is the moving object closest to "
            "at the beginning?\n"
            "Step 2: Which colored border is it closest to at the end?\n"
            "Step 3: Based on this change, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "colored_edge_informed_full": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has colored borders on each edge: "
            "Red on top, Blue on bottom, Green on left, Yellow on right. "
            "Identify which colored border the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Describe what is moving in the video.\n"
            "Step 2: Which colored border is the moving object closest to "
            "at the beginning?\n"
            "Step 3: Which colored border is it closest to in the middle?\n"
            "Step 4: Which colored border is it closest to at the end?\n"
            "Step 5: Based on the trajectory, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Real-world — Text Edge observe
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "text_edge_observe_light": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has text labels on each edge. "
            "Identify which text label the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "First, read and describe the text labels on each edge of the video.\n"
            "Then, describe what is moving in the video.\n"
            "Finally, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "text_edge_observe_temporal": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has text labels on each edge. "
            "Identify which text label the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Read and describe the text labels on each edge of the video.\n"
            "Step 2: Which text label is the moving object closest to "
            "at the beginning?\n"
            "Step 3: Which text label is it closest to at the end?\n"
            "Step 4: Based on this change, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "text_edge_observe_full": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has text labels on each edge. "
            "Identify which text label the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Read and describe the text labels on each edge of the video.\n"
            "Step 2: Describe what is moving in the video.\n"
            "Step 3: Which text label is the moving object closest to "
            "at the beginning?\n"
            "Step 4: Which text label is it closest to in the middle?\n"
            "Step 5: Which text label is it closest to at the end?\n"
            "Step 6: Based on the trajectory, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Real-world — Text Edge informed
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "text_edge_informed_light": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has directional text labels on each edge: "
            "'Up' on top, 'Down' on bottom, 'Left' on the left, 'Right' on the right. "
            "Identify which text label the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "First, describe what is moving in the video.\n"
            "Then, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "text_edge_informed_temporal": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has directional text labels on each edge: "
            "'Up' on top, 'Down' on bottom, 'Left' on the left, 'Right' on the right. "
            "Identify which text label the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Which text label is the moving object closest to "
            "at the beginning?\n"
            "Step 2: Which text label is it closest to at the end?\n"
            "Step 3: Based on this change, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },

    "text_edge_informed_full": {
        "task": (
            "[Task]\n"
            "Watch the video carefully. "
            "The video has directional text labels on each edge: "
            "'Up' on top, 'Down' on bottom, 'Left' on the left, 'Right' on the right. "
            "Identify which text label the moving object is heading toward.\n"
        ),
        "instruction": (
            "Let's think step by step.\n"
            "Step 1: Describe what is moving in the video.\n"
            "Step 2: Which text label is the moving object closest to "
            "at the beginning?\n"
            "Step 3: Which text label is it closest to in the middle?\n"
            "Step 4: Which text label is it closest to at the end?\n"
            "Step 5: Based on the trajectory, select the correct answer.\n"
            "End your response with: Therefore, the answer is <your choice>.\n"
        ),
    },
}


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
    video_path = os.path.join(HF_DATASETS_CACHE, "ssv2_VP", video_rel)
    if not os.path.exists(video_path):
        eval_logger.error(f"Video path: {video_path} does not exist")
    return [video_path]


# ============================================================
# doc_to_text
# ============================================================
def close_ended_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    candidates = doc["candidates"]

    # 프롬프트 스타일 결정
    # 프롬프트 스타일 결정 (환경변수 > kwargs > default)
    prompt_style = os.environ.get("PROMPT_STYLE", None)
    if prompt_style is None and lmms_eval_specific_kwargs:
        prompt_style = lmms_eval_specific_kwargs.get("prompt_style", "minimal")
    if prompt_style is None:
        prompt_style = "minimal"

    template = PROMPT_TEMPLATES.get(prompt_style, PROMPT_TEMPLATES["minimal"])

    # 예시 답변 (첫 번째 옵션으로)
    example = f"A. {candidates[0]}" if candidates else "A. Up"

    # 선택지 블록
    options_block = ""
    for i, opt in enumerate(candidates):
        options_block += f"{chr(ord('A') + i)}. {opt}\n"

    # 조립
    prompt = template["task"]
    prompt += template["instruction"].format(example=example)
    prompt += f"[Question]\n{question}\n"
    prompt += options_block
    if prompt_style == "minimal":
        prompt += "\nAnswer with the option letter only.\n"

    return prompt


# ============================================================
# 공통 헬퍼: pred_letter 추출
# ============================================================
def _extract_pred_letter(pred_raw, n_candidates, candidates=None):
    valid_letters = [chr(ord("A") + i) for i in range(n_candidates)]
    letter_set = "".join(valid_letters)
    text = pred_raw.strip()
    text_upper = text.upper()

    # ── 1순위: <answer> 태그 (CoT 프롬프트용) ──
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        tag_content = answer_match.group(1).upper()
        tag_match = re.search(r"[" + letter_set + r"]", tag_content)
        if tag_match:
            return tag_match.group(0)

    # ── 2순위: "X." 패턴 (가장 흔한 모델 출력) ── last match
    dot_matches = list(re.finditer(r"([" + letter_set + r"])\.", text_upper))
    if dot_matches:
        return dot_matches[-1].group(1)

    # ── 3순위: 다양한 자연어 패턴 ──
    nl_patterns = [
        r"(?:^|\n)(?:ANSWER:?\s*)?([" + letter_set + r"])(?:\.|$|\s)",  # 줄 시작 Answer: A
        r"[\*\"\']([" + letter_set + r"])[\*\"\']",                      # *A*, "A", 'A'
        r"\bANSWER:?\s*([" + letter_set + r"])\b",                      # Answer: A (중간)
        r"(?:MY\s+)?ANSWER\s+IS\s+([" + letter_set + r"])",             # (My) answer is A
        r"(?:THE\s+)?ANSWER\s*(?:IS|:)\s*([" + letter_set + r"])",      # The answer is/: A
        r"(?:I\s+(?:THINK|BELIEVE|CHOOSE)\s+)([" + letter_set + r"])",  # I think/choose A
        r"(?:OPTION\s+)([" + letter_set + r"])\b",                      # Option A
        r"(?:CORRECT\s+ANSWER\s*(?:IS|:)\s*)([" + letter_set + r"])",   # Correct answer is A
    ]
    for pattern in nl_patterns:
        matches = list(re.finditer(pattern, text_upper))
        if matches:
            return matches[-1].group(1)

    # ── 4순위: line-by-line 역방향 스캔 ──
    lines = text_upper.split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"\b([" + letter_set + r"])\b", line)
        if m:
            return m.group(1)
        
    # ── 5순위: candidate 텍스트 직접 매칭 ──
    # 모델이 letter 없이 "Right</s>" 등 candidate 텍스트만 출력한 경우
    if candidates:
        # </s>, EOS 토큰 등 제거 후 비교
        cleaned = re.sub(r"</s>|<\|end\|>|<\|eot_id\|>|\[EOS\]", "", text, flags=re.IGNORECASE).strip()
        cleaned_upper = cleaned.upper()
        for i, cand in enumerate(candidates):
            if cand.upper() == cleaned_upper:
                return chr(ord("A") + i)
        # 부분 매칭: candidate가 응답에 포함되어 있는 경우 (긴 응답 대비)
        for i, cand in enumerate(candidates):
            if cand.upper() in cleaned_upper:
                # 여러 개 매칭되면 마지막 등장 기준
                pass  # 아래에서 가장 마지막에 등장하는 것 선택
        last_pos = -1
        last_idx = -1
        for i, cand in enumerate(candidates):
            pos = cleaned_upper.rfind(cand.upper())
            if pos > last_pos:
                last_pos = pos
                last_idx = i
        if last_idx >= 0:
            return chr(ord("A") + last_idx)

    return "NONE"

#  "we use rule-based extraction with multiple fallback patterns; responses not matching any pattern are scored as incorrect"

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

    pred_letter = _extract_pred_letter(pred_raw, n, candidates)
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

    pred_letter = _extract_pred_letter(pred_raw, n, candidates)
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

    pred_letter = _extract_pred_letter(pred_raw, n, candidates)
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

    pred_letter = _extract_pred_letter(pred_raw, n, candidates)
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
    """pred_direction vs gold_direction 기반 confusion matrix (데이터 기반 동적 클래스)."""
    # 데이터에 실제 존재하는 gold_direction만 추출
    present_directions = sorted(
        {r["gold_direction"] for r in results_list if r["gold_direction"] in DIRECTION_CLASSES},
        key=lambda d: DIRECTION_CLASSES.index(d),
    )
    classes = present_directions if present_directions else DIRECTION_CLASSES
    label_map = dict(zip(DIRECTION_CLASSES, DIRECTION_LABELS))
    labels = [label_map[c] for c in classes]
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

# "none" 등 유효하지 않은 pred_direction도 집계에 포함
    # confusion matrix에는 빠지되, accuracy 분모에는 포함
    total_count = 0
    correct_count = 0
    for r in results_list:
        gi = class_to_idx.get(r["gold_direction"], -1)
        pi = class_to_idx.get(r.get("pred_direction", "none"), -1)
        if gi >= 0:
            total_count += 1
            if pi >= 0:
                cm[gi][pi] += 1
                if gi == pi:
                    correct_count += 1

    total = total_count
    correct = correct_count
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
# Aggregation: E2E_Colored_edge (방향 confusion matrix)
# ============================================================
def colored_edge_aggregate(results):
    _save_direction_confusion_matrix(results, "ssv2_vp_colored_edge")
    return _log_per_direction_accuracy(results, "ssv2_vp_colored_edge")


# ============================================================
# Aggregation: E2E_text (방향 confusion matrix)
# ============================================================
def text_aggregate(results):
    _save_direction_confusion_matrix(results, "ssv2_vp_text")
    return _log_per_direction_accuracy(results, "ssv2_vp_text")

# ============================================================
# Aggregation: E2E_VP_default
# ============================================================
def default_aggregate(results):
    _save_direction_confusion_matrix(results, "ssv2_vp_default")
    return _log_per_direction_accuracy(results, "ssv2_vp_default")