# tasks/mvbench/utils.py
# MVBench 데이터셋 변환 함수
# answer는 항상 option letter (A, B, C, D, E) 로 반환
# → Attention Knockout에서 단일 토큰 추적 가능

import os


def doc_to_visual(doc, task_kwargs=None, video_folder="", image_folder=""):
    """비디오 경로 반환"""
    video_rel = doc.get("video", "")
    if video_folder and video_rel:
        full_path = os.path.join(video_folder, video_rel)
        if os.path.exists(full_path):
            return [full_path]
    return [video_rel]


def _answer_text_to_letter(doc):
    """정답 텍스트 → 옵션 letter (A, B, C, ...) 변환"""
    answer_text = str(doc.get("answer", "")).strip()
    candidates = doc.get("candidates", [])

    for i, cand in enumerate(candidates):
        if str(cand).strip() == answer_text:
            return chr(65 + i)  # A, B, C, ...

    # 이미 letter면 그대로
    if len(answer_text) == 1 and answer_text.upper() in "ABCDEFGH":
        return answer_text.upper()

    return answer_text


def doc_to_text(doc, task_kwargs=None):
    """
    question + options 포맷팅.
    post_prompt는 넣지 않음 (CustomDataset이 suffix 추가하므로).
    """
    question = doc.get("question", "")
    candidates = doc.get("candidates", [])

    if candidates:
        opt_lines = []
        for i, opt in enumerate(candidates):
            opt_lines.append(f"({chr(65 + i)}) {opt}")
        option_str = "\n".join(opt_lines)
        question = f"{question}\n{option_str}"

    return question


def doc_to_target(doc, task_kwargs=None):
    """정답 option letter 반환 (A, B, C, D, ...)"""
    return _answer_text_to_letter(doc)


def doc_to_false_option(doc, task_kwargs=None):
    """오답 option letter들을 | 로 연결 (e.g., "B | C | D")"""
    correct_letter = _answer_text_to_letter(doc)
    candidates = doc.get("candidates", [])
    false_letters = []
    for i in range(len(candidates)):
        letter = chr(65 + i)
        if letter != correct_letter:
            false_letters.append(letter)
    return " | ".join(false_letters)
