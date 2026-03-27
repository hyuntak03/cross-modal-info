# tasks/videomme/utils.py
# Video-MME 데이터셋 변환 함수
# answer는 항상 option letter (A, B, C, D) 로 반환

import os


def doc_to_visual(doc, task_kwargs=None, video_folder="", image_folder=""):
    """비디오 경로: videoID + .mp4"""
    video_id = doc.get("videoID", "")
    ext = ".mp4"
    if task_kwargs and isinstance(task_kwargs, dict):
        ext = task_kwargs.get("video_ext", ".mp4")

    filename = f"{video_id}{ext}" if video_id else ""

    if video_folder and filename:
        for subdir in ["data", ""]:
            full_path = os.path.join(video_folder, subdir, filename) if subdir else os.path.join(video_folder, filename)
            if os.path.exists(full_path):
                return [full_path]

    return [filename]


def _answer_to_letter(doc):
    """정답 → option letter 변환"""
    answer = str(doc.get("answer", "")).strip()
    options = doc.get("options", [])

    # 이미 letter면 그대로
    if len(answer) == 1 and answer.upper() in "ABCDEFGH":
        return answer.upper()

    # 옵션 텍스트와 매칭
    for i, opt in enumerate(options):
        if str(opt).strip() == answer:
            return chr(65 + i)

    return answer


def doc_to_text(doc, task_kwargs=None):
    """question + options 포맷팅 (post_prompt 없음)"""
    question = doc.get("question", "")
    options = doc.get("options", [])

    if options:
        opt_lines = []
        for i, opt in enumerate(options):
            opt_lines.append(f"({chr(65 + i)}) {opt}")
        question = f"{question}\n" + "\n".join(opt_lines)

    return question


def doc_to_target(doc, task_kwargs=None):
    """정답 option letter 반환"""
    return _answer_to_letter(doc)


def doc_to_false_option(doc, task_kwargs=None):
    """오답 option letter들을 | 로 연결"""
    correct_letter = _answer_to_letter(doc)
    options = doc.get("options", [])
    false_letters = []
    for i in range(len(options)):
        letter = chr(65 + i)
        if letter != correct_letter:
            false_letters.append(letter)
    return " | ".join(false_letters)
