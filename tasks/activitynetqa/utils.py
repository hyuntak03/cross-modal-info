# tasks/activitynetqa/utils.py
# ActivityNet-QA 데이터셋 변환 함수

import os


def doc_to_visual(doc, task_kwargs=None, video_folder="", image_folder=""):
    """비디오 경로: v_ + video_name + .mp4"""
    video_name = doc.get("video_name", "")
    prefix = "v_"
    ext = ".mp4"
    if task_kwargs and isinstance(task_kwargs, dict):
        prefix = task_kwargs.get("video_prefix", "v_")
        ext = task_kwargs.get("video_ext", ".mp4")

    filename = f"{prefix}{video_name}{ext}" if video_name else ""

    if video_folder and filename:
        full_path = os.path.join(video_folder, filename)
        if os.path.exists(full_path):
            return [full_path]

    return [filename]


def doc_to_text(doc, task_kwargs=None):
    """question 그대로"""
    return doc.get("question", "")
