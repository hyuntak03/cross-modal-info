# tasks/mvbench/utils.py
# MVBench 데이터셋 변환 함수
# lmms_eval의 mvbench/utils.py 로직을 따름
# answer는 항상 option letter (A, B, C, D, E) 로 반환

import os
import string
from pathlib import Path

import yaml

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

# lmms_eval 패턴: HF_DATASETS_CACHE + YAML cache_dir로 base_cache_dir 구성
hf_home = os.getenv("HF_DATASETS_CACHE", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("cache_dir", "")


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


def doc_to_visual(doc, task_kwargs=None, video_folder="", image_folder=""):
    """비디오 경로 반환. lmms_eval 패턴: base_cache_dir/cache_name/DATA_LIST[sub_task]/video"""
    sub_task = (task_kwargs or {}).get("sub_task", "")
    dataset_folder = DATA_LIST.get(sub_task, "")

    cache_dir = os.path.join(base_cache_dir, cache_name) if cache_name else base_cache_dir
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])

    if os.path.exists(video_path):
        return [video_path]

    # alternative path fallback (clevrer, star)
    if os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            return [alternative_video_path]

    # video_folder 인자 fallback
    if video_folder:
        full_path = os.path.join(video_folder, dataset_folder, doc["video"])
        if os.path.exists(full_path):
            return [full_path]
        full_path = os.path.join(video_folder, doc["video"])
        if os.path.exists(full_path):
            return [full_path]

    print(f"[WARN] Video path not found: {video_path}")
    return [video_path]


def doc_to_text(doc, task_kwargs=None):
    """question + options 포맷팅. lmms_eval 패턴 따름."""
    question = doc.get("question", "")
    candidates = doc.get("candidates", [])
    option_letters = string.ascii_uppercase
    post_prompt = (task_kwargs or {}).get("post_prompt", "")

    option_prompt = ""
    for char_index, option in enumerate(candidates):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"

    full_text = "Question:" + question + "\nOption:\n" + option_prompt + post_prompt
    return full_text


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
