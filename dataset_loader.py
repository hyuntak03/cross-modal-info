# dataset_loader.py
# YAML 기반 자동 task 발견 + HuggingFace/CSV 통합 데이터셋 로더
# lmms_eval 패턴 참고: tasks/ 폴더에 YAML 넣으면 자동 등록

import os
import importlib.util
import collections
from typing import List, Optional

import yaml
import datasets
import pandas as pd


# ============================================================
#  YAML 로더: !function 태그 지원
# ============================================================

class _FunctionLoader(yaml.SafeLoader):
    """!function utils.my_func 태그를 Python 함수로 변환하는 YAML 로더"""
    pass


def _function_constructor(loader, node):
    """!function 태그 처리: 같은 디렉토리의 utils.py에서 함수 import"""
    func_string = loader.construct_scalar(node)
    yaml_dir = os.path.dirname(loader.name) if hasattr(loader, 'name') else "."

    parts = func_string.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"!function 형식 오류: '{func_string}' (module.function 형태여야 함)")
    module_name, func_name = parts

    # 1) YAML 파일과 같은 디렉토리에서 상대 import
    module_path = os.path.join(yaml_dir, f"{module_name}.py")
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(
            f"tasks.{module_name}_{id(loader)}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    # 2) 절대 import 폴백
    try:
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    except Exception as ex:
        raise ImportError(
            f"!function '{func_string}' 로딩 실패. "
            f"상대 경로: {module_path}, 절대 import: {module_name}"
        ) from ex


class _SimpleLoader(yaml.SafeLoader):
    """!function을 문자열로 그대로 유지하는 YAML 로더 (발견 단계용)"""
    pass


def _function_passthrough(loader, node):
    return loader.construct_scalar(node)


_FunctionLoader.add_constructor('!function', _function_constructor)
_SimpleLoader.add_constructor('!function', _function_passthrough)


def load_yaml_config(yaml_path, mode="full"):
    """
    YAML 파일 로드. include 상속 지원.

    Args:
        mode: "full" → !function을 실제 함수로 변환
              "simple" → !function을 문자열로 유지
    """
    loader_cls = _FunctionLoader if mode == "full" else _SimpleLoader

    with open(yaml_path, 'r') as f:
        # loader.name에 파일 경로 저장 (!function 상대 경로 해결용)
        loader = loader_cls(f)
        loader.name = yaml_path
        try:
            config = loader.get_single_data()
        finally:
            loader.dispose()

    if config is None:
        config = {}

    yaml_dir = os.path.dirname(yaml_path)

    # include 처리 (템플릿 상속)
    if "include" in config:
        include_paths = config.pop("include")
        if isinstance(include_paths, str):
            include_paths = [include_paths]

        base_config = {}
        for inc_path in include_paths:
            if not os.path.isabs(inc_path):
                inc_path = os.path.join(yaml_dir, inc_path)
            parent = load_yaml_config(inc_path, mode=mode)
            base_config.update(parent)

        # child가 parent를 override
        base_config.update(config)
        return base_config

    return config


# ============================================================
#  Task 자동 발견 (tasks/ 폴더 스캔)
# ============================================================

_TASK_REGISTRY = {}
_DEFAULT_TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")


def discover_tasks(task_dir: str = None) -> dict:
    """
    task_dir 하위를 재귀 탐색하여 YAML task 설정을 자동 발견.
    task: "이름" (문자열)이 있는 YAML만 등록.
    """
    global _TASK_REGISTRY

    if task_dir is None:
        task_dir = _DEFAULT_TASKS_DIR

    if not os.path.isdir(task_dir):
        print(f"[dataset_loader] Task directory not found: {task_dir}")
        return _TASK_REGISTRY

    ignore_dirs = {"__pycache__", ".ipynb_checkpoints", ".git"}

    for root, dirs, files in os.walk(task_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for f in files:
            if not f.endswith(".yaml"):
                continue
            # _default_template 은 단독 task가 아님
            if f.startswith("_"):
                continue

            yaml_path = os.path.join(root, f)
            try:
                config = load_yaml_config(yaml_path, mode="simple")
            except Exception as e:
                print(f"[WARN] YAML 로드 실패: {yaml_path}: {e}")
                continue

            if not config:
                continue

            # task (문자열)이 있으면 등록
            if "task" in config and isinstance(config["task"], str):
                _TASK_REGISTRY[config["task"]] = yaml_path

    return _TASK_REGISTRY


def list_tasks() -> List[str]:
    """등록된 task 이름 목록 반환"""
    if not _TASK_REGISTRY:
        discover_tasks()
    return sorted(_TASK_REGISTRY.keys())


def get_task_config(task_name: str) -> dict:
    """task 이름으로 완전한 config (함수 포함) 로드"""
    if not _TASK_REGISTRY:
        discover_tasks()

    if task_name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: '{task_name}'. "
            f"Available: {list_tasks()}"
        )

    yaml_path = _TASK_REGISTRY[task_name]
    return load_yaml_config(yaml_path, mode="full")


# ============================================================
#  통합 로더: HF/CSV → questions 리스트 변환
# ============================================================

def load_dataset_as_questions(
    task_name: str = None,
    csv_path: str = None,
    video_folder: str = "",
    image_folder: str = "",
    hf_cache_dir: str = None,
    max_samples: int = -1,
    split_override: str = None,
) -> tuple:
    """
    HuggingFace 또는 CSV → 통합 questions 포맷.

    통합 포맷 (각 dict):
        q_id, question, answer, img_id, video, false option, ...

    Returns:
        (questions: list[dict], dataset_dict: dict)
    """

    # ---- CSV 로딩 (기존 호환) ----
    if csv_path:
        print(f"[dataset_loader] Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path, dtype={"question_id": str}).fillna('')
        dataset_dict = df.set_index('question_id').T.to_dict('dict')
        questions = [{**detail, "q_id": qu_id} for qu_id, detail in dataset_dict.items()]

        if max_samples > 0:
            questions = questions[:max_samples]
            dataset_dict = {q["q_id"]: q for q in questions}

        return questions, dataset_dict

    # ---- HuggingFace 로딩 ----
    if task_name is None:
        raise ValueError("task_name 또는 csv_path 중 하나는 필수")

    config = get_task_config(task_name)

    hf_path = config["dataset_path"]
    hf_name = config.get("dataset_name", None)
    hf_split = split_override or config.get("test_split", "test")
    hf_kwargs = config.get("dataset_kwargs", {})

    # 캐시 디렉토리 우선순위: 인자 > HF_DATASETS_CACHE 환경변수 > YAML의 cache_dir
    if hf_cache_dir:
        hf_kwargs["cache_dir"] = hf_cache_dir
    elif os.environ.get("HF_DATASETS_CACHE"):
        hf_kwargs["cache_dir"] = os.environ["HF_DATASETS_CACHE"]

    print(f"[dataset_loader] Loading from HuggingFace: {hf_path}"
          f"{f' / {hf_name}' if hf_name else ''} (split={hf_split})")

    ds = datasets.load_dataset(
        path=hf_path,
        name=hf_name,
        split=hf_split,
        download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
        **hf_kwargs,
    )

    # YAML의 변환 함수 가져오기
    doc_to_visual = config.get("doc_to_visual")
    doc_to_text = config.get("doc_to_text")
    doc_to_target = config.get("doc_to_target")
    doc_to_false_option = config.get("doc_to_false_option", None)

    field_map = config.get("field_map", {})
    task_kwargs = config.get("task_specific_kwargs", {})

    questions = []
    for idx, doc in enumerate(ds):
        if max_samples > 0 and idx >= max_samples:
            break

        # question ID
        qid_field = field_map.get("question_id", "question_id")
        q_id = str(doc.get(qid_field, idx))
        q_id = f"{q_id}_{idx}"

        # question 텍스트
        if callable(doc_to_text):
            question_text = doc_to_text(doc, task_kwargs)
        else:
            question_text = str(doc.get(doc_to_text or "question", ""))

        # answer
        if callable(doc_to_target):
            answer_text = doc_to_target(doc, task_kwargs)
        elif isinstance(doc_to_target, str):
            answer_text = str(doc.get(doc_to_target, ""))
        else:
            answer_text = str(doc.get("answer", ""))

        # visual path
        if callable(doc_to_visual):
            vis_result = doc_to_visual(doc, task_kwargs, video_folder=video_folder, image_folder=image_folder)
            vis_path = vis_result[0] if isinstance(vis_result, list) else str(vis_result)
        else:
            vid_field = field_map.get("video", "video")
            img_field = field_map.get("image", "img_id")
            if vid_field in doc and doc[vid_field]:
                vis_path = str(doc[vid_field])
            elif img_field in doc and doc[img_field]:
                vis_path = str(doc[img_field])
            else:
                vis_path = ""

        # false option
        false_option = ""
        if callable(doc_to_false_option):
            false_option = doc_to_false_option(doc, task_kwargs)

        # 비디오 vs 이미지 판별
        vid_field = field_map.get("video", "video")
        is_video = vid_field in doc and doc[vid_field]

        q = {
            "q_id": q_id,
            "question": question_text,
            "answer": answer_text,
            "img_id": "" if is_video else vis_path,
            "video": vis_path if is_video else "",
            "false option": false_option,
        }

        # 원본 필드 보존
        for k, v in doc.items():
            if k not in q:
                q[k] = v if not isinstance(v, (list, dict)) else str(v)

        questions.append(q)

    dataset_dict = {q["q_id"]: q for q in questions}
    print(f"[dataset_loader] Loaded {len(questions)} samples from {task_name}")

    return questions, dataset_dict


# ============================================================
#  초기화: import 시 자동 발견
# ============================================================

discover_tasks()


if __name__ == "__main__":
    print("=== Registered Tasks ===")
    for t in list_tasks():
        print(f"  - {t}")
