# dataset_loader.py
# YAML 기반 자동 task 발견 + HuggingFace/CSV 통합 데이터셋 로더
# lmms_eval 패턴 참고: tasks/ 폴더에 YAML 넣으면 자동 등록

import os
import inspect
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
_GROUP_REGISTRY = {}   # group 이름 → 하위 task 이름 리스트
_DEFAULT_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")


def discover_tasks(task_dir: str = None) -> dict:
    """
    task_dir 하위를 재귀 탐색하여 YAML task 설정을 자동 발견.
    - task: "이름" (문자열) → 개별 task 등록
    - task: [리스트]  + group: "이름" → 그룹 등록 (lmms_eval 패턴)
    """
    global _TASK_REGISTRY, _GROUP_REGISTRY

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

            task_val = config.get("task")
            if task_val is None:
                continue

            # 개별 task: task가 문자열
            if isinstance(task_val, str):
                _TASK_REGISTRY[task_val] = yaml_path
            # 그룹: task가 리스트
            elif isinstance(task_val, list):
                group_name = config.get("group")
                if group_name and isinstance(group_name, str):
                    _GROUP_REGISTRY[group_name] = task_val

    return _TASK_REGISTRY


def _expand_group(group_name: str) -> List[str]:
    """그룹을 재귀적으로 펼쳐서 개별 task 이름 리스트로 반환"""
    tasks = []
    for name in _GROUP_REGISTRY.get(group_name, []):
        if name in _GROUP_REGISTRY:
            tasks.extend(_expand_group(name))
        elif name in _TASK_REGISTRY:
            tasks.append(name)
        else:
            print(f"[WARN] 그룹 '{group_name}'의 하위 task '{name}'이 등록되지 않음 (건너뜀)")
    return tasks


def list_tasks(include_groups: bool = True) -> List[str]:
    """등록된 task (+ 그룹) 이름 목록 반환"""
    if not _TASK_REGISTRY:
        discover_tasks()
    names = set(_TASK_REGISTRY.keys())
    if include_groups:
        names |= set(_GROUP_REGISTRY.keys())
    return sorted(names)


def get_task_config(task_name: str) -> dict:
    """task 이름으로 완전한 config (함수 포함) 로드"""
    if not _TASK_REGISTRY:
        discover_tasks()

    if task_name in _TASK_REGISTRY:
        yaml_path = _TASK_REGISTRY[task_name]
        return load_yaml_config(yaml_path, mode="full")

    if task_name in _GROUP_REGISTRY:
        raise ValueError(
            f"'{task_name}'은 그룹입니다. "
            f"expand_group('{task_name}')으로 하위 task 목록을 얻거나, "
            f"개별 task 이름을 사용하세요.\n"
            f"하위 tasks: {_expand_group(task_name)}"
        )

    raise ValueError(
        f"Unknown task: '{task_name}'. "
        f"Available: {list_tasks()}"
    )


def expand_group(task_name: str) -> List[str]:
    """
    task 이름이 그룹이면 하위 task 리스트 반환,
    개별 task면 [task_name] 반환.
    그룹 안의 그룹도 재귀적으로 펼침.
    """
    if not _TASK_REGISTRY:
        discover_tasks()

    if task_name in _GROUP_REGISTRY:
        return _expand_group(task_name)
    elif task_name in _TASK_REGISTRY:
        return [task_name]
    else:
        raise ValueError(
            f"Unknown task or group: '{task_name}'. "
            f"Available: {list_tasks()}"
        )


# ============================================================
#  통합 로더: HF/CSV → questions 리스트 변환
# ============================================================

def load_dataset_as_questions(
    task_name: str = None,
    csv_path: str = None,
    video_folder: str = "",
    image_folder: str = "",
    hf_cache_dir: str = None,
    limit: int = -1,
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

        if limit > 0:
            questions = questions[:limit]
            dataset_dict = {q["q_id"]: q for q in questions}

        return questions, dataset_dict

    # ---- HuggingFace 로딩 ----
    if task_name is None:
        raise ValueError("task_name 또는 csv_path 중 하나는 필수")

    # 그룹이면 하위 task들 합쳐서 반환
    if task_name in _GROUP_REGISTRY:
        sub_tasks = expand_group(task_name)
        print(f"[dataset_loader] 그룹 '{task_name}' → {len(sub_tasks)}개 하위 task 로딩")
        all_questions = []
        for st in sub_tasks:
            qs, _ = load_dataset_as_questions(
                task_name=st,
                video_folder=video_folder,
                image_folder=image_folder,
                hf_cache_dir=hf_cache_dir,
                limit=limit,
                split_override=split_override,
            )
            # task 출처 표시
            for q in qs:
                q["source_task"] = st
            all_questions.extend(qs)
        all_dict = {q["q_id"]: q for q in all_questions}
        print(f"[dataset_loader] 그룹 '{task_name}' 총 {len(all_questions)}개 샘플 로딩 완료")
        return all_questions, all_dict

    config = get_task_config(task_name)

    hf_path = config["dataset_path"]
    hf_name = config.get("dataset_name", None)
    hf_split = split_override or config.get("test_split", "test")
    hf_kwargs = dict(config.get("dataset_kwargs", {}))

    # lmms_eval 전용 커스텀 키 분리 (datasets.load_dataset에 넘기면 에러)
    _CUSTOM_KEYS = {"video", "force_download", "force_unzip", "create_link",
                    "builder_script", "From_YouTube", "load_from_disk"}
    is_video_dataset = hf_kwargs.pop("video", False)
    for ck in _CUSTOM_KEYS - {"video"}:
        hf_kwargs.pop(ck, None)

    # 빈 문자열 cache_dir 제거 (lmms_eval YAML에서 cache_dir: "" 로 오는 경우)
    if hf_kwargs.get("cache_dir") == "":
        hf_kwargs.pop("cache_dir")

    # 캐시 디렉토리 우선순위: 인자 > HF_DATASETS_CACHE 환경변수 > YAML의 cache_dir
    if hf_cache_dir:
        hf_kwargs["cache_dir"] = hf_cache_dir
    elif os.environ.get("HF_DATASETS_CACHE"):
        hf_kwargs["cache_dir"] = os.environ["HF_DATASETS_CACHE"]

    # token=True이면 환경변수 HF_TOKEN 또는 huggingface-cli 로그인 토큰 사용
    if hf_kwargs.get("token") is True:
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            hf_kwargs["token"] = env_token

    # video 데이터셋이면 snapshot_download로 비디오 파일 미리 확보
    # if is_video_dataset:
    #     from huggingface_hub import snapshot_download
    #     cache_dir = hf_kwargs.get("cache_dir", None)
    #     revision = hf_kwargs.get("revision", None)
    #     token = hf_kwargs.get("token", None)
    #     try:
    #         snapshot_download(
    #             repo_id=hf_path, repo_type="dataset",
    #             cache_dir=cache_dir, revision=revision, token=token,
    #             local_files_only=False,
    #         )
    #     except Exception as e:
    #         print(f"[dataset_loader] snapshot_download 실패 (캐시 사용 시도): {e}")

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
    # task_specific_kwargs 또는 lmms_eval_specific_kwargs 지원
    task_kwargs = config.get("task_specific_kwargs", {})
    lmms_kwargs = config.get("lmms_eval_specific_kwargs", {})
    if lmms_kwargs:
        # lmms_eval_specific_kwargs.default 구조 지원
        if "default" in lmms_kwargs:
            task_kwargs.update(lmms_kwargs["default"])
        else:
            task_kwargs.update(lmms_kwargs)

    questions = []
    for idx, doc in enumerate(ds):
        if limit > 0 and idx >= limit:
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

        # MCQ: answer가 텍스트이고 candidates가 있으면 자동으로 option letter로 변환
        options_field = field_map.get("options", "candidates")
        candidates = doc.get(options_field, [])
        if candidates and len(answer_text) > 1:
            for ci, cand in enumerate(candidates):
                if str(cand).strip() == answer_text.strip():
                    answer_text = chr(65 + ci)  # A, B, C, ...
                    break

        # visual path
        # video_folder 비어있으면 HF_DATASETS_CACHE fallback
        effective_video_folder = video_folder or os.environ.get("HF_DATASETS_CACHE", "")
        if callable(doc_to_visual):
            # 함수 시그니처에 따라 호출 방식 분기
            sig = inspect.signature(doc_to_visual)
            if "video_folder" in sig.parameters:
                vis_result = doc_to_visual(doc, task_kwargs, video_folder=effective_video_folder, image_folder=image_folder)
            elif len(sig.parameters) == 1:
                vis_result = doc_to_visual(doc)
            else:
                vis_result = doc_to_visual(doc, task_kwargs)
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
