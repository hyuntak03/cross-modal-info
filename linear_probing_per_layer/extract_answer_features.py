"""
Layer별 Answer Token Hidden States 추출 스크립트.

기존 extract_vision_features.py가 vision token 위치의 hidden states를 추출했다면,
이 스크립트는 answer token 위치 (prefill의 마지막 토큰 = 다음 토큰 생성 위치)의
hidden states를 추출한다.

Feature shape: (num_samples, hidden_dim) — 단일 토큰이므로 vision 대비 훨씬 작음.

Vision probing과 비교하여:
  Case A: Answer token도 80%+ → LLM 내부에 정보 있음, 디코딩 문제
  Case B: Answer token 낮음 → Vision→Answer 라우팅 실패, bottleneck layer 특정 가능
  Case C: 중간에서 올라갔다 후반에 하락 → MLP가 방향 정보를 덮어씀

지원 모델:
  - LLaVA 계열 (LLaVA-OneVision, LLaVA-Video 등)
  - Qwen3-VL 계열 (Qwen3-VL-4B-Instruct 등)

Usage (LLaVA):
    python linear_probing_per_layer/extract_answer_features.py \
        --model_args "pretrained=...,conv_template=qwen_1_5,device_map=auto" \
        --task direction_testbed_ablation_8way \
        --output_dir output/MODEL_NAME/answer_probe_features

Usage (Qwen3-VL):
    python linear_probing_per_layer/extract_answer_features.py \
        --model_type qwen3_vl \
        --model_args "pretrained=/path/to/Qwen3-VL-4B-Instruct" \
        --task direction_testbed_ablation_8way \
        --output_dir output/MODEL_NAME/answer_probe_features
"""

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import ast
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
import math
import string

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

torch.set_grad_enabled(False)

# TF32 활성화 (A100/H100 bf16 유사 속도)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# core/__init__.py의 무거운 import chain을 피하기 위해 dataset_loader를 직접 로드
def _import_module_direct(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dataset_loader = _import_module_direct(
    "core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py")
)
load_dataset_as_questions = _dataset_loader.load_dataset_as_questions


# ============================================================
#  모델 타입 자동 감지
# ============================================================

def detect_model_type(pretrained_path):
    """config.json의 model_type으로 자동 감지."""
    config_path = os.path.join(pretrained_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
        if "qwen3_vl" in model_type:
            return "qwen3_vl"
    return "llava"


def parse_model_args(args_string):
    """lmms_eval 스타일 model_args 파싱."""
    if not args_string:
        return {}
    result = {}
    for item in args_string.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() == "none":
            val = None
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        result[key.strip()] = val
    return result


# ============================================================
#  공통: 라벨 셋 구성
# ============================================================

def build_label_set(questions):
    """전체 샘플의 GT answer에서 unique label set 구성."""
    unique_answers = set()
    for q in questions:
        ans = str(q["answer"]).strip()
        if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
            candidates_raw = q.get("candidates", [])
            if isinstance(candidates_raw, str):
                candidates_raw = ast.literal_eval(candidates_raw)
            idx = ord(ans.upper()) - ord('A')
            if idx < len(candidates_raw):
                ans = str(candidates_raw[idx]).strip()
        unique_answers.add(ans)

    label_list = sorted(unique_answers)
    answer_to_idx = {a: i for i, a in enumerate(label_list)}
    return label_list, answer_to_idx


def resolve_answer(line):
    """질문의 answer를 텍스트로 변환."""
    answer = str(line["answer"]).strip()
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        candidates_raw = line.get("candidates", [])
        if isinstance(candidates_raw, str):
            candidates_raw = ast.literal_eval(candidates_raw)
        idx = ord(answer.upper()) - ord('A')
        if idx < len(candidates_raw):
            answer = str(candidates_raw[idx]).strip()
    return answer


# ============================================================
#  LLaVA 계열 추출
# ============================================================

def extract_features_llava(args):
    """LLaVA 계열 모델에서 answer token 위치의 hidden states 추출."""
    _model_loader = _import_module_direct(
        "core.model_loader", os.path.join(_PROJECT_ROOT, "core", "model_loader.py")
    )
    _data_pipeline = _import_module_direct(
        "core.data_pipeline", os.path.join(_PROJECT_ROOT, "core", "data_pipeline.py")
    )
    parse_model_args_llava = _model_loader.parse_model_args
    load_model_from_args = _model_loader.load_model_from_args
    create_data_loader = _data_pipeline.create_data_loader

    cache_dir = os.environ.get("HF_HOME", None)

    model_args_dict = parse_model_args_llava(args.model_args)
    tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
    args.conv_mode = conv_template
    model.eval()
    model.tie_weights()

    num_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size

    questions, dataset_dict = load_dataset_as_questions(
        task_name=args.task,
        video_folder=args.video_folder,
        image_folder=args.image_folder,
        hf_cache_dir=cache_dir,
        limit=args.limit,
    )

    data_loader = create_data_loader(
        questions, args.image_folder, args.batch_size, args.num_workers,
        tokenizer, image_processor, model.config, args.task, args.conv_mode,
        video_folder=args.video_folder, video_fps=args.video_fps,
        frames_upbound=args.frames_upbound, force_sample=args.force_sample,
    )

    label_list, answer_to_idx = build_label_set(questions)
    num_classes = len(label_list)
    print(f"[INFO] Classes ({num_classes}): {label_list}")

    all_features = {layer: [] for layer in range(num_layers)}
    all_labels = []
    all_qids = []

    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions), desc="Extracting answer features"
    ):
        question_id = line["q_id"]
        answer = resolve_answer(line)

        if answer not in answer_to_idx:
            print(f"[WARN] answer '{answer}' not in label set, skipping {question_id}")
            continue

        label_idx = answer_to_idx[answer]

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        # === prefill만 필요: prepare_multimodal + forward (generate 오버헤드 제거) ===
        with torch.inference_mode():
            (
                _, position_ids, attention_mask, _, inputs_embeds, _
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor,
                modalities=[effective_modality], image_sizes=original_image_sizes,
            )
            output = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # 전체 layer의 last token → GPU stack → single CPU transfer
        hidden_states = output.hidden_states  # tuple of (1, seq_len, hidden_dim) × (num_layers+1)
        # stack: (L, D) on GPU
        layer_stack = torch.stack(
            [hidden_states[l][0, -1, :] for l in range(num_layers)], dim=0
        )
        layer_stack_cpu = layer_stack.cpu().to(torch.float16)  # (L, D) 1회 전송
        for layer_idx in range(num_layers):
            all_features[layer_idx].append(layer_stack_cpu[layer_idx])
        del layer_stack, layer_stack_cpu, output

        all_labels.append(label_idx)
        all_qids.append(question_id)

    return all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, args.task, {
        "hidden_dim": hidden_dim,
        "token_type": "answer",
    }


# ============================================================
#  Qwen3-VL 계열 추출
# ============================================================

def load_video_frames(video_path, num_frames=8, resize=None):
    """decord로 비디오 프레임 로드 -> PIL Image 리스트.

    Args:
        resize: (height, width) tuple. 지정 시 모든 프레임을 해당 크기로 리사이즈.
                Qwen3-VL에서 vision token 수를 고정하기 위해 사용.
    """
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(f) for f in frames]
    if resize is not None:
        h, w = resize
        pil_frames = [f.resize((w, h)) for f in pil_frames]
    return pil_frames, len(indices)


def extract_features_qwen3_vl(args):
    """Qwen3-VL 계열 모델에서 answer token 위치의 hidden states 추출."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_args_dict = parse_model_args(args.model_args)
    pretrained = model_args_dict.get("pretrained", "")
    cache_dir = os.environ.get("HF_HOME", None)

    print(f"[MODEL] Loading Qwen3-VL: {pretrained}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pretrained,
        dtype="auto",
        device_map=model_args_dict.get("device_map", "auto"),
        cache_dir=cache_dir,
    )
    # Processor kwargs (min_pixels, max_pixels 등)
    processor_kwargs = {}
    for pk in ("min_pixels", "max_pixels"):
        if pk in model_args_dict:
            processor_kwargs[pk] = int(model_args_dict[pk])
    processor = AutoProcessor.from_pretrained(pretrained, cache_dir=cache_dir, **processor_kwargs)
    model.eval()

    max_num_frames = int(model_args_dict.get("max_num_frames", args.frames_upbound))

    # 프레임 리사이즈 (vision token 수 고정용)
    resize_frames = None
    if args.resize_frames:
        parts = args.resize_frames.split("x")
        resize_frames = (int(parts[0]), int(parts[1]))

    model_name = os.path.basename(pretrained.rstrip("/"))
    num_layers = model.config.text_config.num_hidden_layers + 1
    hidden_dim = model.config.text_config.hidden_size

    print(f"[INFO] num_layers: {num_layers}, hidden_dim: {hidden_dim}")
    if processor_kwargs:
        print(f"[INFO] processor kwargs: {processor_kwargs}")
    print(f"[INFO] max_num_frames: {max_num_frames}")
    if resize_frames:
        print(f"[INFO] resize_frames: {resize_frames[0]}x{resize_frames[1]}")

    # Dataset
    questions, dataset_dict = load_dataset_as_questions(
        task_name=args.task,
        video_folder=args.video_folder,
        image_folder=args.image_folder,
        hf_cache_dir=cache_dir,
        limit=args.limit,
    )

    label_list, answer_to_idx = build_label_set(questions)
    num_classes = len(label_list)
    print(f"[INFO] Classes ({num_classes}): {label_list}")

    language_model = model.model.language_model

    all_features = {layer: [] for layer in range(num_layers)}
    all_labels = []
    all_qids = []

    for line in tqdm(questions, desc="Extracting answer features (Qwen3-VL)"):
        question_id = line["q_id"]
        answer = resolve_answer(line)

        if answer not in answer_to_idx:
            print(f"[WARN] answer '{answer}' not in label set, skipping {question_id}")
            continue

        label_idx = answer_to_idx[answer]

        # 비디오 로드
        video_rel = line["video"]
        if args.video_folder and not os.path.isabs(video_rel):
            video_path = os.path.join(args.video_folder, video_rel)
        else:
            video_path = video_rel

        if not os.path.exists(video_path):
            hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface"))
            video_path = os.path.join(hf_cache, video_rel)

        frames, actual_num_frames = load_video_frames(video_path, max_num_frames, resize=resize_frames)

        # Qwen3-VL 프롬프트 구성
        question_text = line["question"]
        video_content = {"type": "video", "video": frames}
        if max_num_frames:
            video_content["nframes"] = max_num_frames
        messages = [
            {"role": "user", "content": [
                video_content,
                {"type": "text", "text": question_text},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], return_tensors="pt")

        device = model.device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        seq_len = inputs["input_ids"].shape[1]

        # Hook 등록: embedding layer (layer 0) + decoder layers (layer 1..N)
        hidden_states_cache = {}
        hooks = []

        def make_embed_hook():
            def hook_fn(module, input, output):
                hidden_states_cache[0] = output.detach()
            return hook_fn

        def make_layer_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states_cache[idx] = output[0].detach()
                else:
                    hidden_states_cache[idx] = output.detach()
            return hook_fn

        hooks.append(language_model.embed_tokens.register_forward_hook(make_embed_hook()))
        for layer_idx, layer in enumerate(language_model.layers):
            hooks.append(layer.register_forward_hook(make_layer_hook(layer_idx + 1)))

        with torch.inference_mode():
            model(**inputs)

        for h in hooks:
            h.remove()

        # Answer token (마지막 토큰) hidden states 추출
        for layer_idx in range(num_layers):
            hs = hidden_states_cache[layer_idx]
            if hs.dim() == 3:
                hs = hs[0]
            answer_hs = hs[-1, :]  # (hidden_dim,)
            answer_feature = answer_hs.cpu().to(torch.float16)
            all_features[layer_idx].append(answer_feature)

        all_labels.append(label_idx)
        all_qids.append(question_id)

        del hidden_states_cache, inputs
        torch.cuda.empty_cache()

    return all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, args.task, {
        "hidden_dim": hidden_dim,
        "token_type": "answer",
    }


# ============================================================
#  공통: 저장
# ============================================================

def save_results(output_dir, all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, task, extra_meta):
    os.makedirs(output_dir, exist_ok=True)

    labels_array = np.array(all_labels, dtype=np.int64)
    np.save(os.path.join(output_dir, "labels.npy"), labels_array)
    np.save(os.path.join(output_dir, "qids.npy"), np.array(all_qids))

    def _save_layer(layer_idx):
        features = torch.stack(all_features[layer_idx], dim=0).numpy()
        np.save(os.path.join(output_dir, f"features_layer_{layer_idx}.npy"), features)

    with ThreadPoolExecutor(max_workers=min(8, num_layers)) as executor:
        list(executor.map(_save_layer, range(num_layers)))

    meta = {
        "num_layers": num_layers,
        "num_samples": len(all_labels),
        "num_classes": num_classes,
        "label_list": label_list,
        "model_name": model_name,
        "task": task,
        **extra_meta,
    }
    np.save(os.path.join(output_dir, "meta.npy"), meta)

    feat_dim = all_features[0][0].shape[0]
    print(f"[DONE] Saved {len(all_labels)} samples, {num_layers} layers to {output_dir}")
    print(f"  Feature dim per layer: {feat_dim}  (= hidden_dim, single answer token)")
    print(f"  Labels distribution: {np.bincount(labels_array, minlength=num_classes)}")


# ============================================================
#  메인
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract answer token features per layer for linear probing")
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "llava", "qwen3_vl"],
                        help="모델 타입 (auto: config.json에서 자동 감지)")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="e.g., output/MODEL_NAME/answer_probe_features/TASK")
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resize_frames", type=str, default=None,
                        help="Qwen3-VL용: 프레임을 고정 크기로 리사이즈하여 vision token 수 통일. 예: 336x336")

    args = parser.parse_args()

    # 모델 타입 감지
    model_type = args.model_type
    if model_type == "auto":
        model_args_dict = parse_model_args(args.model_args)
        pretrained = model_args_dict.get("pretrained", "")
        model_type = detect_model_type(pretrained)
        print(f"[INFO] Auto-detected model type: {model_type}")

    if model_type == "qwen3_vl":
        results = extract_features_qwen3_vl(args)
    else:
        results = extract_features_llava(args)

    all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, task, extra_meta = results
    save_results(args.output_dir, all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, task, extra_meta)
