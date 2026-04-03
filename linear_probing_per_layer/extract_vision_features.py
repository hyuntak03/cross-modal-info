"""
Layer별 Vision Token Hidden States 추출 스크립트.

각 layer에서 vision token의 hidden states를 temporal 방향으로 concat하여 저장.
- Features: (num_samples, num_vision_tokens, hidden_dim) → flatten to (num_samples, num_vision_tokens * hidden_dim)
- Labels: GT answer의 candidate index (0~N-1)

지원 모델:
  - LLaVA 계열 (LLaVA-OneVision, LLaVA-Video 등)
  - Qwen3-VL 계열 (Qwen3-VL-4B-Instruct 등)

Usage (LLaVA):
    python linear_probing_per_layer/extract_vision_features.py \
        --model_args "pretrained=...,conv_template=qwen_1_5,device_map=auto" \
        --task direction_testbed_ablation_8way \
        --output_dir output/linear_probe_features

Usage (Qwen3-VL):
    python linear_probing_per_layer/extract_vision_features.py \
        --model_type qwen3_vl \
        --model_args "pretrained=/path/to/Qwen3-VL-4B-Instruct" \
        --task direction_testbed_ablation_8way \
        --output_dir output/linear_probe_features_qwen3vl
"""

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import ast
import importlib.util
import json
import math
import string

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

torch.set_grad_enabled(False)

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

def compute_frame_boundaries(model, model_name, input_ids, image_tensor, image_sizes, modality):
    """vision token의 프레임별 위치를 계산 (LLaVA)."""
    from llava.constants import IMAGE_TOKEN_INDEX

    mm_newline_position = getattr(model.config, "mm_newline_position", "one_token")
    mm_spatial_pool_mode = getattr(model.config, "mm_spatial_pool_mode", "bilinear")

    vision_tower = model.get_vision_tower()
    num_patches_per_side = vision_tower.num_patches_per_side
    num_patches_per_frame = num_patches_per_side * num_patches_per_side

    if isinstance(image_tensor, list):
        num_frames = image_tensor[0].shape[0]
    else:
        num_frames = image_tensor.shape[0]

    stride = getattr(model.config, "mm_spatial_pool_stride", 2)

    if mm_spatial_pool_mode == "bilinear":
        pooled_h = math.ceil(num_patches_per_side / stride)
        pooled_w = math.ceil(num_patches_per_side / stride)
    else:
        pooled_h = num_patches_per_side // stride
        pooled_w = num_patches_per_side // stride

    tokens_per_frame = pooled_h * pooled_w

    if mm_newline_position == "one_token":
        total_vis = num_frames * tokens_per_frame + 1
    elif mm_newline_position == "frame":
        tokens_per_frame_with_nl = tokens_per_frame + 1
        total_vis = num_frames * tokens_per_frame_with_nl
    elif mm_newline_position == "grid":
        grid_h = int(math.sqrt(tokens_per_frame))
        tokens_per_frame_grid = grid_h * (grid_h + 1)
        total_vis = num_frames * tokens_per_frame_grid
        tokens_per_frame = tokens_per_frame_grid
    elif mm_newline_position == "no_token":
        total_vis = num_frames * tokens_per_frame
    else:
        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

    image_token_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]

    frame_ranges = []
    offset = image_token_pos

    if mm_newline_position == "one_token":
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))
    elif mm_newline_position == "frame":
        for f in range(num_frames):
            start = offset + f * tokens_per_frame_with_nl
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))
    elif mm_newline_position == "grid":
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))
    else:  # no_token
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))

    return frame_ranges, total_vis, tokens_per_frame, num_frames


def extract_features_llava(args):
    """LLaVA 계열 모델의 feature 추출."""
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
        zip(data_loader, questions), total=len(questions), desc="Extracting features"
    ):
        question_id = line["q_id"]
        answer = resolve_answer(line)

        if answer not in answer_to_idx:
            print(f"[WARN] answer '{answer}' not in label set, skipping {question_id}")
            continue

        label_idx = answer_to_idx[answer]

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        frame_ranges, total_vis, tokens_per_frame, num_frames = compute_frame_boundaries(
            model, model_name, input_ids, image_tensor, original_image_sizes, modality
        )

        all_vision_indices = []
        for fr in frame_ranges:
            all_vision_indices.extend(fr)

        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        inps = {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": original_image_sizes,
            "modalities": [effective_modality],
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": 1,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output = model.generate(**inps)

        prefill_hidden = output['hidden_states'][0]

        for layer_idx in range(num_layers):
            layer_hs = prefill_hidden[layer_idx]
            vision_hs = layer_hs[0, all_vision_indices, :]
            concat_feature = vision_hs.reshape(-1).cpu().to(torch.float16)
            all_features[layer_idx].append(concat_feature)

        all_labels.append(label_idx)
        all_qids.append(question_id)

    return all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, args.task, {
        "num_frames": num_frames,
        "tokens_per_frame": tokens_per_frame,
        "hidden_dim": hidden_dim,
    }


# ============================================================
#  Qwen3-VL 계열 추출
# ============================================================

def load_video_frames(video_path, num_frames=8):
    """decord로 비디오 프레임 로드 → PIL Image 리스트."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames], len(indices)


def extract_features_qwen3_vl(args):
    """Qwen3-VL 계열 모델의 feature 추출."""
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
    processor = AutoProcessor.from_pretrained(pretrained, cache_dir=cache_dir)
    model.eval()

    model_name = os.path.basename(pretrained.rstrip("/"))
    num_layers = model.config.text_config.num_hidden_layers + 1  # embedding + decoder layers
    hidden_dim = model.config.text_config.hidden_size

    VIDEO_PAD_TOKEN_ID = processor.tokenizer.encode("<|video_pad|>", add_special_tokens=False)[0]
    print(f"[INFO] video_pad token id: {VIDEO_PAD_TOKEN_ID}")
    print(f"[INFO] num_layers: {num_layers}, hidden_dim: {hidden_dim}")

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

    # Forward hook으로 모든 layer hidden states 수집
    # Qwen3VLForConditionalGeneration → model (Qwen3VLModel) → language_model (Qwen3VLTextModel)
    #   → embed_tokens, layers[0..N-1], norm
    language_model = model.model.language_model

    all_features = {layer: [] for layer in range(num_layers)}
    all_labels = []
    all_qids = []
    expected_num_vision_tokens = None

    for line in tqdm(questions, desc="Extracting features (Qwen3-VL)"):
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
            # HF cache fallback
            hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface"))
            video_path = os.path.join(hf_cache, video_rel)

        frames, actual_num_frames = load_video_frames(video_path, args.frames_upbound)

        # Qwen3-VL 프롬프트 구성
        question_text = line["question"]
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question_text},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], return_tensors="pt")

        # GPU로 이동
        device = model.device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # vision token 위치 파악
        input_ids = inputs["input_ids"][0]
        vision_mask = (input_ids == VIDEO_PAD_TOKEN_ID)
        vision_indices = vision_mask.nonzero(as_tuple=True)[0].tolist()
        num_vision_tokens = len(vision_indices)

        if expected_num_vision_tokens is None:
            expected_num_vision_tokens = num_vision_tokens
            print(f"[INFO] Vision tokens per sample: {num_vision_tokens} "
                  f"(seq_len={input_ids.shape[0]})")
        elif num_vision_tokens != expected_num_vision_tokens:
            print(f"[WARN] {question_id}: vision tokens={num_vision_tokens} "
                  f"(expected {expected_num_vision_tokens}), skipping")
            continue

        # Hook 등록: embedding layer (layer 0) + decoder layers (layer 1..N)
        hidden_states_cache = {}
        hooks = []

        def make_embed_hook():
            def hook_fn(module, input, output):
                hidden_states_cache[0] = output.detach()
            return hook_fn

        def make_layer_hook(idx):
            def hook_fn(module, input, output):
                # decoder layer output: (hidden_states, ...) or BaseModelOutput
                if isinstance(output, tuple):
                    hidden_states_cache[idx] = output[0].detach()
                else:
                    hidden_states_cache[idx] = output.detach()
            return hook_fn

        hooks.append(language_model.embed_tokens.register_forward_hook(make_embed_hook()))
        for layer_idx, layer in enumerate(language_model.layers):
            hooks.append(layer.register_forward_hook(make_layer_hook(layer_idx + 1)))

        # Forward pass
        with torch.inference_mode():
            model(**inputs)

        # Hook 제거
        for h in hooks:
            h.remove()

        # Vision token hidden states 추출
        for layer_idx in range(num_layers):
            hs = hidden_states_cache[layer_idx]  # (1, seq_len, hidden_dim) or (seq_len, hidden_dim)
            if hs.dim() == 3:
                hs = hs[0]
            vision_hs = hs[vision_indices, :]  # (num_vision_tokens, hidden_dim)
            concat_feature = vision_hs.reshape(-1).cpu().to(torch.float16)
            all_features[layer_idx].append(concat_feature)

        all_labels.append(label_idx)
        all_qids.append(question_id)

        # 메모리 정리
        del hidden_states_cache, inputs
        torch.cuda.empty_cache()

    tokens_per_frame = expected_num_vision_tokens // max(actual_num_frames, 1)

    return all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, args.task, {
        "num_frames": actual_num_frames,
        "tokens_per_frame": tokens_per_frame,
        "hidden_dim": hidden_dim,
    }


# ============================================================
#  공통: 저장
# ============================================================

def save_results(output_dir, all_features, all_labels, all_qids, num_layers, num_classes, label_list, model_name, task, extra_meta):
    os.makedirs(output_dir, exist_ok=True)

    labels_array = np.array(all_labels, dtype=np.int64)
    np.save(os.path.join(output_dir, "labels.npy"), labels_array)
    np.save(os.path.join(output_dir, "qids.npy"), np.array(all_qids))

    for layer_idx in range(num_layers):
        features = torch.stack(all_features[layer_idx], dim=0).numpy()
        np.save(os.path.join(output_dir, f"features_layer_{layer_idx}.npy"), features)

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

    feat_dim = extra_meta.get("num_frames", 0) * extra_meta.get("tokens_per_frame", 0) * extra_meta.get("hidden_dim", 0)
    print(f"[DONE] Saved {len(all_labels)} samples, {num_layers} layers to {output_dir}")
    print(f"  Feature dim per layer: {all_features[0][0].shape[0]}  "
          f"[{extra_meta.get('num_frames',0)} frames x {extra_meta.get('tokens_per_frame',0)} tokens x {extra_meta.get('hidden_dim',0)} dim]")
    print(f"  Labels distribution: {np.bincount(labels_array, minlength=num_classes)}")


# ============================================================
#  메인
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vision token features per layer for linear probing")
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "llava", "qwen3_vl"],
                        help="모델 타입 (auto: config.json에서 자동 감지)")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/linear_probe_features")
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

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
