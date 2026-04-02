"""
Layer별 Vision Token Hidden States 추출 스크립트.

각 layer에서 vision token의 hidden states를 temporal 방향으로 concat하여 저장.
- Features: (num_samples, num_frames * tokens_per_frame, hidden_dim) → flatten to (num_samples, num_frames * tokens_per_frame * hidden_dim)
- Labels: GT answer의 candidate index (0~N-1)

Usage:
    python linear_probing_per_layer/extract_vision_features.py \
        --model_args "pretrained=...,conv_template=qwen_1_5,device_map=auto" \
        --task direction_testbed_ablation_8way \
        --output_dir output/linear_probe_features
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import ast
import math
import string

import numpy as np
import torch
from tqdm import tqdm

torch.set_grad_enabled(False)

from llava.constants import IMAGE_TOKEN_INDEX

from core.data_pipeline import create_data_loader
from core.model_loader import parse_model_args, load_model_from_args
from core.dataset_loader import load_dataset_as_questions, list_tasks


# ============================================================
#  프레임 경계 계산 (video_information_flow.py에서 가져옴)
# ============================================================

def compute_frame_boundaries(model, model_name, input_ids, image_tensor, image_sizes, modality):
    """vision token의 프레임별 위치를 계산."""
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


def extract_features(args):
    cache_dir = os.environ.get("HF_HOME", None)

    # Model 로드
    model_args_dict = parse_model_args(args.model_args)
    tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
    args.conv_mode = conv_template
    model.eval()
    model.tie_weights()

    num_layers = model.config.num_hidden_layers + 1  # embedding layer 포함

    # Dataset 로드
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

    # 전체 샘플의 GT answer 텍스트에서 unique label set 구성
    # (샘플마다 candidate 순서가 다를 수 있으므로 candidates 기준 X, answer 텍스트 기준 O)
    unique_answers = set()
    for q in questions:
        # answer가 letter이면 candidates에서 실제 텍스트로 변환
        ans = str(q["answer"]).strip()
        if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
            candidates_raw = q.get("candidates", [])
            if isinstance(candidates_raw, str):
                candidates_raw = ast.literal_eval(candidates_raw)
            idx = ord(ans.upper()) - ord('A')
            if idx < len(candidates_raw):
                ans = str(candidates_raw[idx]).strip()
        unique_answers.add(ans)

    label_list = sorted(unique_answers)  # 정렬해서 고정된 순서 보장
    answer_to_idx = {a: i for i, a in enumerate(label_list)}
    num_classes = len(label_list)
    print(f"[INFO] Classes ({num_classes}): {label_list}")

    # 결과 저장용
    # layer별로 features를 모으기 (메모리 절약을 위해 리스트로)
    all_features = {layer: [] for layer in range(num_layers)}
    all_labels = []
    all_qids = []

    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions), desc="Extracting features"
    ):
        question_id = line["q_id"]
        answer = str(line["answer"]).strip()

        # answer가 letter이면 candidates에서 실제 텍스트로 변환
        if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
            candidates_raw = line.get("candidates", [])
            if isinstance(candidates_raw, str):
                candidates_raw = ast.literal_eval(candidates_raw)
            idx = ord(answer.upper()) - ord('A')
            if idx < len(candidates_raw):
                answer = str(candidates_raw[idx]).strip()

        if answer not in answer_to_idx:
            print(f"[WARN] answer '{answer}' not in label set, skipping {question_id}")
            continue

        label_idx = answer_to_idx[answer]

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        # 프레임 경계 계산
        frame_ranges, total_vis, tokens_per_frame, num_frames = compute_frame_boundaries(
            model, model_name, input_ids, image_tensor, original_image_sizes, modality
        )

        # vision token의 전체 인덱스 (temporal concat)
        all_vision_indices = []
        for fr in frame_ranges:
            all_vision_indices.extend(fr)

        # Forward pass with hidden states
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

        # hidden_states[0] = prefill step, [layer_idx] = (batch, seq_len, hidden_dim)
        # vision token 위치에서 hidden states 추출
        prefill_hidden = output['hidden_states'][0]  # prefill step

        for layer_idx in range(num_layers):
            layer_hs = prefill_hidden[layer_idx]  # (1, seq_len, hidden_dim)
            # vision token 전체를 pooling 없이 그대로 flatten
            vision_hs = layer_hs[0, all_vision_indices, :]  # (num_frames * tokens_per_frame, hidden_dim)
            concat_feature = vision_hs.reshape(-1).cpu().to(torch.float16)  # (num_frames * tokens_per_frame * hidden_dim,)
            all_features[layer_idx].append(concat_feature)

        all_labels.append(label_idx)
        all_qids.append(question_id)

    # 저장
    os.makedirs(args.output_dir, exist_ok=True)

    labels_array = np.array(all_labels, dtype=np.int64)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels_array)
    np.save(os.path.join(args.output_dir, "qids.npy"), np.array(all_qids))

    for layer_idx in range(num_layers):
        features = torch.stack(all_features[layer_idx], dim=0).numpy()  # (N, num_frames * hidden_dim)
        np.save(os.path.join(args.output_dir, f"features_layer_{layer_idx}.npy"), features)

    # 메타 정보 저장
    meta = {
        "num_layers": num_layers,
        "num_samples": len(all_labels),
        "num_classes": num_classes,
        "num_frames": num_frames,
        "tokens_per_frame": tokens_per_frame,
        "hidden_dim": model.config.hidden_size,
        "label_list": label_list,
        "model_name": model_name,
        "task": args.task,
    }
    np.save(os.path.join(args.output_dir, "meta.npy"), meta)

    feat_dim = num_frames * tokens_per_frame * model.config.hidden_size
    print(f"[DONE] Saved {len(all_labels)} samples, {num_layers} layers to {args.output_dir}")
    print(f"  Feature shape per layer: ({len(all_labels)}, {feat_dim})  "
          f"[{num_frames} frames x {tokens_per_frame} tokens x {model.config.hidden_size} dim]")
    print(f"  Labels distribution: {np.bincount(labels_array, minlength=num_classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vision token features per layer for linear probing")
    parser.add_argument("--model_args", type=str, required=True)
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
    extract_features(args)
