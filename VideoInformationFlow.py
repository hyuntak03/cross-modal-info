# CrossFrameFlow.py
# Cross-Frame Interaction Analysis for LLaVA-OneVision Video Models
# - 기존 InformationFlow.py의 Attention Knockout 프레임워크 기반
# - map-the-flow의 cross-frame interaction 분석 기법 통합
# - LLaVA-OneVision의 spatial_unpad + 2dPool 비디오 토큰 구조 대응

import copy
import math
import itertools
import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

torch.set_grad_enabled(False)
tqdm.pandas()

from matplotlib import pyplot as plt
import seaborn as sns

from types import MethodType
from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerateOutput

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, process_video_with_decord
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from methods import trace_with_attn_block_llava, set_block_attn_hooks_llava, remove_wrapper_llava
from utils import generate_plot
from InformationFlow import (
    CustomDataset, collate_fn, create_data_loader,
    find_token_range, generate_llava, blockdesc2range
)
from dataset_loader import load_dataset_as_questions, list_tasks


# ============================================================
#  model_args 파싱 (lmms_eval 호환)
# ============================================================

def parse_model_args(args_string):
    """
    lmms_eval 스타일 model_args 파싱.
    "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto"
    → {"pretrained": "...", "conv_template": "qwen_1_5", "device_map": "auto"}
    """
    if not args_string:
        return {}
    result = {}
    for item in args_string.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        # bool/int 자동 변환
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


def load_model_from_args(model_args_dict):
    """
    model_args dict로부터 모델을 로드한다.
    lmms_eval 패턴 지원:
      - pretrained: HF repo name 또는 local path
      - lora_pretrained: LoRA weight 경로 (optional)
      - conv_template: conversation template
      - device_map: "auto" 등
      - max_frames_num, video_decode_backend, ...
    """
    pretrained = model_args_dict.get("pretrained", "")
    lora_pretrained = model_args_dict.get("lora_pretrained", None)
    device_map = model_args_dict.get("device_map", "auto")
    attn_implementation = model_args_dict.get("attn_implementation", None)
    conv_template = model_args_dict.get("conv_template", "qwen_1_5")
    cache_dir = os.environ.get("HF_HOME", None)

    if lora_pretrained:
        # LoRA: lora_pretrained이 model_path, pretrained이 model_base
        model_path = lora_pretrained
        model_base = pretrained
        model_name = get_model_name_from_path(lora_pretrained)
        print(f"[MODEL] LoRA loading: base={pretrained}, lora={lora_pretrained}")
    else:
        # 일반: pretrained가 model_path
        model_path = pretrained
        model_base = None
        model_name = get_model_name_from_path(pretrained)
        print(f"[MODEL] Loading: {pretrained}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        device_map=device_map,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )
    model.eval()

    # video 관련 config 오버라이드
    if "max_frames_num" in model_args_dict:
        pass  # frames_upbound로 외부에서 제어
    if "mm_spatial_pool_stride" in model_args_dict:
        model.config.mm_spatial_pool_stride = model_args_dict["mm_spatial_pool_stride"]
    if "mm_spatial_pool_mode" in model_args_dict:
        model.config.mm_spatial_pool_mode = model_args_dict["mm_spatial_pool_mode"]

    return tokenizer, model, image_processor, context_len, model_name, conv_template


# ============================================================
#  프레임 경계 계산
# ============================================================

def compute_frame_boundaries(model, model_name, input_ids, image_tensor, image_sizes, modality):
    """
    LLaVA-OneVision 비디오 입력에서 프레임별 vision token 경계를 계산한다.

    Returns:
        frame_ranges: list of list[int]  -- frame_ranges[i] = [token_idx, ...] (absolute position in input_embeds)
        num_vis_tokens: int              -- 전체 vision token 수
        newline_token_idx: int or None   -- one_token newline 위치 (있으면)
    """
    mm_newline_position = getattr(model.config, "mm_newline_position", "one_token")
    mm_spatial_pool_mode = getattr(model.config, "mm_spatial_pool_mode", "bilinear")

    # Vision tower 정보
    vision_tower = model.get_vision_tower()
    num_patches_per_side = vision_tower.num_patches_per_side  # 27 for SigLIP-384
    num_patches_per_frame = num_patches_per_side * num_patches_per_side  # 729

    # 프레임 수: image_tensor shape = [T, C, H, W] 또는 list
    if isinstance(image_tensor, list):
        # image_tensor[0]이 [T, C, H, W] 형태
        num_frames = image_tensor[0].shape[0]
    else:
        num_frames = image_tensor.shape[0]

    # Spatial pooling 후 프레임당 토큰 수 계산
    stride = getattr(model.config, "mm_spatial_pool_stride", 2)

    if mm_spatial_pool_mode == "bilinear":
        pooled_h = math.ceil(num_patches_per_side / stride)
        pooled_w = math.ceil(num_patches_per_side / stride)
    else:  # average, max
        pooled_h = num_patches_per_side // stride
        pooled_w = num_patches_per_side // stride

    tokens_per_frame = pooled_h * pooled_w

    # mm_newline_position에 따른 추가 토큰
    if mm_newline_position == "one_token":
        # 전체 프레임 flatten 후 마지막에 newline 1개
        total_vis = num_frames * tokens_per_frame + 1
        has_trailing_newline = True
    elif mm_newline_position == "frame":
        # 프레임마다 newline 1개
        tokens_per_frame_with_nl = tokens_per_frame + 1
        total_vis = num_frames * tokens_per_frame_with_nl
        has_trailing_newline = False
    elif mm_newline_position == "grid":
        # 프레임마다 행별 newline
        grid_h = int(math.sqrt(tokens_per_frame))
        tokens_per_frame_grid = grid_h * (grid_h + 1)  # grid_h rows * (grid_h tokens + 1 newline)
        total_vis = num_frames * tokens_per_frame_grid
        tokens_per_frame = tokens_per_frame_grid
        has_trailing_newline = False
    elif mm_newline_position == "no_token":
        total_vis = num_frames * tokens_per_frame
        has_trailing_newline = False
    else:
        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

    # input_ids에서 <image> 토큰 위치 찾기
    image_token_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]

    # 프레임별 range 계산 (absolute position in input_embeds)
    frame_ranges = []
    offset = image_token_pos

    if mm_newline_position == "one_token":
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))
        newline_idx = offset + num_frames * tokens_per_frame
    elif mm_newline_position == "frame":
        for f in range(num_frames):
            start = offset + f * tokens_per_frame_with_nl
            end = start + tokens_per_frame  # newline 제외
            frame_ranges.append(list(range(start, end)))
        newline_idx = None
    elif mm_newline_position == "grid":
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))
        newline_idx = None
    else:  # no_token
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))
        newline_idx = None

    return frame_ranges, total_vis, newline_idx


def find_cross_frame_block_ranges(frame_ranges):
    """
    map-the-flow의 find_inter_frame_block_ranges와 동일한 로직.
    뒤 프레임이 앞 프레임을 attend하는 것을 차단하는 (query, key) 쌍을 생성한다.

    Returns:
        all_pairs: list of (target_row, source_col) -- attention mask에서 0으로 만들 위치들
    """
    all_pairs = []
    for i in range(1, len(frame_ranges)):
        query_tokens = frame_ranges[i]
        # 이전 모든 프레임의 토큰들
        key_tokens = list(itertools.chain.from_iterable(frame_ranges[:i]))
        for q in query_tokens:
            for k in key_tokens:
                all_pairs.append((q, k))
    return all_pairs


def find_intra_frame_block_ranges(frame_ranges):
    """
    같은 프레임 내 토큰 간 attention을 차단한다 (cross-frame의 반대 실험).
    프레임 내 spatial interaction의 중요성을 측정하기 위함.

    Returns:
        all_pairs: list of (target_row, source_col)
    """
    all_pairs = []
    for frame_tokens in frame_ranges:
        for q in frame_tokens:
            for k in frame_tokens:
                if q != k:
                    all_pairs.append((q, k))
    return all_pairs


def find_frame_to_text_block_ranges(frame_ranges, text_range):
    """
    비디오 프레임 → 텍스트 토큰 방향의 attention을 차단한다.
    비디오 정보가 텍스트(question/last)로 전달되는 경로를 분석.

    Args:
        frame_ranges: list of list[int] -- 프레임별 토큰 인덱스
        text_range: list[int]           -- 텍스트 토큰 인덱스 (question 또는 last)

    Returns:
        all_pairs: list of (target_row, source_col)
    """
    all_video_tokens = list(itertools.chain.from_iterable(frame_ranges))
    all_pairs = []
    for q in text_range:
        for k in all_video_tokens:
            all_pairs.append((q, k))
    return all_pairs


def find_perframe_to_text_block_ranges(frame_ranges, text_range):
    """
    개별 프레임 → 텍스트 토큰 방향의 attention을 프레임별로 차단한다.
    어떤 프레임이 답변에 더 중요한지 분석.

    Returns:
        per_frame_pairs: list of (list of (target_row, source_col), frame_idx)
    """
    per_frame_pairs = []
    for f_idx, frame_tokens in enumerate(frame_ranges):
        pairs = []
        for q in text_range:
            for k in frame_tokens:
                pairs.append((q, k))
        per_frame_pairs.append((pairs, f_idx))
    return per_frame_pairs


# ============================================================
#  원본 모델 실행 (generate_llava 기반)
# ============================================================

def run_original(model, inps, tokenizer, model_name, answer, args=None):
    with torch.inference_mode():
        model.old_generate = model.generate
        model.generate = MethodType(generate_llava, model)
        inputs_embeds_shape, output_details = model.generate(args=args, **inps)
        model.generate = model.old_generate

    answer_token_id = output_details['sequences']
    generated_first_id = answer_token_id[:, 0]
    decoded_generated_first_id = tokenizer.decode(generated_first_id.item())

    answer_cap = answer.capitalize()
    gt_token_ids = tokenizer.encode(answer_cap, add_special_tokens=False)
    gt_first_token_id = gt_token_ids[0]
    gt_first_token_id_tensor = torch.tensor([gt_first_token_id], device=generated_first_id.device)

    logits_first_answer_token = output_details['scores'][0]
    probs = torch.softmax(logits_first_answer_token, dim=-1)[0]
    gt_base_score = probs[gt_first_token_id_tensor].item()
    predicted_base_score = probs[generated_first_id].item()

    is_correct_bool = decoded_generated_first_id.strip().lower() == answer_cap.strip().lower()
    predicted_answer = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip().lower()

    return gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id_tensor, generated_first_id, inputs_embeds_shape, is_correct_bool


# ============================================================
#  Visualization
# ============================================================

def generate_crossframe_plot(data, save_file, x="layer", y="relative diff first",
                             hue="block_desc", layers=0):
    sns.set(context="notebook")
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))

    palette = sns.color_palette("Set2", n_colors=len(data[hue].unique()))
    ax = sns.lineplot(data, x=x, y=y, hue=hue, style=hue,
                      palette=palette, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"% change in {y}")
    ax.set_xlim(0, layers + 0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    plt.close()


# ============================================================
#  Main: Cross-Frame Information Flow Analysis
# ============================================================

def CrossFrameFlowAna(args):

    cache_dir = os.environ.get("HF_HOME", None)

    # Model: model_args 스타일 또는 기존 --model-path 스타일
    if args.model_args:
        model_args_dict = parse_model_args(args.model_args)
        tokenizer, model, image_processor, context_len, model_name, conv_template = \
            load_model_from_args(model_args_dict)
        args.conv_mode = conv_template
    else:
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name,
            device_map="auto", attn_implementation=None, cache_dir=cache_dir
        )
        model.eval()

    # Dataset: HuggingFace task 또는 CSV에서 로딩
    if args.task:
        # HuggingFace 데이터셋 로딩
        task_name = args.task
        questions, dataset_dict = load_dataset_as_questions(
            task_name=args.task,
            video_folder=args.video_folder,
            image_folder=args.image_folder,
            hf_cache_dir=cache_dir,
            max_samples=args.max_samples,
        )
    elif args.refined_dataset:
        # 기존 CSV 로딩 (하위 호환)
        if args.option == "MCQ":
            task_name = "MCQ"
        else:
            task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
        questions, dataset_dict = load_dataset_as_questions(
            csv_path=args.refined_dataset,
            max_samples=args.max_samples,
        )
    else:
        raise ValueError("--task (HuggingFace) 또는 --refined_dataset (CSV) 중 하나는 필수")

    data_loader = create_data_loader(
        questions, args.image_folder, args.batch_size, args.num_workers,
        tokenizer, image_processor, model.config, task_name, args.conv_mode,
        video_folder=args.video_folder, video_fps=args.video_fps,
        frames_upbound=args.frames_upbound, force_sample=args.force_sample
    )

    # Run analysis
    results = []
    index = 0

    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality), line in tqdm(
            zip(data_loader, questions), total=len(questions)):

        question_id = line["q_id"]

        if "video" in line and line["video"] != "":
            img_id = str(line["video"])
        else:
            img_id = str(line["img_id"])

        # video only
        is_video = (modality == "video")
        if not is_video:
            print(f"[SKIP] {question_id} is not a video sample, skipping cross-frame analysis.")
            continue

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        inps = {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": original_image_sizes,
            "modalities": [effective_modality],
            "do_sample": True if args.temperature > 0 else False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        question = dataset_dict[question_id]["question"]
        answer = dataset_dict[question_id]["answer"]

        # Baseline forward
        gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id, predicted_first_token_id, inputs_embeds_shape, is_correct_bool = \
            run_original(model, inps, tokenizer, model_name, answer, args=args)

        if not is_correct_bool:
            is_correct = False
        else:
            is_correct = True
            index += 1
            print(f"Correct samples so far: {index}")

        # Inference only 모드
        if args.inference_only:
            results.append({
                "question_id": question_id,
                "image": img_id,
                "goden answer": answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "question": question,
                "gt_base_score": gt_base_score,
                "predicted_base_score": predicted_base_score,
            })
            continue

        # ========== 프레임 경계 계산 ==========
        frame_ranges, num_vis_tokens, newline_idx = compute_frame_boundaries(
            model, model_name, input_ids, image_tensor, original_image_sizes, modality
        )
        num_frames = len(frame_ranges)
        print(f"  [INFO] {question_id}: {num_frames} frames, {num_vis_tokens} vision tokens, "
              f"tokens_per_frame={len(frame_ranges[0])}")

        # ========== 텍스트 토큰 범위 ==========
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        ntoks = input_ids.shape[1] + image_dim - 1
        last_token_idx = ntoks - 1

        question_range = blockdesc2range(
            "Question", dataset_dict, question_id, input_ids,
            inputs_embeds_shape, tokenizer, model_name, args=args
        )

        # ========== Block description별 pair 생성 ==========
        block_targets = []

        for target_name in args.cross_frame_targets:
            if target_name == "cross-frame":
                pairs = find_cross_frame_block_ranges(frame_ranges)
                block_targets.append((pairs, "Cross-Frame (Frame_i -/-> Frame_j<i)"))

            elif target_name == "intra-frame":
                pairs = find_intra_frame_block_ranges(frame_ranges)
                block_targets.append((pairs, "Intra-Frame (within-frame spatial)"))

            elif target_name == "video-to-question":
                pairs = find_frame_to_text_block_ranges(frame_ranges, question_range)
                block_targets.append((pairs, "Video -/-> Question"))

            elif target_name == "video-to-last":
                pairs = find_frame_to_text_block_ranges(frame_ranges, [last_token_idx])
                block_targets.append((pairs, "Video -/-> Last"))

            elif target_name == "question-to-last":
                pairs = [(last_token_idx, q) for q in question_range]
                block_targets.append((pairs, "Question -/-> Last"))

            elif target_name == "perframe-to-last":
                per_frame = find_perframe_to_text_block_ranges(frame_ranges, [last_token_idx])
                for pairs, f_idx in per_frame:
                    block_targets.append((pairs, f"Frame{f_idx} -/-> Last"))

            else:
                raise ValueError(f"Unknown cross-frame target: {target_name}")

        # ========== Attention Knockout ==========
        for block_pairs, block_desc in block_targets:
            if len(block_pairs) == 0:
                continue

            if args.block_all_layers:
                # 전체 레이어에서 한번에 knockout
                block_config = {
                    l: copy.deepcopy(block_pairs)
                    for l in range(model.config.num_hidden_layers)
                }
                inps["max_new_tokens"] = 1

                new_probs, knocked_predicted_answer = trace_with_attn_block_llava(
                    model, inps, block_config, block_desc, model_name,
                    tokenizer=tokenizer, last_token_idx=last_token_idx
                )

                new_score_gt = new_probs[gt_first_token_id].cpu().item()
                new_score_predicted = new_probs[predicted_first_token_id].cpu().item()

                re_gt = {
                    "question_id": question_id,
                    "image": img_id,
                    "goden answer": answer,
                    "origin_predicted_answer": predicted_answer,
                    "knocked_predicted_answer": knocked_predicted_answer,
                    "is_correct": is_correct,
                    "question": question,
                    "num_frames": num_frames,
                    "block_desc": block_desc,
                    "layer": "all",
                    "trace_target": "gt_answer",
                    "base_score_first": gt_base_score,
                    "new_score_first": new_score_gt,
                    "relative diff first": (new_score_gt - gt_base_score) * 100.0 / gt_base_score if gt_base_score != 0 else 0.0,
                }
                results.append(re_gt)

                if not is_correct:
                    re_pred = {
                        "question_id": question_id,
                        "image": img_id,
                        "goden answer": answer,
                        "origin_predicted_answer": predicted_answer,
                        "knocked_predicted_answer": knocked_predicted_answer,
                        "is_correct": is_correct,
                        "question": question,
                        "num_frames": num_frames,
                        "block_desc": block_desc,
                        "layer": "all",
                        "trace_target": "predicted_answer",
                        "base_score_first": predicted_base_score,
                        "new_score_first": new_score_predicted,
                        "relative diff first": (new_score_predicted - predicted_base_score) * 100.0 / predicted_base_score if predicted_base_score != 0 else 0.0,
                    }
                    results.append(re_pred)

            else:
                # Layer-wise sliding window knockout
                for layer in range(model.config.num_hidden_layers):
                    layerlist = [
                        l for l in range(
                            max(0, layer - args.window // 2),
                            min(model.config.num_hidden_layers, layer - (-args.window // 2))
                        )
                    ]
                    block_config = {
                        l: copy.deepcopy(block_pairs)
                        for l in layerlist
                    }
                    inps["max_new_tokens"] = 1

                    new_probs, knocked_predicted_answer = trace_with_attn_block_llava(
                        model, inps, block_config, block_desc, model_name,
                        tokenizer=tokenizer, last_token_idx=last_token_idx
                    )

                    new_score_gt = new_probs[gt_first_token_id].cpu().item()
                    new_score_predicted = new_probs[predicted_first_token_id].cpu().item()

                    re_gt = {
                        "question_id": question_id,
                        "image": img_id,
                        "goden answer": answer,
                        "origin_predicted_answer": predicted_answer,
                        "knocked_predicted_answer": knocked_predicted_answer,
                        "is_correct": is_correct,
                        "question": question,
                        "num_frames": num_frames,
                        "block_desc": block_desc,
                        "layer": layer,
                        "trace_target": "gt_answer",
                        "base_score_first": gt_base_score,
                        "new_score_first": new_score_gt,
                        "relative diff first": (new_score_gt - gt_base_score) * 100.0 / gt_base_score if gt_base_score != 0 else 0.0,
                    }
                    results.append(re_gt)

                    if not is_correct:
                        re_pred = {
                            "question_id": question_id,
                            "image": img_id,
                            "goden answer": answer,
                            "origin_predicted_answer": predicted_answer,
                            "knocked_predicted_answer": knocked_predicted_answer,
                            "is_correct": is_correct,
                            "question": question,
                            "num_frames": num_frames,
                            "block_desc": block_desc,
                            "layer": layer,
                            "trace_target": "predicted_answer",
                            "base_score_first": predicted_base_score,
                            "new_score_first": new_score_predicted,
                            "relative diff first": (new_score_predicted - predicted_base_score) * 100.0 / predicted_base_score if predicted_base_score != 0 else 0.0,
                        }
                        results.append(re_pred)

    # ========== 결과 저장 ==========
    if len(results) == 0:
        print("[WARN] No results collected.")
        return

    tmp = pd.DataFrame.from_records(results)
    model_name_safe = model_name.replace('-', '_').replace('.', '_')
    target_str = "+".join(args.cross_frame_targets)

    if args.inference_only:
        os.makedirs(f"output/inference_only/{model_name_safe}", exist_ok=True)
        dataset_name = args.task if args.task else args.refined_dataset.split("/")[-1].split(".csv")[0]
        out_path = f"output/inference_only/{model_name_safe}/{dataset_name}_crossframe_inference.csv"
        tmp.to_csv(out_path, index=False)
        acc = tmp["is_correct"].sum() / len(tmp) * 100
        print(f"\n{'='*50}")
        print(f"  Accuracy: {acc:.2f}% ({tmp['is_correct'].sum()}/{len(tmp)})")
        print(f"  Saved: {out_path}")
        print(f"{'='*50}")
        return

    save_dir = f"output/cross_frame_flow/{model_name_safe}/{task_name}/val/{target_str}"
    os.makedirs(save_dir, exist_ok=True)

    base_name = args.task if args.task else args.refined_dataset.split("/")[-1].split(".csv")[0]

    if args.block_all_layers:
        save_suffix = f"_block_all_layers_window{args.window}"
    else:
        save_suffix = f"_window{args.window}"

    csv_path = f"{save_dir}/{base_name}{save_suffix}.csv"
    tmp.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # GT answer tracing만 plot
    tmp_gt = tmp[tmp["trace_target"] == "gt_answer"] if "trace_target" in tmp.columns else tmp

    if args.block_all_layers:
        # block_all_layers: bar plot
        for label, df_sub in [("all", tmp_gt),
                               ("correct", tmp_gt[tmp_gt["is_correct"] == True]),
                               ("incorrect", tmp_gt[tmp_gt["is_correct"] == False])]:
            if len(df_sub) == 0:
                continue
            generate_plot(df_sub,
                          f"{save_dir}/{base_name}{save_suffix}_{label}.pdf",
                          y="relative diff first",
                          block_all_layers=True,
                          block_description=target_str)
    else:
        # layer-wise line plot
        for label, df_sub in [("all", tmp_gt),
                               ("correct", tmp_gt[tmp_gt["is_correct"] == True]),
                               ("incorrect", tmp_gt[tmp_gt["is_correct"] == False])]:
            if len(df_sub) == 0:
                continue
            pdf_path = f"{save_dir}/{base_name}{save_suffix}_{label}.pdf"
            generate_crossframe_plot(
                df_sub, pdf_path,
                x="layer", y="relative diff first", hue="block_desc",
                layers=model.config.num_hidden_layers
            )
            print(f"[PLOT] {pdf_path}")


# ============================================================
#  Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Frame Information Flow Analysis: "
                    "Trace cross-frame interaction dynamics via attention knockout in LLaVA-OneVision video models."
    )
    # ===== Model (둘 중 하나 사용) =====
    # 방법 1: lmms_eval 스타일 (추천)
    #   --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto"
    # 방법 2: 기존 스타일
    #   --model-path /local/path --conv-mode qwen_1_5
    parser.add_argument("--model_args", type=str, default=None,
                        help='lmms_eval 스타일 모델 인자. '
                             'e.g., "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto" '
                             'LoRA: "lora_pretrained=/path/to/lora,pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto"')
    parser.add_argument("--model-path", type=str, default=None,
                        help="모델 경로 (HF repo name 또는 local path). --model_args 미사용시.")
    parser.add_argument("--model-base", type=str, default=None,
                        help="LoRA base 모델 (기존 방식)")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--image-folder", type=str, default="")

    # 데이터셋 소스 (둘 중 하나 필수)
    parser.add_argument('--task', type=str, default=None,
                        help=f"HuggingFace task 이름. 사용 가능: {list_tasks()}")
    parser.add_argument('--refined_dataset', type=str, default=None,
                        help="CSV 파일 경로 (기존 방식, --task와 택일)")
    parser.add_argument('--max_samples', type=int, default=-1,
                        help="최대 샘플 수 제한 (-1이면 전체). 디버깅용.")

    # Video 관련
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)

    # Cross-frame 분석 타겟
    parser.add_argument("--cross_frame_targets", type=str, nargs='+',
                        default=["cross-frame"],
                        choices=["cross-frame", "intra-frame",
                                 "video-to-question", "video-to-last",
                                 "question-to-last", "perframe-to-last"],
                        help="Cross-frame analysis targets. "
                             "cross-frame: block inter-frame attention (Frame_i -/-> Frame_j<i). "
                             "intra-frame: block within-frame spatial attention. "
                             "video-to-question: block all video -> question. "
                             "video-to-last: block all video -> last token. "
                             "question-to-last: block question -> last token. "
                             "perframe-to-last: block each frame -> last token individually.")

    # Knockout 설정
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument('--block_all_layers', default=False, action="store_true",
                        help="Block attention across all layers at once")
    parser.add_argument('--block_ASSIST', default=False, action="store_true",
                        help="Also block ASSISTANT tokens in Instruction range")

    # MCQ 옵션
    parser.add_argument("--option", type=str, default="standard")

    # Inference only
    parser.add_argument('--inference_only', default=False, action="store_true",
                        help="Run inference only without knockout")

    args = parser.parse_args()

    # validation
    if not args.model_args and not args.model_path:
        parser.error("--model_args 또는 --model-path 중 하나는 필수")
    if not args.task and not args.refined_dataset:
        parser.error("--task 또는 --refined_dataset 중 하나는 필수")

    # certain_part_image는 cross-frame에서 사용 안 함
    args.certain_part_image = False

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    CrossFrameFlowAna(args)
