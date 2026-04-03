"""
Attention extraction CLI for LLaVA-NeXT VLMs.

Extracts full attention matrices and saves as .pt for bertviz-style visualization.

Usage:
  # Single image
  python Attention_map/extract_attention.py \
    --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto" \
    --image_path /path/to/image.jpg \
    --question "Describe this image." \
    --output_dir output/attention

  # Single video
  python Attention_map/extract_attention.py \
    --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto,max_frames_num=8" \
    --video_path /path/to/video.mp4 \
    --question "What is happening in the video?" \
    --output_dir output/attention

  # Batch (task)
  python Attention_map/extract_attention.py \
    --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto,max_frames_num=16" \
    --task mvbench \
    --video_folder /path/to/videos \
    --limit 5 \
    --output_dir output/attention
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import re

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

torch.set_grad_enabled(False)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import process_video_with_decord
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from core.model_loader import parse_model_args, load_model_from_args, load_model_legacy
from core.dataset_loader import load_dataset_as_questions, list_tasks
from core.data_pipeline import CustomDataset, collate_fn, create_data_loader

from Attention_map.attention_utils import (
    extract_attention,
    build_token_labels,
    collapse_vision_tokens,
    build_prompt,
    get_vision_grid_size,
    extract_answer_vision_attention,
    split_vision_attention_by_frames,
)


def save_attention_data(result, token_labels, collapsed_attentions, collapsed_labels,
                        save_path, metadata=None, raw_attentions=None,
                        grid_size=None, frames=None):
    """Save extracted attention data to .pt file.

    Args:
        result: extract_attention() output
        raw_attentions: list of (1, H, S, S) tensors (full, uncollapsed) — for heatmap projection
        grid_size: (grid_h, grid_w) — vision token spatial grid
        frames: list of PIL.Image or numpy array of video frames
    """
    data = {
        "attentions_collapsed": [a.half() for a in collapsed_attentions],
        "tokens_collapsed": collapsed_labels,
        "tokens_full": token_labels,
        "image_token_range": result["image_token_range"],
        "inputs_embeds_shape": list(result["inputs_embeds_shape"]),
        "predicted_token": result["predicted_token"],
    }
    if metadata:
        data["metadata"] = metadata
    if grid_size:
        data["grid_size"] = grid_size

    # Raw attentions for heatmap projection (half precision to save space)
    if raw_attentions is not None:
        data["attentions_raw"] = [a.half() for a in raw_attentions]

    # Precompute rollout + avg vision attention (small, always save)
    try:
        attn_list = raw_attentions if raw_attentions is not None else [a for a in result["attentions"]]
        img_range = result["image_token_range"]
        data["vision_attn_rollout"] = extract_answer_vision_attention(attn_list, img_range, method="rollout")
        data["vision_attn_avg"] = extract_answer_vision_attention(attn_list, img_range, method="avg")
    except Exception as e:
        print(f"  [WARN] Could not compute vision attention: {e}")

    # Video frames as numpy
    if frames is not None:
        if isinstance(frames, np.ndarray):
            data["frames"] = frames
        else:
            data["frames"] = np.array([np.array(f) for f in frames])

    torch.save(data, save_path)
    print(f"[SAVED] {save_path}")


def run_single_image(args, model, tokenizer, image_processor, model_name, conv_template):
    """Extract attention for a single image."""
    print(f"[INFO] Loading image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [t.to(dtype=torch.float16) for t in image_tensor]
    image_sizes = [image.size]

    input_ids = build_prompt(args.question, conv_template, model_name, tokenizer)

    print("[INFO] Extracting attention...")
    result = extract_attention(
        model, tokenizer, input_ids,
        image_tensor, image_sizes,
        modalities=["image"],
    )
    print(f"[INFO] Predicted: {result['predicted_token']}")

    token_labels = build_token_labels(
        tokenizer, result["input_ids"], result["inputs_embeds_shape"],
        result["image_token_range"], num_frames=None, model=model,
    )

    collapsed_attentions, collapsed_labels = collapse_vision_tokens(
        result["attentions"], token_labels, result["image_token_range"],
        num_frames=None, model=model,
    )

    grid_h, grid_w = get_vision_grid_size(model)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_attn.pt")

    save_attention_data(result, token_labels, collapsed_attentions, collapsed_labels,
                        save_path, metadata={
                            "question": args.question,
                            "source": args.image_path,
                            "type": "image",
                        },
                        raw_attentions=result["attentions"],
                        grid_size=(grid_h, grid_w),
                        frames=[image])

    return result


def run_single_video(args, model, tokenizer, image_processor, model_name, conv_template):
    """Extract attention for a single video."""
    print(f"[INFO] Loading video: {args.video_path}")
    video_data_args = argparse.Namespace(
        video_fps=args.video_fps,
        frames_upbound=args.frames_upbound,
        force_sample=args.force_sample,
    )
    video_frames, video_time, frame_time, num_frames = process_video_with_decord(
        args.video_path, video_data_args
    )
    print(f"[INFO] Extracted {num_frames} frames")

    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
    image_tensor = [image_tensor.to(dtype=torch.float16)]

    if isinstance(video_frames, np.ndarray):
        h, w = video_frames.shape[1], video_frames.shape[2]
        image_sizes = [(w, h)]
    else:
        image_sizes = [video_frames[0].size]

    input_ids = build_prompt(args.question, conv_template, model_name, tokenizer)

    print("[INFO] Extracting attention...")
    result = extract_attention(
        model, tokenizer, input_ids,
        image_tensor, image_sizes,
        modalities=["video"],
    )
    print(f"[INFO] Predicted: {result['predicted_token']}")

    token_labels = build_token_labels(
        tokenizer, result["input_ids"], result["inputs_embeds_shape"],
        result["image_token_range"], num_frames=num_frames, model=model,
    )

    collapsed_attentions, collapsed_labels = collapse_vision_tokens(
        result["attentions"], token_labels, result["image_token_range"],
        num_frames=num_frames, model=model,
    )

    grid_h, grid_w = get_vision_grid_size(model)
    mm_newline = getattr(model.config, "mm_newline_position", "one_token")

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_attn.pt")

    save_attention_data(result, token_labels, collapsed_attentions, collapsed_labels,
                        save_path, metadata={
                            "question": args.question,
                            "source": args.video_path,
                            "type": "video",
                            "num_frames": num_frames,
                            "tokens_per_frame": grid_h * grid_w,
                            "include_newline": mm_newline == "one_token",
                        },
                        raw_attentions=result["attentions"],
                        grid_size=(grid_h, grid_w),
                        frames=video_frames)

    return result


def run_batch(args, model, tokenizer, image_processor, model_name, conv_template):
    """Extract attention for a batch dataset (--task or --refined_dataset)."""
    if args.task:
        questions, dataset_dict = load_dataset_as_questions(
            task_name=args.task,
            video_folder=args.video_folder,
            image_folder=args.image_folder,
            limit=args.limit,
        )
        task_name = args.task
    else:
        questions, dataset_dict = load_dataset_as_questions(
            csv_path=args.refined_dataset,
            video_folder=args.video_folder,
            image_folder=args.image_folder,
            limit=args.limit,
        )
        task_name = os.path.splitext(os.path.basename(args.refined_dataset))[0]

    print(f"[INFO] Dataset: {task_name}, samples: {len(questions)}")

    data_loader = create_data_loader(
        questions, args.image_folder, args.batch_size, args.num_workers,
        tokenizer, image_processor, model.config, task_name, conv_template,
        video_folder=args.video_folder, video_fps=args.video_fps,
        frames_upbound=args.frames_upbound, force_sample=args.force_sample,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    grid_h, grid_w = get_vision_grid_size(model)
    tokens_per_frame = grid_h * grid_w
    mm_newline = getattr(model.config, "mm_newline_position", "one_token")
    has_newline = mm_newline == "one_token"

    results_summary = []

    for batch, line in tqdm(zip(data_loader, questions), total=len(questions), desc="Extracting"):
        if batch is None:
            continue

        input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality = batch
        question_id = line["q_id"]

        if "video" in line and line["video"] != "":
            sample_id = str(line["video"])
        else:
            sample_id = str(line.get("img_id", question_id))

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        try:
            result = extract_attention(
                model, tokenizer, input_ids,
                image_tensor, original_image_sizes,
                modalities=[effective_modality],
            )
        except Exception as e:
            print(f"\n[WARN] Sample {question_id} failed: {e}")
            continue

        # Estimate num_frames for video
        num_vision = result["image_token_range"][1] - result["image_token_range"][0]
        stride = tokens_per_frame + (1 if has_newline else 0)
        est_num_frames = max(1, num_vision // stride) if effective_modality == "video" else None

        token_labels = build_token_labels(
            tokenizer, result["input_ids"], result["inputs_embeds_shape"],
            result["image_token_range"], num_frames=est_num_frames, model=model,
        )

        collapsed_attentions, collapsed_labels = collapse_vision_tokens(
            result["attentions"], token_labels, result["image_token_range"],
            num_frames=est_num_frames, model=model,
        )

        safe_name = re.sub(r'[^\w\-.]', '_', os.path.basename(sample_id).split('.')[0])
        save_path = os.path.join(args.output_dir, f"{safe_name}_attn.pt")

        save_attention_data(result, token_labels, collapsed_attentions, collapsed_labels,
                            save_path, metadata={
                                "q_id": question_id,
                                "question": line.get("question", ""),
                                "answer": line.get("answer", ""),
                                "source": sample_id,
                                "type": effective_modality,
                                "num_frames": est_num_frames,
                            })

        results_summary.append({
            "q_id": question_id,
            "sample_id": sample_id,
            "predicted": result["predicted_token"],
            "saved": save_path,
        })

        del result, collapsed_attentions
        torch.cuda.empty_cache()

    # Summary
    if results_summary:
        summary_path = os.path.join(args.output_dir, f"{task_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Summary: {summary_path}")

    print(f"[DONE] {len(results_summary)} samples processed")


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention maps from LLaVA-NeXT for bertviz visualization."
    )

    # Model
    parser.add_argument("--model_args", type=str, default=None,
                        help='lmms_eval style. e.g., "pretrained=...,conv_template=qwen_1_5,device_map=auto,max_frames_num=8"')
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")

    # Single input
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--question", type=str, default="Describe this image in detail.")

    # Batch
    parser.add_argument('--task', type=str, default=None,
                        help=f"Task name. Available: {list_tasks()}")
    parser.add_argument('--refined_dataset', type=str, default=None)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)

    # Video
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)

    # Output
    parser.add_argument("--output_dir", type=str, default="output/attention")

    args = parser.parse_args()

    if not args.model_args and not args.model_path:
        parser.error("--model_args or --model-path required")
    if not any([args.image_path, args.video_path, args.task, args.refined_dataset]):
        parser.error("--image_path, --video_path, --task, or --refined_dataset required")

    # Load model
    if args.model_args:
        model_args_dict = parse_model_args(args.model_args)
        tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
        args.conv_mode = conv_template

        # Apply video config from model_args
        if "max_frames_num" in model_args_dict and args.frames_upbound == 32:
            args.frames_upbound = int(model_args_dict["max_frames_num"])
        if "force_sample" in model_args_dict:
            args.force_sample = bool(model_args_dict["force_sample"])
        if "video_fps" in model_args_dict and args.video_fps == 1:
            args.video_fps = int(model_args_dict["video_fps"])
    else:
        tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_legacy(
            args.model_path, args.model_base, args.conv_mode
        )
    model.eval()

    print(f"[INFO] Model: {model_name} (frames_upbound={args.frames_upbound})")

    if args.image_path:
        run_single_image(args, model, tokenizer, image_processor, model_name, conv_template)
    elif args.video_path:
        run_single_video(args, model, tokenizer, image_processor, model_name, conv_template)
    elif args.task or args.refined_dataset:
        run_batch(args, model, tokenizer, image_processor, model_name, conv_template)


if __name__ == "__main__":
    main()
