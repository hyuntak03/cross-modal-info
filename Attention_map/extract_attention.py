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

try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.utils import process_video_with_decord
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from core.model_loader import parse_model_args, load_model_from_args, load_model_legacy
    from core.data_pipeline import CustomDataset, collate_fn, create_data_loader
    _LLAVA_AVAILABLE = True
except (ImportError, Exception):
    _LLAVA_AVAILABLE = False

# Direct import of dataset_loader to avoid core/__init__.py's heavy llava imports
import importlib.util
def _import_module_direct(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dataset_loader = _import_module_direct(
    "core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py")
)
load_dataset_as_questions = _dataset_loader.load_dataset_as_questions
list_tasks = _dataset_loader.list_tasks

# parse_model_args: use core.model_loader version if available, else local fallback
if not _LLAVA_AVAILABLE:
    def parse_model_args(args_string):
        """lmms_eval style model_args parsing (standalone fallback)."""
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

from Attention_map.attention_utils import (
    extract_attention,
    extract_attention_fast,
    build_token_labels,
    collapse_vision_tokens,
    build_prompt,
    get_vision_grid_size,
    get_tokens_per_frame,
    extract_answer_vision_attention,
    split_vision_attention_by_frames,
    compute_cross_frame_analysis,
    # Qwen3-VL
    extract_attention_qwen3vl_fast,
    build_token_labels_qwen3vl,
    get_qwen3vl_vision_token_id,
    get_qwen3vl_grid_size,
)


def save_attention_data(result, token_labels, collapsed_attentions, collapsed_labels,
                        save_path, metadata=None,
                        grid_size=None, frames=None):
    """Save extracted attention data to .pt file.

    In fast mode, result already contains precomputed cross_frame/vision_attn.
    In slow mode, computes from result["attentions"] (collapsed).
    Raw attentions are never saved (~100KB instead of ~4GB per sample).
    """
    data = {
        "attentions_collapsed": [a.bfloat16() for a in collapsed_attentions],
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

    # Vision attention (precomputed in fast mode, or compute from raw attentions)
    if "vision_attn_rollout" in result:
        data["vision_attn_rollout"] = result["vision_attn_rollout"]
        data["vision_attn_avg"] = result["vision_attn_avg"]
    elif "attentions" in result:
        # Slow mode fallback: compute from full attentions
        try:
            img_range = result["image_token_range"]
            data["vision_attn_rollout"] = extract_answer_vision_attention(
                result["attentions"], img_range, method="rollout")
            data["vision_attn_avg"] = extract_answer_vision_attention(
                result["attentions"], img_range, method="avg")
        except Exception as e:
            print(f"  [WARN] Could not compute vision attention: {e}")

    # Cross-frame (precomputed in fast mode)
    if "cross_frame" in result and result["cross_frame"] is not None:
        data["cross_frame"] = result["cross_frame"]

    # Video frames (small: ~3.5MB for 8×384×384)
    if frames is not None:
        if isinstance(frames, np.ndarray):
            data["frames"] = frames
        else:
            data["frames"] = np.array([np.array(f) for f in frames])

    torch.save(data, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[SAVED] {save_path} ({size_mb:.1f}MB)")


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

    grid_h, grid_w = get_vision_grid_size(model)
    tpf, inter = get_tokens_per_frame(model)

    print("[INFO] Extracting attention (fast)...")
    result = extract_attention_fast(
        model, tokenizer, input_ids,
        image_tensor, image_sizes,
        modalities=["video"],
        num_frames=num_frames, tokens_per_frame=tpf, inter_frame_tokens=inter,
        layer_stride=args.layer_stride,
    )
    print(f"[INFO] Predicted: {result['predicted_token']}")

    token_labels = build_token_labels(
        tokenizer, result["input_ids"], result["inputs_embeds_shape"],
        result["image_token_range"], num_frames=num_frames, model=model,
    )

    # Build collapsed labels (same logic as collapse_vision_tokens but labels only)
    img_s, img_e = result["image_token_range"]
    collapsed_labels = []
    for i in range(img_s):
        collapsed_labels.append(token_labels[i])
    if num_frames and num_frames > 1:
        stride_c = tpf + inter
        for f in range(num_frames):
            gs = img_s + f * stride_c
            if gs < img_e:
                collapsed_labels.append(f"[F{f}]")
    else:
        collapsed_labels.append("[IMG]")
    for i in range(img_e, len(token_labels)):
        collapsed_labels.append(token_labels[i])

    # Fill in question_tokens from token_labels
    if result.get("cross_frame") is not None:
        result["cross_frame"]["question_tokens"] = token_labels[img_e:]

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_attn.pt")

    save_attention_data(result, token_labels, result["attentions"], collapsed_labels,
                        save_path, metadata={
                            "question": args.question,
                            "source": args.video_path,
                            "type": "video",
                            "num_frames": num_frames,
                            "tokens_per_frame": tpf,
                            "inter_frame_tokens": inter,
                        },
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
    tokens_per_frame, inter_frame_tokens = get_tokens_per_frame(model)

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

        # Estimate num_frames
        # For fast mode, we need num_frames before extraction
        # Do a quick forward to get inputs_embeds_shape, then estimate
        est_num_frames = None
        if effective_modality == "video":
            # Estimate from model config and data
            # We'll get the actual count after extraction
            est_num_frames = args.frames_upbound  # upper bound, refined after

        import time
        t0 = time.time()

        try:
            result = extract_attention_fast(
                model, tokenizer, input_ids,
                image_tensor, original_image_sizes,
                modalities=[effective_modality],
                num_frames=est_num_frames,
                tokens_per_frame=tokens_per_frame,
                inter_frame_tokens=inter_frame_tokens,
                layer_stride=args.layer_stride,
            )
        except Exception as e:
            import traceback
            print(f"\n[WARN] Sample {question_id} failed: {e}")
            traceback.print_exc()
            continue

        t_extract = time.time() - t0

        # Refine num_frames from actual vision tokens
        num_vision = result["image_token_range"][1] - result["image_token_range"][0]
        stride = tokens_per_frame + inter_frame_tokens
        est_num_frames = max(1, (num_vision + stride - 1) // stride) if effective_modality == "video" else None

        token_labels = build_token_labels(
            tokenizer, result["input_ids"], result["inputs_embeds_shape"],
            result["image_token_range"], num_frames=est_num_frames, model=model,
        )

        # Build collapsed labels
        img_s, img_e = result["image_token_range"]
        collapsed_labels = []
        for i in range(img_s):
            collapsed_labels.append(token_labels[i])
        if est_num_frames and est_num_frames > 1:
            stride_c = tokens_per_frame + inter_frame_tokens
            for f in range(est_num_frames):
                gs = img_s + f * stride_c
                if gs < img_e:
                    collapsed_labels.append(f"[F{f}]")
        else:
            collapsed_labels.append("[IMG]")
        for i in range(img_e, len(token_labels)):
            collapsed_labels.append(token_labels[i])

        safe_name = re.sub(r'[^\w\-.]', '_', os.path.basename(sample_id).split('.')[0])
        save_path = os.path.join(args.output_dir, f"{safe_name}_attn.pt")

        # Load original frames for heatmap overlay
        batch_frames = None
        if effective_modality == "video":
            video_path = sample_id
            if not os.path.isabs(video_path) and args.video_folder:
                video_path = os.path.join(args.video_folder, video_path)
            if os.path.exists(video_path):
                try:
                    vda = argparse.Namespace(
                        video_fps=args.video_fps,
                        frames_upbound=args.frames_upbound,
                        force_sample=args.force_sample,
                    )
                    batch_frames, _, _, _ = process_video_with_decord(video_path, vda)
                except Exception:
                    pass

        # Fill in question_tokens from token_labels
        if result.get("cross_frame") is not None:
            img_s, img_e = result["image_token_range"]
            result["cross_frame"]["question_tokens"] = token_labels[img_e:]

        t1 = time.time()
        save_attention_data(result, token_labels, result["attentions"], collapsed_labels,
                            save_path, metadata={
                                "q_id": question_id,
                                "question": line.get("question", ""),
                                "answer": line.get("answer", ""),
                                "source": sample_id,
                                "type": effective_modality,
                                "num_frames": est_num_frames,
                                "tokens_per_frame": tokens_per_frame,
                                "inter_frame_tokens": inter_frame_tokens,
                            },
                            grid_size=(grid_h, grid_w),
                            frames=batch_frames)
        t_save = time.time() - t1
        t_total = time.time() - t0
        print(f"  [{question_id}] extract={t_extract:.1f}s save={t_save:.1f}s total={t_total:.1f}s pred={result['predicted_token']}")

        results_summary.append({
            "q_id": question_id,
            "sample_id": sample_id,
            "predicted": result["predicted_token"],
            "saved": save_path,
        })

        del result, batch_frames
        torch.cuda.empty_cache()

    # Summary
    if results_summary:
        summary_path = os.path.join(args.output_dir, f"{task_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Summary: {summary_path}")

    print(f"[DONE] {len(results_summary)} samples processed")


# ============================================================
# Qwen3-VL Support
# ============================================================

def detect_model_type(pretrained_path):
    """Detect model type from config.json or model name heuristic.

    Supports: local paths, HF cache layout (models--Org--Name/), HF repo names.
    """
    # 1. Try local config.json directly
    candidates = [os.path.join(pretrained_path, "config.json")]

    # 2. HF cache layout: models--Org--Name/snapshots/hash/
    snap_dir = os.path.join(pretrained_path, "snapshots")
    if os.path.isdir(snap_dir):
        for h in os.listdir(snap_dir):
            candidates.append(os.path.join(snap_dir, h, "config.json"))

    # 3. HF_HOME cache: convert "Org/Model" → "$HF_HOME/models--Org--Model/snapshots/*/config.json"
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    if "/" in pretrained_path and not os.path.isdir(pretrained_path):
        cache_dir_name = "models--" + pretrained_path.replace("/", "--")
        cache_snap = os.path.join(hf_home, cache_dir_name, "snapshots")
        if os.path.isdir(cache_snap):
            for h in os.listdir(cache_snap):
                candidates.append(os.path.join(cache_snap, h, "config.json"))

    for config_path in candidates:
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model_type = config.get("model_type", "")
                if "qwen3_vl" in model_type:
                    return "qwen3_vl"
                if model_type:  # found a valid config, trust it
                    return "llava"
            except (json.JSONDecodeError, IOError):
                continue

    # 4. Fallback: name-based heuristic
    name_lower = pretrained_path.lower()
    if "qwen3-vl" in name_lower or "qwen3_vl" in name_lower:
        return "qwen3_vl"

    return "llava"


def _resolve_pretrained_path(pretrained_path):
    """Resolve HF cache layout or HF repo name to actual snapshot path.

    Handles:
      - Direct path with config.json → return as-is
      - models--Org--Name/snapshots/hash/ layout → return snapshot path
      - HF repo name "Org/Model" → look up in $HF_HOME cache
      - Otherwise return as-is (let from_pretrained handle it)
    """
    # Direct path
    if os.path.exists(os.path.join(pretrained_path, "config.json")):
        return pretrained_path

    # Local cache layout: models--Org--Name/snapshots/hash/
    snap_dir = os.path.join(pretrained_path, "snapshots")
    if os.path.isdir(snap_dir):
        hashes = os.listdir(snap_dir)
        if hashes:
            return os.path.join(snap_dir, hashes[0])

    # HF repo name → $HF_HOME cache
    if "/" in pretrained_path and not os.path.isdir(pretrained_path):
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_dir_name = "models--" + pretrained_path.replace("/", "--")
        cache_snap = os.path.join(hf_home, cache_dir_name, "snapshots")
        if os.path.isdir(cache_snap):
            hashes = os.listdir(cache_snap)
            if hashes:
                resolved = os.path.join(cache_snap, hashes[0])
                print(f"  [resolve] {pretrained_path} → {resolved}")
                return resolved

    return pretrained_path


def load_qwen3vl_model(model_args_dict):
    """Load Qwen3-VL model with eager attention for attention extraction.

    Supports model_args keys:
      - pretrained: model path or HF repo
      - device_map: device placement (default: auto)
      - min_pixels, max_pixels: vision resolution bounds (passed to AutoProcessor)
      - attn_implementation: ignored, always forced to 'eager'
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    pretrained = model_args_dict.get("pretrained", "")
    pretrained = _resolve_pretrained_path(pretrained)
    device_map = model_args_dict.get("device_map", "auto")

    print(f"[MODEL] Loading Qwen3-VL: {pretrained}")
    print(f"  attn_implementation=eager (required for attention extraction)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pretrained,
        dtype="auto",
        device_map=device_map,
        attn_implementation="eager",
    )

    # Pass min_pixels/max_pixels to processor for vision resolution control
    processor_kwargs = {}
    if "min_pixels" in model_args_dict:
        processor_kwargs["min_pixels"] = int(model_args_dict["min_pixels"])
    if "max_pixels" in model_args_dict:
        processor_kwargs["max_pixels"] = int(model_args_dict["max_pixels"])
    processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)

    if processor_kwargs:
        print(f"  processor: {processor_kwargs}")

    model.eval()

    model_name = os.path.basename(pretrained.rstrip("/"))
    return model, processor, model_name


def load_video_frames_qwen3vl(video_path, num_frames=8):
    """Load video frames as PIL Images for Qwen3-VL processor."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames], len(indices)


def run_single_video_qwen3vl(args, model, processor, model_name):
    """Extract attention for a single video (Qwen3-VL)."""
    print(f"[INFO] Loading video: {args.video_path}")
    frames, actual_num_frames = load_video_frames_qwen3vl(
        args.video_path, args.frames_upbound
    )
    print(f"[INFO] Loaded {actual_num_frames} frames")

    # Qwen3-VL temporal merge: temporal_patch_size=2 → temporal_slots = ceil(frames/2)
    temporal_patch_size = getattr(model.config.vision_config, 'temporal_patch_size', 2)
    num_temporal_slots = (actual_num_frames + temporal_patch_size - 1) // temporal_patch_size

    # Build input
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[frames], return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    vision_pad_id = get_qwen3vl_vision_token_id(processor, "video")

    print("[INFO] Extracting attention (Qwen3-VL)...")
    result = extract_attention_qwen3vl_fast(
        model, processor, inputs, vision_pad_id,
        num_frames=num_temporal_slots,
        layer_stride=args.layer_stride,
    )
    print(f"[INFO] Predicted: {result['predicted_token']}")

    img_start, img_end = result["image_token_range"]
    num_vision = img_end - img_start
    tokens_per_frame = num_vision // max(num_temporal_slots, 1)

    token_labels = build_token_labels_qwen3vl(
        processor, result["input_ids"][0], result["image_token_range"],
        num_frames=num_temporal_slots, tokens_per_frame=tokens_per_frame,
    )

    # Collapsed labels
    collapsed_labels = []
    for i in range(img_start):
        collapsed_labels.append(token_labels[i])
    if num_temporal_slots > 1:
        for f in range(num_temporal_slots):
            gs = img_start + f * tokens_per_frame
            if gs < img_end:
                collapsed_labels.append(f"[F{f}]")
    else:
        collapsed_labels.append("[IMG]")
    for i in range(img_end, len(token_labels)):
        collapsed_labels.append(token_labels[i])

    # Fill question_tokens
    if result.get("cross_frame") is not None:
        result["cross_frame"]["question_tokens"] = token_labels[img_end:]

    grid_h, grid_w = get_qwen3vl_grid_size(tokens_per_frame)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_attn.pt")

    # Convert PIL frames to numpy for saving
    frames_np = np.array([np.array(f) for f in frames])

    save_attention_data(result, token_labels, result["attentions"], collapsed_labels,
                        save_path, metadata={
                            "question": args.question,
                            "source": args.video_path,
                            "type": "video",
                            "num_frames": num_temporal_slots,
                            "tokens_per_frame": tokens_per_frame,
                            "inter_frame_tokens": 0,
                            "actual_video_frames": actual_num_frames,
                            "temporal_patch_size": temporal_patch_size,
                        },
                        grid_size=(grid_h, grid_w),
                        frames=frames_np)

    return result


def run_single_image_qwen3vl(args, model, processor, model_name):
    """Extract attention for a single image (Qwen3-VL)."""
    print(f"[INFO] Loading image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    vision_pad_id = get_qwen3vl_vision_token_id(processor, "image")

    print("[INFO] Extracting attention (Qwen3-VL)...")
    result = extract_attention_qwen3vl_fast(
        model, processor, inputs, vision_pad_id,
        num_frames=None,
        layer_stride=args.layer_stride,
    )
    print(f"[INFO] Predicted: {result['predicted_token']}")

    img_start, img_end = result["image_token_range"]
    num_vision = img_end - img_start

    token_labels = build_token_labels_qwen3vl(
        processor, result["input_ids"][0], result["image_token_range"],
    )

    collapsed_labels = []
    for i in range(img_start):
        collapsed_labels.append(token_labels[i])
    collapsed_labels.append("[IMG]")
    for i in range(img_end, len(token_labels)):
        collapsed_labels.append(token_labels[i])

    grid_h, grid_w = get_qwen3vl_grid_size(num_vision)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_attn.pt")

    save_attention_data(result, token_labels, result["attentions"], collapsed_labels,
                        save_path, metadata={
                            "question": args.question,
                            "source": args.image_path,
                            "type": "image",
                        },
                        grid_size=(grid_h, grid_w),
                        frames=[image])

    return result


def run_batch_qwen3vl(args, model, processor, model_name):
    """Extract attention for a batch dataset (Qwen3-VL)."""
    # load_dataset_as_questions already imported at top level (direct module load)
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

    os.makedirs(args.output_dir, exist_ok=True)

    vision_pad_id_video = get_qwen3vl_vision_token_id(processor, "video")
    vision_pad_id_image = get_qwen3vl_vision_token_id(processor, "image")

    temporal_patch_size = getattr(model.config.vision_config, 'temporal_patch_size', 2)

    results_summary = []

    for line in tqdm(questions, desc="Extracting (Qwen3-VL)"):
        question_id = line["q_id"]
        question_text = line.get("question", "")

        # Determine media type and load
        has_video = "video" in line and line["video"] != ""
        has_image = "img_id" in line and line["img_id"] != ""

        if has_video:
            sample_id = str(line["video"])
            video_path = sample_id
            if not os.path.isabs(video_path) and args.video_folder:
                video_path = os.path.join(args.video_folder, video_path)

            if not os.path.exists(video_path):
                print(f"  [WARN] Video not found: {video_path}, skipping")
                continue

            try:
                frames, actual_num_frames = load_video_frames_qwen3vl(
                    video_path, args.frames_upbound
                )
            except Exception as e:
                print(f"  [WARN] Failed to load video {video_path}: {e}")
                continue

            num_temporal_slots = (actual_num_frames + temporal_patch_size - 1) // temporal_patch_size
            media_type = "video"
            vision_pad_id = vision_pad_id_video

            messages = [{"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question_text},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=[frames], return_tensors="pt")
        elif has_image:
            sample_id = str(line["img_id"])
            image_path = sample_id
            if not os.path.isabs(image_path) and args.image_folder:
                image_path = os.path.join(args.image_folder, image_path)

            if not os.path.exists(image_path):
                print(f"  [WARN] Image not found: {image_path}, skipping")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                frames = [image]
            except Exception as e:
                print(f"  [WARN] Failed to load image {image_path}: {e}")
                continue

            num_temporal_slots = None
            media_type = "image"
            vision_pad_id = vision_pad_id_image

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        else:
            print(f"  [WARN] No media for {question_id}, skipping")
            continue

        device = model.device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        import time
        t0 = time.time()

        try:
            result = extract_attention_qwen3vl_fast(
                model, processor, inputs, vision_pad_id,
                num_frames=num_temporal_slots,
                layer_stride=args.layer_stride,
            )
        except Exception as e:
            import traceback
            print(f"\n[WARN] Sample {question_id} failed: {e}")
            traceback.print_exc()
            continue

        t_extract = time.time() - t0

        img_start, img_end = result["image_token_range"]
        num_vision = img_end - img_start

        if media_type == "video" and num_temporal_slots and num_temporal_slots > 1:
            tokens_per_frame = num_vision // num_temporal_slots
        else:
            tokens_per_frame = num_vision
            num_temporal_slots = None

        token_labels = build_token_labels_qwen3vl(
            processor, result["input_ids"][0], result["image_token_range"],
            num_frames=num_temporal_slots, tokens_per_frame=tokens_per_frame,
        )

        # Collapsed labels
        collapsed_labels = []
        for i in range(img_start):
            collapsed_labels.append(token_labels[i])
        if num_temporal_slots and num_temporal_slots > 1:
            for f in range(num_temporal_slots):
                gs = img_start + f * tokens_per_frame
                if gs < img_end:
                    collapsed_labels.append(f"[F{f}]")
        else:
            collapsed_labels.append("[IMG]")
        for i in range(img_end, len(token_labels)):
            collapsed_labels.append(token_labels[i])

        if result.get("cross_frame") is not None:
            result["cross_frame"]["question_tokens"] = token_labels[img_end:]

        grid_h, grid_w = get_qwen3vl_grid_size(tokens_per_frame)

        safe_name = re.sub(r'[^\w\-.]', '_', os.path.basename(sample_id).split('.')[0])
        save_path = os.path.join(args.output_dir, f"{safe_name}_attn.pt")

        # Frames for heatmap
        batch_frames = np.array([np.array(f) for f in frames])

        t1 = time.time()
        save_attention_data(result, token_labels, result["attentions"], collapsed_labels,
                            save_path, metadata={
                                "q_id": question_id,
                                "question": question_text,
                                "answer": line.get("answer", ""),
                                "source": sample_id,
                                "type": media_type,
                                "num_frames": num_temporal_slots,
                                "tokens_per_frame": tokens_per_frame,
                                "inter_frame_tokens": 0,
                            },
                            grid_size=(grid_h, grid_w),
                            frames=batch_frames)
        t_save = time.time() - t1
        t_total = time.time() - t0
        print(f"  [{question_id}] extract={t_extract:.1f}s save={t_save:.1f}s "
              f"total={t_total:.1f}s pred={result['predicted_token']}")

        results_summary.append({
            "q_id": question_id,
            "sample_id": sample_id,
            "predicted": result["predicted_token"],
            "saved": save_path,
        })

        del result, inputs, batch_frames
        torch.cuda.empty_cache()

    if results_summary:
        summary_path = os.path.join(args.output_dir, f"{task_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Summary: {summary_path}")

    print(f"[DONE] {len(results_summary)} samples processed")


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention maps from VLMs (LLaVA-NeXT, Qwen3-VL)."
    )

    # Model
    parser.add_argument("--model_args", type=str, default=None,
                        help='lmms_eval style. e.g., "pretrained=...,conv_template=qwen_1_5,device_map=auto,max_frames_num=8"')
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "llava", "qwen3_vl"],
                        help="Model type (auto: detect from config.json)")
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
    parser.add_argument("--layer_stride", type=int, default=1,
                        help="Compute attention every N layers (default=1, all layers).")

    args = parser.parse_args()

    if not args.model_args and not args.model_path:
        parser.error("--model_args or --model-path required")
    if not any([args.image_path, args.video_path, args.task, args.refined_dataset]):
        parser.error("--image_path, --video_path, --task, or --refined_dataset required")

    # Detect model type
    model_type = args.model_type
    if model_type == "auto":
        if args.model_args:
            model_args_dict = parse_model_args(args.model_args)
            pretrained = model_args_dict.get("pretrained", "")
        else:
            pretrained = args.model_path or ""
        model_type = detect_model_type(pretrained)
        print(f"[INFO] Auto-detected model type: {model_type}")

    if model_type == "qwen3_vl":
        # ---- Qwen3-VL path ----
        if args.model_args:
            model_args_dict = parse_model_args(args.model_args)
        else:
            model_args_dict = {"pretrained": args.model_path, "device_map": "auto"}

        # Support both max_frames_num (LLaVA convention) and max_num_frames (Qwen convention)
        if args.frames_upbound == 32:
            for key in ("max_frames_num", "max_num_frames"):
                if key in model_args_dict:
                    args.frames_upbound = int(model_args_dict[key])
                    break

        model, processor, model_name = load_qwen3vl_model(model_args_dict)
        print(f"[INFO] Model: {model_name} (frames_upbound={args.frames_upbound})")

        if args.image_path:
            run_single_image_qwen3vl(args, model, processor, model_name)
        elif args.video_path:
            run_single_video_qwen3vl(args, model, processor, model_name)
        elif args.task or args.refined_dataset:
            run_batch_qwen3vl(args, model, processor, model_name)
    else:
        # ---- LLaVA path (original) ----
        if args.model_args:
            model_args_dict = parse_model_args(args.model_args)
            tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
            args.conv_mode = conv_template

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
