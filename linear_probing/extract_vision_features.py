"""
통합 Vision Feature 추출 스크립트.

3단계 feature를 한번에 추출:
  1. Vision Encoder (pre-projector): raw SigLIP/CLIP 출력
  2. After Projector: projector + spatial pooling 후 (LLM 입력 직전)
  3. LLM Per-Layer: LLM 내부 각 layer에서의 vision token hidden states

Features 저장 구조:
  {feat_base_dir}/vision_encoder/{task}/features.npy    (N, ve_dim)
  {feat_base_dir}/after_projector/{task}/features.npy   (N, ap_dim)
  {feat_base_dir}/vision_token/{task}/features_layer_*.npy  (N, vt_dim)

지원 모델:
  - LLaVA 계열: 3단계 모두 추출
  - Qwen3-VL 계열: LLM per-layer만 추출

Usage (LLaVA):
    python linear_probing/extract_vision_features.py \
        --model_args "pretrained=...,device_map=auto" \
        --task vlm_direction_testbed_R2R_shape_color \
        --feat_base_dir linear_probe_features/llava-video-7b \
        --frames_upbound 8 --force_sample
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

torch.set_grad_enabled(False)

# TF32 활성화 (A100/H100에서 bf16 유사 속도, 정밀도 무시할만큼 차이)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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
    config_path = os.path.join(pretrained_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if "qwen3_vl" in config.get("model_type", ""):
            return "qwen3_vl"
    return "llava"


def parse_model_args(args_string):
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
#  공통 유틸
# ============================================================

def build_label_set(questions):
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
    answer = str(line["answer"]).strip()
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        candidates_raw = line.get("candidates", [])
        if isinstance(candidates_raw, str):
            candidates_raw = ast.literal_eval(candidates_raw)
        idx = ord(answer.upper()) - ord('A')
        if idx < len(candidates_raw):
            answer = str(candidates_raw[idx]).strip()
    return answer


def _apply_spatial_pool(model, features, stride, pool_mode):
    """LLaVA spatial pooling 복제. model.get_2dPool이 있으면 사용, 없으면 직접 수행."""
    if stride <= 1:
        return features
    try:
        return model.get_2dPool(features, stride)
    except (AttributeError, TypeError):
        B, N, D = features.shape
        h = w = int(math.sqrt(N))
        x = features.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        if pool_mode == "bilinear":
            oh, ow = math.ceil(h / stride), math.ceil(w / stride)
            x = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False)
        else:
            x = F.avg_pool2d(x, stride)
        return x.permute(0, 2, 3, 1).reshape(B, -1, D)


# ============================================================
#  Streaming Feature Writer (mmap 기반, RAM 절약 + I/O 분산)
# ============================================================

class FeatureWriter:
    """numpy mmap으로 feature를 추출 즉시 disk에 streaming write.

    RAM 사용량: 상수 (sample 수에 무관)
    I/O: 추출 시간에 걸쳐 분산 (종료 시 burst 없음)
    """

    def __init__(self, feat_base_dir, task, num_samples):
        self.feat_base_dir = feat_base_dir
        self.task = task
        self.num_samples = num_samples
        self._mmaps = {}   # (stage, name) → mmap array
        self._dirs = {}    # stage → directory path
        self._idx = 0

    def _get_dir(self, stage):
        if stage not in self._dirs:
            d = os.path.join(self.feat_base_dir, stage, self.task)
            os.makedirs(d, exist_ok=True)
            self._dirs[stage] = d
        return self._dirs[stage]

    def _get_mmap(self, stage, filename, feat_dim):
        key = (stage, filename)
        if key not in self._mmaps:
            path = os.path.join(self._get_dir(stage), filename)
            self._mmaps[key] = np.lib.format.open_memmap(
                path, mode='w+', dtype=np.float16,
                shape=(self.num_samples, feat_dim))
        return self._mmaps[key]

    def write_layer(self, layer_idx, feat_tensor):
        """vision_token per-layer feature 1개 샘플 기록."""
        arr = feat_tensor.numpy()
        mmap = self._get_mmap("vision_token", f"features_layer_{layer_idx}.npy", arr.shape[0])
        mmap[self._idx] = arr

    def write_single(self, stage, feat_tensor):
        """vision_encoder 또는 after_projector feature 1개 샘플 기록."""
        arr = feat_tensor.numpy()
        mmap = self._get_mmap(stage, "features.npy", arr.shape[0])
        mmap[self._idx] = arr

    def advance(self):
        self._idx += 1

    @property
    def actual_samples(self):
        return self._idx

    def finalize(self, labels, qids, meta_common, extra_meta, num_layers):
        """mmap flush + labels/meta 저장. 샘플 수가 예상보다 적으면 truncate."""
        actual = self._idx
        needs_truncate = actual < self.num_samples

        # mmap flush + truncate if needed
        items = list(self._mmaps.items())
        desc = f"Saving {len(items)} feature files" + (" (truncating)" if needs_truncate else "")
        for (stage, filename), mmap in tqdm(items, desc=desc):
            mmap.flush()
            if needs_truncate:
                path = os.path.join(self._get_dir(stage), filename)
                truncated = np.array(mmap[:actual])
                del mmap
                np.save(path, truncated)
                del truncated
            else:
                del mmap
        self._mmaps.clear()

        labels_array = np.array(labels, dtype=np.int64)
        qids_array = np.array(qids)

        # vision_token meta
        vt_dir = self._get_dir("vision_token")
        np.save(os.path.join(vt_dir, "labels.npy"), labels_array)
        np.save(os.path.join(vt_dir, "qids.npy"), qids_array)
        vt_meta = {**meta_common, "num_layers": num_layers, "num_samples": actual, **extra_meta}
        np.save(os.path.join(vt_dir, "meta.npy"), vt_meta)
        print(f"[SAVED] vision_token: {num_layers} layers, {actual} samples → {vt_dir}")

        # vision_encoder meta
        if "vision_encoder" in self._dirs:
            ve_dir = self._dirs["vision_encoder"]
            np.save(os.path.join(ve_dir, "labels.npy"), labels_array)
            ve_meta = {**meta_common, "num_samples": actual, **extra_meta}
            np.save(os.path.join(ve_dir, "meta.npy"), ve_meta)
            print(f"[SAVED] vision_encoder → {ve_dir}")

        # after_projector meta
        if "after_projector" in self._dirs:
            ap_dir = self._dirs["after_projector"]
            np.save(os.path.join(ap_dir, "labels.npy"), labels_array)
            ap_meta = {**meta_common, "num_samples": actual, **extra_meta}
            np.save(os.path.join(ap_dir, "meta.npy"), ap_meta)
            print(f"[SAVED] after_projector → {ap_dir}")

        # after_gate meta (channel_gate 모델만)
        if "after_gate" in self._dirs:
            ag_dir = self._dirs["after_gate"]
            np.save(os.path.join(ag_dir, "labels.npy"), labels_array)
            ag_meta = {**meta_common, "num_samples": actual, **extra_meta}
            np.save(os.path.join(ag_dir, "meta.npy"), ag_meta)
            print(f"[SAVED] after_gate → {ag_dir}")

        print(f"[DONE] {actual} samples saved to {self.feat_base_dir}/*/{self.task}")


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
            frame_start = offset + f * tokens_per_frame_grid
            indices = []
            for row in range(grid_h):
                row_start = frame_start + row * (grid_h + 1)
                indices.extend(range(row_start, row_start + grid_h))
            frame_ranges.append(indices)
        tokens_per_frame = grid_h * grid_h
    else:
        for f in range(num_frames):
            start = offset + f * tokens_per_frame
            end = start + tokens_per_frame
            frame_ranges.append(list(range(start, end)))

    return frame_ranges, total_vis, tokens_per_frame, num_frames


def extract_features_llava(args):
    """LLaVA 계열 모델의 3단계 feature 추출."""
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

    # Vision encoder 정보
    vision_tower = model.get_vision_tower()
    num_patches_per_side = vision_tower.num_patches_per_side
    vision_hidden_dim = vision_tower.config.hidden_size
    stride = getattr(model.config, "mm_spatial_pool_stride", 2)
    pool_mode = getattr(model.config, "mm_spatial_pool_mode", "bilinear")

    if pool_mode == "bilinear":
        pooled_per_side = math.ceil(num_patches_per_side / stride)
    else:
        pooled_per_side = num_patches_per_side // stride

    tokens_per_frame_pre = num_patches_per_side * num_patches_per_side
    tokens_per_frame_post = pooled_per_side * pooled_per_side

    print(f"[INFO] Vision encoder: {tokens_per_frame_pre} tokens/frame, dim={vision_hidden_dim}")
    print(f"[INFO] After projector+pool: {tokens_per_frame_post} tokens/frame, dim={hidden_dim}")

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

    # 유효 샘플 수 사전 계산 (mmap pre-allocation용)
    num_valid = sum(1 for q in questions if resolve_answer(q) in answer_to_idx)
    print(f"[INFO] Valid samples: {num_valid} / {len(questions)}")

    # Streaming writer: feature를 추출 즉시 disk에 기록
    writer = FeatureWriter(args.feat_base_dir, args.task, num_valid)
    all_labels = []
    all_qids = []
    actual_num_frames = None

    # Vision tower + projector + channel_gate hook으로 generate 중 캡처
    ve_cache = {}
    ap_cache = {}
    gate_cache = {}

    def _vt_hook(module, input, output):
        ve_cache['out'] = output.detach()

    def _proj_hook(module, input, output):
        ap_cache['out'] = output.detach()

    def _gate_hook(module, input, output):
        # channel_gate.forward returns (gated_features, aux_dict)
        gate_cache['out'] = output[0].detach()

    h_vt = model.get_model().get_vision_tower().register_forward_hook(_vt_hook)
    h_proj = model.get_model().mm_projector.register_forward_hook(_proj_hook)

    # channel_gate가 있으면 hook 등록 (channel_gate 모델만)
    has_gate = hasattr(model.get_model(), 'channel_gate') and model.get_model().channel_gate is not None
    h_gate = None
    if has_gate:
        h_gate = model.get_model().channel_gate.register_forward_hook(_gate_hook)
        print("[INFO] Channel gate detected — after_gate features will be extracted")

    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions), desc="Extracting features"
    ):
        question_id = line["q_id"]
        answer = resolve_answer(line)
        if answer not in answer_to_idx:
            continue
        label_idx = answer_to_idx[answer]

        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        frame_ranges, total_vis, tpf, num_frames = compute_frame_boundaries(
            model, model_name, input_ids, image_tensor, original_image_sizes, modality
        )
        actual_num_frames = num_frames

        all_vision_indices = []
        for fr in frame_ranges:
            all_vision_indices.extend(fr)

        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        # === 단일 forward: generate가 vision tower + projector + gate + LLM 전부 실행 ===
        ve_cache.clear()
        ap_cache.clear()
        gate_cache.clear()

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

        # Stage 3: LLM per-layer — GPU에서 stack+pool 후 single CPU transfer
        # pool_spatial=True: (T, N_post, D) → mean(N_post) → (T, D)
        prefill_hidden = output.hidden_states  # tuple of (B, S, D) × (num_layers+1)
        # vision position만 뽑아서 stack → (L, T*N_post, D)
        vidx = torch.tensor(all_vision_indices, device=prefill_hidden[0].device)
        layer_stack = torch.stack(
            [prefill_hidden[l][0].index_select(0, vidx) for l in range(num_layers)],
            dim=0,
        )  # (L, T*N_post, D)
        if args.pool_spatial:
            L_, _, D_ = layer_stack.shape
            layer_stack = layer_stack.reshape(L_, num_frames, -1, D_).mean(dim=2)  # (L, T, D)
        layer_stack_cpu = layer_stack.reshape(num_layers, -1).cpu().to(torch.float16)  # (L, flat)
        for layer_idx in range(num_layers):
            writer.write_layer(layer_idx, layer_stack_cpu[layer_idx])
        del layer_stack, layer_stack_cpu, output

        # Stage 1: Vision Encoder (hook 캡처) → GPU pool → CPU 1회
        if 'out' in ve_cache:
            ve_out = ve_cache['out']
            if args.pool_spatial and ve_out.dim() == 3:
                ve_out = ve_out.mean(dim=1)  # (T, D_ve)
            writer.write_single("vision_encoder", ve_out.reshape(-1).cpu().to(torch.float16))

        # Stage 2: After Projector (hook 캡처 + pooling, gating 전)
        if 'out' in ap_cache:
            with torch.inference_mode():
                pooled = _apply_spatial_pool(model, ap_cache['out'], stride, pool_mode)
            if args.pool_spatial and pooled.dim() == 3:
                pooled = pooled.mean(dim=1)
            writer.write_single("after_projector", pooled.reshape(-1).cpu().to(torch.float16))

        # Stage 2.5: After Gate
        if 'out' in gate_cache:
            gate_out = gate_cache['out']
            if args.pool_spatial and gate_out.dim() == 3:
                gate_out = gate_out.mean(dim=1)
            writer.write_single("after_gate", gate_out.reshape(-1).cpu().to(torch.float16))

        writer.advance()
        all_labels.append(label_idx)
        all_qids.append(question_id)

        # GPU 메모리 해제 (empty_cache는 50샘플마다 — 매번 호출 시 sync 병목)
        ve_cache.clear()
        ap_cache.clear()
        gate_cache.clear()
        if writer.actual_samples % 50 == 0:
            torch.cuda.empty_cache()

    h_vt.remove()
    h_proj.remove()
    if h_gate is not None:
        h_gate.remove()

    # mmap flush + meta/labels 저장
    meta_common = {
        "num_classes": num_classes,
        "label_list": label_list,
        "model_name": model_name,
        "task": args.task,
    }
    extra_meta = {
        "num_frames": actual_num_frames,
        "tokens_per_frame_post": 1 if args.pool_spatial else tokens_per_frame_post,
        "tokens_per_frame_pre": 1 if args.pool_spatial else tokens_per_frame_pre,
        "hidden_dim": hidden_dim,
        "vision_hidden_dim": vision_hidden_dim,
        "pool_spatial": bool(args.pool_spatial),
    }
    writer.finalize(all_labels, all_qids, meta_common, extra_meta, num_layers)
    return None  # 이미 disk에 저장 완료


# ============================================================
#  Qwen3-VL 계열 추출 (LLM per-layer만)
# ============================================================

def load_video_frames(video_path, num_frames=8, resize=None):
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
    """Qwen3-VL: LLM per-layer만 추출."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_args_dict = parse_model_args(args.model_args)
    pretrained = model_args_dict.get("pretrained", "")
    cache_dir = os.environ.get("HF_HOME", None)

    print(f"[MODEL] Loading Qwen3-VL: {pretrained}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pretrained, dtype="auto",
        device_map=model_args_dict.get("device_map", "auto"),
        cache_dir=cache_dir,
    )
    processor_kwargs = {}
    for pk in ("min_pixels", "max_pixels"):
        if pk in model_args_dict:
            processor_kwargs[pk] = int(model_args_dict[pk])
    processor = AutoProcessor.from_pretrained(pretrained, cache_dir=cache_dir, **processor_kwargs)
    model.eval()

    max_num_frames = int(model_args_dict.get("max_num_frames", args.frames_upbound))
    resize_frames = None
    if args.resize_frames:
        parts = args.resize_frames.split("x")
        resize_frames = (int(parts[0]), int(parts[1]))

    model_name = os.path.basename(pretrained.rstrip("/"))
    num_layers = model.config.text_config.num_hidden_layers + 1
    hidden_dim = model.config.text_config.hidden_size

    VIDEO_PAD_TOKEN_ID = processor.tokenizer.encode("<|video_pad|>", add_special_tokens=False)[0]
    print(f"[INFO] num_layers: {num_layers}, hidden_dim: {hidden_dim}")
    print(f"[INFO] max_num_frames: {max_num_frames}")
    print(f"[WARN] Qwen3-VL: vision_encoder / after_projector 추출 미지원, per-layer만 추출")

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

    # 유효 샘플 수 사전 계산
    num_valid = sum(1 for q in questions if resolve_answer(q) in answer_to_idx)
    print(f"[INFO] Valid samples: {num_valid} / {len(questions)}")

    writer = FeatureWriter(args.feat_base_dir, args.task, num_valid)
    all_labels = []
    all_qids = []
    expected_num_vision_tokens = None
    actual_num_frames_final = None

    # Hook을 루프 밖에서 1회만 등록 (매 샘플 register/remove 방지)
    hidden_states_cache = {}

    def _make_embed_hook():
        def hook_fn(module, input, output):
            hidden_states_cache[0] = output.detach()
        return hook_fn

    def _make_layer_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states_cache[idx] = output[0].detach()
            else:
                hidden_states_cache[idx] = output.detach()
        return hook_fn

    hooks = []
    hooks.append(language_model.embed_tokens.register_forward_hook(_make_embed_hook()))
    for layer_idx, layer in enumerate(language_model.layers):
        hooks.append(layer.register_forward_hook(_make_layer_hook(layer_idx + 1)))

    for line in tqdm(questions, desc="Extracting features (Qwen3-VL)"):
        question_id = line["q_id"]
        answer = resolve_answer(line)
        if answer not in answer_to_idx:
            continue
        label_idx = answer_to_idx[answer]

        video_rel = line["video"]
        if args.video_folder and not os.path.isabs(video_rel):
            video_path = os.path.join(args.video_folder, video_rel)
        else:
            video_path = video_rel
        if not os.path.exists(video_path):
            hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface"))
            video_path = os.path.join(hf_cache, video_rel)

        frames, actual_num_frames = load_video_frames(video_path, max_num_frames, resize=resize_frames)
        actual_num_frames_final = actual_num_frames

        question_text = line["question"]
        video_content = {"type": "video", "video": frames}
        if max_num_frames:
            video_content["nframes"] = max_num_frames
        messages = [{"role": "user", "content": [
            video_content, {"type": "text", "text": question_text},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], return_tensors="pt")

        device = model.device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"][0]
        vision_mask = (input_ids == VIDEO_PAD_TOKEN_ID)
        vision_indices = vision_mask.nonzero(as_tuple=True)[0].tolist()
        num_vision_tokens = len(vision_indices)

        if expected_num_vision_tokens is None:
            expected_num_vision_tokens = num_vision_tokens
            print(f"[INFO] Vision tokens per sample: {num_vision_tokens}")
        elif num_vision_tokens != expected_num_vision_tokens:
            print(f"[WARN] {question_id}: vision tokens={num_vision_tokens}, skipping")
            continue

        hidden_states_cache.clear()

        with torch.inference_mode():
            model(**inputs)

        for layer_idx in range(num_layers):
            hs = hidden_states_cache[layer_idx]
            if hs.dim() == 3:
                hs = hs[0]
            vision_hs = hs[vision_indices, :]
            writer.write_layer(layer_idx, vision_hs.reshape(-1).cpu().to(torch.float16))

        writer.advance()
        all_labels.append(label_idx)
        all_qids.append(question_id)

        del inputs
        hidden_states_cache.clear()
        if writer.actual_samples % 50 == 0:
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    tokens_per_frame = (expected_num_vision_tokens or 0) // max(actual_num_frames_final or 1, 1)

    meta_common = {
        "num_classes": num_classes,
        "label_list": label_list,
        "model_name": model_name,
        "task": args.task,
    }
    extra_meta = {
        "num_frames": actual_num_frames_final,
        "tokens_per_frame_post": tokens_per_frame,
        "hidden_dim": hidden_dim,
    }
    writer.finalize(all_labels, all_qids, meta_common, extra_meta, num_layers)
    return None


# ============================================================
#  메인
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vision features (all stages) for linear probing")
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "llava", "qwen3_vl"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--feat_base_dir", type=str, required=True,
                        help="e.g., linear_probe_features/llava-video-7b")
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize_frames", type=str, default=None)
    parser.add_argument("--pool_spatial", action="store_true", default=False,
                        help="Spatial(N) axis mean-pool 후 저장. Shape: (N_samples, T, D) flattened. "
                             "Temporal 구조 보존, direction/motion 분석에 권장.")

    args = parser.parse_args()

    model_type = args.model_type
    if model_type == "auto":
        model_args_dict = parse_model_args(args.model_args)
        pretrained = model_args_dict.get("pretrained", "")
        model_type = detect_model_type(pretrained)
        print(f"[INFO] Auto-detected model type: {model_type}")

    if model_type == "qwen3_vl":
        extract_features_qwen3_vl(args)
    else:
        extract_features_llava(args)
