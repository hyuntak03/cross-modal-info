"""
Attention extraction & bertviz-format utilities for LLaVA-NeXT VLMs.

Core responsibilities:
  1. Hook vision tower to capture ViT attention
  2. Run single-forward inference with output_attentions=True (hook-based CPU offload)
  3. Build token labels for VLM sequences (system/image/video/question/answer)
  4. Collapse vision tokens per-frame for manageable visualization
  5. Format attention into bertviz-compatible shape: (num_layers, num_heads, seq_len, seq_len)
"""

import copy
import math
import torch
import numpy as np
from functools import wraps
from typing import List, Tuple, Optional, Dict

try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    _LLAVA_AVAILABLE = True
except ImportError:
    _LLAVA_AVAILABLE = False
    IMAGE_TOKEN_INDEX = -200  # fallback, not used in Qwen3-VL path


# ============================================================
# Vision Tower Hooking
# ============================================================

def hook_vision_tower(model):
    """
    Monkey-patch vision tower forward to capture attention weights.
    After calling: model.get_vision_tower().image_attentions
    Returns restore_fn to undo the patch.
    """
    vision_tower = model.get_vision_tower()
    original_forward = vision_tower.forward

    @wraps(original_forward)
    def patched_forward(images):
        inner_model = vision_tower.vision_tower
        if isinstance(images, list):
            all_features, all_attentions = [], []
            for image in images:
                inp = image.to(device=vision_tower.device, dtype=vision_tower.dtype).unsqueeze(0)
                out = inner_model(inp, output_hidden_states=True, output_attentions=True)
                all_attentions.append(out.attentions)
                if hasattr(vision_tower, 'feature_select'):
                    feat = vision_tower.feature_select(out).to(image.dtype)
                else:
                    feat = out.hidden_states[-1].to(image.dtype)
                all_features.append(feat)
            vision_tower.image_attentions = all_attentions[0]
            return all_features if len(all_features) > 1 else all_features[0]
        else:
            inp = images.to(device=vision_tower.device, dtype=vision_tower.dtype)
            out = inner_model(inp, output_hidden_states=True, output_attentions=True)
            vision_tower.image_attentions = out.attentions
            if hasattr(vision_tower, 'feature_select'):
                features = vision_tower.feature_select(out).to(images.dtype)
            else:
                features = out.hidden_states[-1].to(images.dtype)
            return features

    vision_tower.forward = patched_forward

    def restore_fn():
        vision_tower.forward = original_forward
        for attr in ('image_attentions', '_all_image_attentions'):
            if hasattr(vision_tower, attr):
                delattr(vision_tower, attr)

    return restore_fn


# ============================================================
# Attention Extraction (single forward pass, hook-based)
# ============================================================

@torch.no_grad()
def extract_attention(
    model, tokenizer, input_ids, image_tensor, image_sizes,
    modalities=["image"],
):
    """
    Run single forward pass and extract full attention matrices.

    Returns dict:
        - "attentions": list of (1, num_heads, seq_len, seq_len) CPU tensors, one per layer
        - "vit_attentions": tuple of ViT layer attentions (or None)
        - "input_ids": original input_ids
        - "inputs_embeds_shape": shape of fused embeddings
        - "image_token_range": (start, end) of vision tokens
        - "predicted_token": str
        - "predicted_id": int
    """
    restore_fn = hook_vision_tower(model)

    try:
        position_ids = None
        attention_mask = None

        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, None, None,
                image_tensor, modalities, image_sizes=image_sizes
            )

        inputs_embeds_shape = inputs_embeds.shape

        # image token range
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        img_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
        image_start = img_positions[0] if img_positions else 0
        image_token_range = (image_start, image_start + image_dim)

        vit_attentions = getattr(model.get_vision_tower(), 'image_attentions', None)

        # Hook-based attention capture (CPU offload per layer to avoid OOM)
        captured_attentions = []
        hooks = []

        def _make_hook(layer_idx):
            def _hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    captured_attentions.append(output[1].cpu())
                    return (output[0], None) + output[2:]
                return output
            return _hook

        decoder_layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            decoder_layers = model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
            decoder_layers = model.model.model.layers

        if decoder_layers is not None:
            for i, layer in enumerate(decoder_layers):
                hooks.append(layer.register_forward_hook(_make_hook(i)))

        try:
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                return_dict=True,
            )

            logits = outputs.logits
            predicted_id = logits[0, -1].argmax().item()
            predicted_token = tokenizer.decode([predicted_id], skip_special_tokens=True)

            if captured_attentions:
                attentions = captured_attentions
            else:
                attentions = [a.cpu() for a in outputs.attentions]

            del outputs
            torch.cuda.empty_cache()

        finally:
            for h in hooks:
                h.remove()

        return {
            "attentions": attentions,  # list of (1, num_heads, seq_len, seq_len)
            "vit_attentions": vit_attentions,
            "input_ids": input_ids.cpu(),
            "inputs_embeds_shape": inputs_embeds_shape,
            "image_token_range": image_token_range,
            "predicted_token": predicted_token,
            "predicted_id": predicted_id,
        }

    finally:
        restore_fn()


# ============================================================
# Fast Extraction (in-hook GPU aggregation)
# ============================================================

@torch.no_grad()
def extract_attention_fast(
    model, tokenizer, input_ids, image_tensor, image_sizes,
    modalities=["image"],
    num_frames=None, tokens_per_frame=None, inter_frame_tokens=0,
    layer_stride=4,
):
    """
    Fast attention extraction — hooks into attention modules to compute Q·K^T
    attention scores on GPU, while the model runs with flash attention.

    Args:
        layer_stride: only compute attention every N layers (default=4, 28→7 layers).
                      Set to 1 for all layers.

    Returns dict with precomputed results.
    """
    restore_fn = hook_vision_tower(model)

    try:
        position_ids = None
        attention_mask = None

        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, None, None,
                image_tensor, modalities, image_sizes=image_sizes
            )

        inputs_embeds_shape = inputs_embeds.shape
        seq_len = inputs_embeds_shape[1]

        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        img_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
        image_start = img_positions[0] if img_positions else 0
        image_token_range = (image_start, image_start + image_dim)
        img_start, img_end = image_token_range

        vit_attentions = getattr(model.get_vision_tower(), 'image_attentions', None)

        # Frame ranges
        frame_ranges = []
        if num_frames and num_frames > 1 and tokens_per_frame:
            stride = tokens_per_frame + inter_frame_tokens
            for f in range(num_frames):
                s = img_start + f * stride
                e = s + tokens_per_frame
                frame_ranges.append((s, min(e, img_end)))

        q_indices = list(range(img_end, seq_len))

        # Collapse groups for bertviz
        if num_frames and num_frames > 1 and tokens_per_frame:
            collapse_groups = []
            for i in range(img_start):
                collapse_groups.append((i, i + 1))
            stride_c = tokens_per_frame + inter_frame_tokens
            for f in range(num_frames):
                gs = img_start + f * stride_c
                ge = min(gs + tokens_per_frame, img_end)
                if gs < img_end:
                    collapse_groups.append((gs, ge))
            for i in range(img_end, seq_len):
                collapse_groups.append((i, i + 1))
        else:
            collapse_groups = []
            for i in range(img_start):
                collapse_groups.append((i, i + 1))
            collapse_groups.append((img_start, img_end))
            for i in range(img_end, seq_len):
                collapse_groups.append((i, i + 1))
        n_collapsed = len(collapse_groups)

        # Precompute weight matrices for vectorized block averaging
        # collapsed = W @ attn @ W.T  (two matmuls, no Python loop)
        collapse_weights = torch.zeros(n_collapsed, seq_len)
        for ci, (gs, ge) in enumerate(collapse_groups):
            n = ge - gs
            collapse_weights[ci, gs:ge] = 1.0 / n

        # Frame weight matrix: (F, seq_len) — mean over patches within each frame
        frame_weights = None
        if frame_ranges:
            nf = len(frame_ranges)
            frame_weights = torch.zeros(nf, seq_len)
            for fi, (si, ei) in enumerate(frame_ranges):
                frame_weights[fi, si:ei] = 1.0 / (ei - si)

        # Accumulators (CPU, small)
        per_layer_last_row = []   # (L, seq_len)
        per_layer_f2f = []        # (L, F, F)
        per_layer_a2f = []        # (L, F)
        per_layer_q2f = []        # (L, Q, F)
        per_layer_collapsed = []  # (L, 1, 1, C, C)
        hooks = []

        # Strategy: capture Q, K from q_proj/k_proj hooks (already computed by model),
        # only on selected layers (layer_stride), compute attention with half precision.
        decoder_layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            decoder_layers = model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
            decoder_layers = model.model.model.layers

        causal_mask_cache = {}  # cache causal mask per seq_len

        if decoder_layers is not None:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            n_total_layers = len(decoder_layers)
            selected_layers = list(range(0, n_total_layers, layer_stride))
            # Always include last layer
            if selected_layers[-1] != n_total_layers - 1:
                selected_layers.append(n_total_layers - 1)

            # Detect RoPE API once
            import inspect as _inspect
            _first_attn = decoder_layers[0].self_attn
            _rope_uses_pos_ids = 'position_ids' in _inspect.signature(_first_attn.rotary_emb.forward).parameters

            print(f"  [hooks] {len(selected_layers)}/{n_total_layers} layers (stride={layer_stride})")

            for i in selected_layers:
                layer = decoder_layers[i]
                attn_mod = layer.self_attn
                captured = {}

                def _make_q_hook(cap):
                    def _hook(module, input, output):
                        cap['q'] = output
                        return output
                    return _hook

                def _make_k_hook(cap):
                    def _hook(module, input, output):
                        cap['k'] = output
                        return output
                    return _hook

                def _make_layer_hook(layer_idx, attn_mod, cap):
                    def _hook(module, input, output):
                        if 'q' not in cap or 'k' not in cap:
                            return output

                        q_raw = cap.pop('q')
                        k_raw = cap.pop('k')

                        head_dim = attn_mod.head_dim
                        n_heads = attn_mod.num_heads
                        n_kv_heads = attn_mod.num_key_value_heads
                        n_groups = n_heads // n_kv_heads
                        bsz = q_raw.shape[0]
                        q_len = q_raw.shape[1]

                        # Half precision for speed
                        q = q_raw.view(bsz, q_len, n_heads, head_dim).transpose(1, 2).half()
                        k = k_raw.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2).half()

                        # Use actual position_ids from prepare_inputs_labels_for_multimodal
                        pos_ids = position_ids if position_ids is not None else \
                            torch.arange(q_len, device=q.device).unsqueeze(0)
                        if _rope_uses_pos_ids:
                            cos, sin = attn_mod.rotary_emb(k, pos_ids)
                        else:
                            cos, sin = attn_mod.rotary_emb(k, seq_len=q_len)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

                        if n_groups > 1:
                            k = k.repeat_interleave(n_groups, dim=1)

                        # Batched matmul in half precision
                        a = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

                        # Cached causal mask
                        if q_len not in causal_mask_cache:
                            causal_mask_cache[q_len] = torch.triu(
                                torch.ones(q_len, q_len, device=a.device, dtype=torch.bool), diagonal=1)
                        a.masked_fill_(causal_mask_cache[q_len], float('-inf'))
                        a = torch.softmax(a, dim=-1).squeeze(0).float().mean(dim=0)  # (S, S)

                        del q, k, q_raw, k_raw

                        if a.isnan().any():
                            del a
                            return output

                        # --- Extract all metrics with ONE .cpu() call, NO .item() ---
                        a_cpu = a.cpu()  # single GPU→CPU transfer
                        del a

                        per_layer_last_row.append(a_cpu[-1])

                        if frame_weights is not None:
                            fw = frame_weights
                            nf = fw.shape[0]

                            # Answer→Frame: sum of attention from last token to each frame's patches
                            last_row = a_cpu[-1]  # (S,)
                            a2f_row = torch.zeros(nf)
                            for fi, (si, ei) in enumerate(frame_ranges):
                                a2f_row[fi] = last_row[si:ei].sum()
                            a2f_total = a2f_row.sum().clamp(min=1e-8)
                            per_layer_a2f.append((a2f_row / a2f_total).numpy())

                            # Frame-to-frame: fw @ a_cpu @ fw.T → (F, F)
                            f2f = fw @ a_cpu @ fw.T  # (F, F) block mean
                            rs = f2f.sum(dim=1, keepdim=True).clamp(min=1e-8)
                            per_layer_f2f.append((f2f / rs).numpy())

                            # Question→Frame: sum of attention from each q token to each frame
                            if q_indices:
                                q_rows = a_cpu[q_indices]  # (Q, S)
                                q2f = torch.zeros(len(q_indices), nf)
                                for fi, (si, ei) in enumerate(frame_ranges):
                                    q2f[:, fi] = q_rows[:, si:ei].sum(dim=1)
                                qt = q2f.sum(dim=1, keepdim=True).clamp(min=1e-8)
                                per_layer_q2f.append((q2f / qt).numpy())

                        # Collapsed attention — two matmuls, no Python loop
                        # collapsed = W @ a_cpu @ W.T
                        collapsed = collapse_weights @ a_cpu @ collapse_weights.T  # (C, C)
                        rs = collapsed.sum(dim=1, keepdim=True).clamp(min=1e-8)
                        per_layer_collapsed.append((collapsed / rs).unsqueeze(0).unsqueeze(0))

                        del a_cpu
                        return output
                    return _hook

                hooks.append(attn_mod.q_proj.register_forward_hook(_make_q_hook(captured)))
                hooks.append(attn_mod.k_proj.register_forward_hook(_make_k_hook(captured)))
                hooks.append(layer.register_forward_hook(_make_layer_hook(i, attn_mod, captured)))

        try:
            import time
            n_total_layers = len(decoder_layers) if decoder_layers else 0
            _t_start = time.time()

            print(f"  [forward] seq_len={seq_len}, layers={n_total_layers}, starting...")
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                return_dict=True,
            )
            _t_forward = time.time() - _t_start
            print(f"  [forward] done in {_t_forward:.1f}s, hooks captured {len(per_layer_last_row)}/{n_total_layers} layers")

            logits = outputs.logits
            predicted_id = logits[0, -1].argmax().item()
            predicted_token = tokenizer.decode([predicted_id], skip_special_tokens=True)

            del outputs
            torch.cuda.empty_cache()

        finally:
            for h in hooks:
                h.remove()

        # Assemble
        collapsed_attentions = per_layer_collapsed

        if per_layer_last_row:
            rows = torch.stack(per_layer_last_row)  # (L, seq_len)
            # Layer average
            vision_avg = rows[:, img_start:img_end].mean(dim=0).numpy()
            v_min, v_max = vision_avg.min(), vision_avg.max()
            if v_max - v_min > 1e-8:
                vision_avg = (vision_avg - v_min) / (v_max - v_min)

            # Rollout approximation: later layers get exponentially more weight
            # (approximates multiplicative rollout without needing full matrix)
            n_layers = len(per_layer_last_row)
            weights = torch.exp(torch.linspace(0, 2, n_layers))  # later layers matter more
            weights = weights / weights.sum()
            vision_rollout = (rows[:, img_start:img_end] * weights.unsqueeze(1)).sum(dim=0).numpy()
            v_min, v_max = vision_rollout.min(), vision_rollout.max()
            if v_max - v_min > 1e-8:
                vision_rollout = (vision_rollout - v_min) / (v_max - v_min)
        else:
            vision_avg = np.zeros(img_end - img_start)
            vision_rollout = vision_avg

        cross_frame = None
        if per_layer_f2f:
            cross_frame = {
                "frame_to_frame": np.array(per_layer_f2f),
                "answer_to_frame": np.array(per_layer_a2f),
                "question_to_frame": np.array(per_layer_q2f) if per_layer_q2f else np.array([]),
                "question_tokens": [],
                "num_valid_layers": len(per_layer_f2f),
            }

        return {
            "attentions": collapsed_attentions,
            "vit_attentions": vit_attentions,
            "input_ids": input_ids.cpu(),
            "inputs_embeds_shape": inputs_embeds_shape,
            "image_token_range": image_token_range,
            "predicted_token": predicted_token,
            "predicted_id": predicted_id,
            "vision_attn_rollout": vision_rollout,
            "vision_attn_avg": vision_avg,
            "cross_frame": cross_frame,
        }

    finally:
        restore_fn()


# ============================================================
# Token Labeling
# ============================================================

def get_vision_grid_size(model):
    """Get (grid_h, grid_w) of vision tokens after spatial pooling."""
    vision_tower = model.get_vision_tower()
    config = model.config

    if hasattr(vision_tower, 'config'):
        vt_config = vision_tower.config
        if hasattr(vt_config, 'image_size') and hasattr(vt_config, 'patch_size'):
            img_size = vt_config.image_size
            patch_size = vt_config.patch_size
            if isinstance(img_size, (tuple, list)):
                grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size
            else:
                grid_h = grid_w = img_size // patch_size
        else:
            num_patches = getattr(vision_tower, 'num_patches', 576)
            grid_h = grid_w = int(num_patches ** 0.5)
    else:
        grid_h = grid_w = 24

    # LLaVA-NeXT uses mm_spatial_pool_stride for video.
    # Default is 2 when mm_spatial_pool_mode is set (see llava_arch.py:464)
    pool_stride = getattr(config, 'mm_spatial_pool_stride', None)
    pool_mode = getattr(config, 'mm_spatial_pool_mode', None)
    if pool_stride is None and pool_mode is not None:
        pool_stride = 2  # LLaVA-NeXT default
    if pool_stride and pool_stride > 1:
        grid_h = math.ceil(grid_h / pool_stride)
        grid_w = math.ceil(grid_w / pool_stride)

    return grid_h, grid_w


def get_tokens_per_frame(model):
    """
    Calculate total tokens per frame including newline tokens.

    mm_newline_position modes:
      - "grid": newline at end of each row → grid_h * (grid_w + 1) per frame, no inter-frame token
      - "one_token": flat patches + 1 newline between frames → grid_h * grid_w + 1
      - "frame": 1 newline per frame → grid_h * grid_w + 1
      - "no_token": no newlines → grid_h * grid_w

    Returns:
        (tokens_per_frame, inter_frame_tokens):
            tokens_per_frame: total tokens within a single frame (patches + any intra-frame newlines)
            inter_frame_tokens: tokens between frames (0 or 1)
    """
    grid_h, grid_w = get_vision_grid_size(model)
    mm_newline = getattr(model.config, "mm_newline_position", "one_token")

    if mm_newline == "grid":
        # Newline at end of each row: grid_h rows × (grid_w patches + 1 newline)
        tokens_per_frame = grid_h * (grid_w + 1)
        inter_frame_tokens = 0
    elif mm_newline in ("one_token", "frame"):
        tokens_per_frame = grid_h * grid_w
        inter_frame_tokens = 1
    else:  # no_token
        tokens_per_frame = grid_h * grid_w
        inter_frame_tokens = 0

    return tokens_per_frame, inter_frame_tokens


def build_token_labels(
    tokenizer, input_ids, inputs_embeds_shape,
    image_token_range, num_frames=None, model=None,
):
    """
    Build human-readable token labels for the full VLM sequence.

    For vision tokens:
      - Image: [IMG_0], [IMG_1], ... or collapsed to [IMG]
      - Video: [F0_p0], [F0_p1], ..., [F0_nl], [F1_p0], ... or collapsed to [F0], [F1], ...

    Args:
        tokenizer: HF tokenizer
        input_ids: (1, text_seq_len) tensor (before vision expansion)
        inputs_embeds_shape: shape of fused embeddings (1, total_seq_len, hidden)
        image_token_range: (start, end) of vision tokens
        num_frames: number of video frames (None for image)
        model: LLaVA model (for grid size calculation)

    Returns:
        list of str, length = total_seq_len
    """
    total_len = inputs_embeds_shape[1]
    img_start, img_end = image_token_range
    num_vision_tokens = img_end - img_start

    # Decode text tokens (before and after image placeholder)
    input_id_list = input_ids[0].tolist()
    img_placeholder_idx = input_id_list.index(IMAGE_TOKEN_INDEX) if IMAGE_TOKEN_INDEX in input_id_list else 0

    # Tokens before vision
    text_before = []
    for tid in input_id_list[:img_placeholder_idx]:
        text_before.append(tokenizer.decode([tid]))

    # Tokens after vision
    text_after = []
    for tid in input_id_list[img_placeholder_idx + 1:]:
        text_after.append(tokenizer.decode([tid]))

    # Vision token labels
    if num_frames is not None and num_frames > 1 and model is not None:
        tpf, inter = get_tokens_per_frame(model)
        stride = tpf + inter

        vision_labels = []
        for f in range(num_frames):
            for p in range(tpf):
                vision_labels.append(f"[F{f}_p{p}]")
            if inter > 0 and f < num_frames - 1:
                vision_labels.append(f"[F{f}_nl]")

        # Pad or truncate
        if len(vision_labels) < num_vision_tokens:
            for i in range(num_vision_tokens - len(vision_labels)):
                vision_labels.append(f"[V_{len(vision_labels) + i}]")
        vision_labels = vision_labels[:num_vision_tokens]
    else:
        vision_labels = [f"[IMG_{i}]" for i in range(num_vision_tokens)]

    labels = text_before + vision_labels + text_after

    # Pad/truncate to match total_len
    if len(labels) < total_len:
        labels += [f"[PAD_{i}]" for i in range(total_len - len(labels))]
    labels = labels[:total_len]

    return labels


def collapse_vision_tokens(
    attentions, token_labels, image_token_range,
    num_frames=None, model=None,
):
    """
    Collapse vision tokens into per-frame (video) or single [IMG] (image) tokens
    by averaging attention weights. Makes visualization manageable.

    Args:
        attentions: list of (1, num_heads, seq_len, seq_len) tensors
        token_labels: list of str, length seq_len
        image_token_range: (start, end)
        num_frames: number of frames (None for image → collapse to single [IMG])
        model: for grid size

    Returns:
        collapsed_attentions: list of (1, num_heads, new_len, new_len) tensors
        collapsed_labels: list of str
    """
    img_start, img_end = image_token_range
    num_vision = img_end - img_start
    seq_len = len(token_labels)

    # Determine groups
    if num_frames is not None and num_frames > 1 and model is not None:
        tpf, inter = get_tokens_per_frame(model)
        stride = tpf + inter

        groups = []  # list of (start_in_seq, end_in_seq, label)
        for f in range(num_frames):
            g_start = img_start + f * stride
            g_end = min(g_start + tpf, img_end)
            if g_start >= img_end:
                break
            groups.append((g_start, g_end, f"[F{f}]"))
    else:
        groups = [(img_start, img_end, "[IMG]")]

    # Build index mapping: new_idx → list of old_indices
    new_indices = []
    new_labels = []

    # Text before vision
    for i in range(img_start):
        new_indices.append([i])
        new_labels.append(token_labels[i])

    # Vision groups
    for g_start, g_end, label in groups:
        new_indices.append(list(range(g_start, g_end)))
        new_labels.append(label)

    # Handle newline tokens between frames (absorb into preceding frame)
    # Already handled by skipping them in the group logic above

    # Text after vision: need to account for any newline tokens we skipped
    for i in range(img_end, seq_len):
        new_indices.append([i])
        new_labels.append(token_labels[i])

    new_len = len(new_indices)

    # Collapse attention matrices
    collapsed_attentions = []
    for layer_attn in attentions:
        # layer_attn: (1, num_heads, seq_len, seq_len)
        attn = layer_attn.float()
        num_heads = attn.shape[1]

        new_attn = torch.zeros(1, num_heads, new_len, new_len)

        for new_i, old_is in enumerate(new_indices):
            for new_j, old_js in enumerate(new_indices):
                # Average attention from group_i to group_j
                if len(old_is) == 1 and len(old_js) == 1:
                    new_attn[0, :, new_i, new_j] = attn[0, :, old_is[0], old_js[0]]
                else:
                    block = attn[0, :, old_is[0]:old_is[-1]+1, old_js[0]:old_js[-1]+1]
                    new_attn[0, :, new_i, new_j] = block.mean(dim=(-2, -1))

        # Re-normalize each row
        row_sums = new_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        new_attn = new_attn / row_sums

        collapsed_attentions.append(new_attn)

    return collapsed_attentions, new_labels


# ============================================================
# LLM Attention Rollout
# ============================================================

def llm_attention_rollout(attentions):
    """
    Attention Rollout across LLM layers.

    For deep models (28+ layers), full matrix multiplication causes numerical
    underflow. Instead we track only the last row (answer token's effective
    attention to all positions) which is what we actually need.

    R_0 = A_0[-1, :]  (last row of first layer)
    R_l = A_l[-1, :] * residual_weight + R_{l-1} @ A_l * residual_weight
    Simplified: r = 0.5 * a[-1] + 0.5 * (r @ a)  per layer, then normalize.

    Args:
        attentions: list of (1, num_heads, seq_len, seq_len) tensors

    Returns:
        rollout: (seq_len,) numpy array — effective attention from last position
    """
    result = None  # (seq_len,) — last token's effective attention
    for layer_idx, layer_attn in enumerate(attentions):
        a = layer_attn.squeeze(0).float().mean(dim=0)  # (seq, seq)

        # Skip layers with NaN (can happen from float16 underflow)
        if a.isnan().any():
            continue

        seq_len = a.shape[0]

        if result is None:
            # First layer: just take last row with residual
            result = 0.5 * a[-1] + 0.5 * torch.zeros(seq_len)
            result[-1] += 0.5  # residual identity for last token
        else:
            # Rollout: combine residual path with attention path
            result = 0.5 * a[-1] + 0.5 * (result @ a)

        # Re-normalize to keep as distribution
        result = result / result.sum().clamp(min=1e-8)

    return result.numpy()


def extract_answer_vision_attention(attentions, image_token_range, method="rollout"):
    """
    Extract attention from the answer position (last token) to vision tokens.

    Args:
        attentions: list of (1, num_heads, seq_len, seq_len) tensors
        image_token_range: (start, end)
        method: "rollout" or "avg" (simple layer average)

    Returns:
        attn_on_vision: (num_vision_tokens,) numpy array
    """
    img_start, img_end = image_token_range

    if method == "rollout":
        rollout = llm_attention_rollout(attentions)  # (seq_len,)
        attn_on_vision = rollout[img_start:img_end]
    else:
        # Simple average across layers
        per_layer = []
        for layer_attn in attentions:
            a = layer_attn.squeeze(0).float().mean(dim=0)  # (seq, seq)
            row = a[-1, img_start:img_end]
            if not row.isnan().any():
                per_layer.append(row.numpy())
        attn_on_vision = np.mean(per_layer, axis=0) if per_layer else np.zeros(img_end - img_start)

    # Normalize to [0, 1]
    v_min, v_max = attn_on_vision.min(), attn_on_vision.max()
    if v_max - v_min > 1e-8:
        attn_on_vision = (attn_on_vision - v_min) / (v_max - v_min)

    return attn_on_vision


def attention_to_heatmap(attn_weights, grid_h, grid_w, image_size):
    """
    Convert 1D attention over vision tokens → 2D heatmap at image resolution.

    Handles both flat layout (grid_h * grid_w tokens) and grid-newline layout
    (grid_h * (grid_w + 1) tokens, where each row has an extra newline token).

    Args:
        attn_weights: (num_vision_tokens,) array
        grid_h, grid_w: spatial grid dimensions (after pooling)
        image_size: (width, height)

    Returns:
        heatmap: (H, W) float [0, 1]
    """
    import cv2

    num_tokens = grid_h * grid_w
    num_tokens_with_nl = grid_h * (grid_w + 1)

    if len(attn_weights) == num_tokens_with_nl:
        # Grid newline mode: remove newline at end of each row
        attn_weights = attn_weights.reshape(grid_h, grid_w + 1)[:, :grid_w].flatten()
    elif len(attn_weights) > num_tokens:
        attn_weights = attn_weights[:num_tokens]
    elif len(attn_weights) < num_tokens:
        attn_weights = np.concatenate([attn_weights, np.zeros(num_tokens - len(attn_weights))])

    attn_map = attn_weights.reshape(grid_h, grid_w)
    w, h = image_size
    heatmap = cv2.resize(attn_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def split_vision_attention_by_frames(attn_on_vision, num_frames, tokens_per_frame,
                                      inter_frame_tokens=1):
    """
    Split vision token attention into per-frame attention vectors.

    Args:
        attn_on_vision: (total_vision_tokens,) attention on all vision tokens
        num_frames: number of video frames
        tokens_per_frame: tokens within a single frame (patches + any intra-frame newlines)
        inter_frame_tokens: tokens between frames (0 for grid mode, 1 for one_token mode)

    Returns:
        list of (tokens_per_frame,) arrays, one per frame
    """
    stride = tokens_per_frame + inter_frame_tokens
    per_frame = []
    for f in range(num_frames):
        start = f * stride
        end = start + tokens_per_frame
        if end <= len(attn_on_vision):
            per_frame.append(attn_on_vision[start:end])
        else:
            per_frame.append(np.zeros(tokens_per_frame))
    return per_frame


# ============================================================
# Cross-frame Interaction Analysis
# ============================================================

def _frame_ranges(image_token_range, num_frames, tokens_per_frame, inter_frame_tokens):
    """Get (start, end) indices in the full sequence for each frame."""
    img_start = image_token_range[0]
    stride = tokens_per_frame + inter_frame_tokens
    ranges = []
    for f in range(num_frames):
        s = img_start + f * stride
        e = s + tokens_per_frame
        ranges.append((s, e))
    return ranges


def compute_cross_frame_analysis(
    attentions, token_labels, image_token_range,
    num_frames, tokens_per_frame, inter_frame_tokens,
):
    """
    Compute comprehensive cross-frame attention analysis.

    Args:
        attentions: list of (1, num_heads, seq_len, seq_len) tensors
        token_labels: list of str (full sequence)
        image_token_range: (start, end)
        num_frames: int
        tokens_per_frame: int (including intra-frame newlines for grid mode)
        inter_frame_tokens: int (0 for grid, 1 for one_token)

    Returns dict with:
        "frame_to_frame": (num_layers, num_frames, num_frames) — how much frame_i attends to frame_j
        "answer_to_frame": (num_layers, num_frames) — answer token → each frame
        "question_to_frame": (num_layers, num_q_tokens, num_frames) — each question token → each frame
        "question_tokens": list of str — question token labels
        "num_valid_layers": int
    """
    img_start, img_end = image_token_range
    frame_ranges = _frame_ranges(image_token_range, num_frames, tokens_per_frame, inter_frame_tokens)

    # Identify question tokens: text tokens after vision, before answer
    # (everything from img_end to end of sequence minus the last few assistant template tokens)
    q_indices = []
    q_labels = []
    for i in range(img_end, len(token_labels)):
        label = token_labels[i]
        # Skip template tokens and padding
        if label.startswith("[PAD"):
            continue
        q_indices.append(i)
        q_labels.append(label)

    num_layers = len(attentions)
    frame_to_frame = []     # (L, F, F)
    answer_to_frame = []    # (L, F)
    question_to_frame = []  # (L, Q, F)

    for layer_attn in attentions:
        a = layer_attn.squeeze(0).float().mean(dim=0)  # (seq, seq)

        if a.isnan().any():
            continue

        # --- Frame-to-frame ---
        f2f = np.zeros((num_frames, num_frames))
        for i, (si, ei) in enumerate(frame_ranges):
            for j, (sj, ej) in enumerate(frame_ranges):
                if si < a.shape[0] and sj < a.shape[1]:
                    block = a[si:min(ei, a.shape[0]), sj:min(ej, a.shape[1])]
                    f2f[i, j] = block.mean().item()
        # Normalize rows
        row_sums = f2f.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-8, row_sums, 1.0)
        f2f = f2f / row_sums
        frame_to_frame.append(f2f)

        # --- Answer→Frame (last token) ---
        a2f = np.zeros(num_frames)
        for j, (sj, ej) in enumerate(frame_ranges):
            if sj < a.shape[1]:
                a2f[j] = a[-1, sj:min(ej, a.shape[1])].sum().item()
        total = a2f.sum()
        if total > 1e-8:
            a2f = a2f / total
        answer_to_frame.append(a2f)

        # --- Question→Frame ---
        q2f = np.zeros((len(q_indices), num_frames))
        for qi, q_idx in enumerate(q_indices):
            if q_idx < a.shape[0]:
                for j, (sj, ej) in enumerate(frame_ranges):
                    if sj < a.shape[1]:
                        q2f[qi, j] = a[q_idx, sj:min(ej, a.shape[1])].sum().item()
                total = q2f[qi].sum()
                if total > 1e-8:
                    q2f[qi] = q2f[qi] / total
        question_to_frame.append(q2f)

    return {
        "frame_to_frame": np.array(frame_to_frame),       # (L, F, F)
        "answer_to_frame": np.array(answer_to_frame),      # (L, F)
        "question_to_frame": np.array(question_to_frame),  # (L, Q, F)
        "question_tokens": q_labels,
        "num_valid_layers": len(frame_to_frame),
    }


# ============================================================
# Prompt Building
# ============================================================

def build_prompt(question, conv_template, model_name, tokenizer):
    """
    Build input_ids from question string using LLaVA conversation template.

    Returns:
        input_ids: (1, seq_len) tensor on CUDA
    """
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if "llama3" in model_name.lower():
        prompt += " \n"

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device='cuda')

    return input_ids


# ============================================================
# Qwen3-VL Support
# ============================================================

def _get_qwen3vl_decoder_layers(model):
    """Find decoder layers in Qwen3-VL model hierarchy."""
    for accessor in [
        lambda: model.model.language_model.layers,
        lambda: model.model.language_model.model.layers,
    ]:
        try:
            layers = accessor()
            if hasattr(layers, '__len__') and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Could not find decoder layers in Qwen3-VL model")


def get_qwen3vl_vision_token_id(processor, media_type="video"):
    """Get vision pad token ID for Qwen3-VL."""
    token = "<|video_pad|>" if media_type == "video" else "<|image_pad|>"
    return processor.tokenizer.encode(token, add_special_tokens=False)[0]


def get_qwen3vl_grid_size(tokens_per_frame):
    """Estimate (grid_h, grid_w) from token count per frame (assumes roughly square)."""
    side = int(math.sqrt(tokens_per_frame))
    if side * side == tokens_per_frame:
        return side, side
    for h in range(side + 1, 0, -1):
        if tokens_per_frame % h == 0:
            return h, tokens_per_frame // h
    return side, max(1, tokens_per_frame // side)


def build_token_labels_qwen3vl(processor, input_ids, image_token_range,
                                num_frames=None, tokens_per_frame=None):
    """
    Build human-readable token labels for Qwen3-VL sequences.

    Args:
        processor: Qwen3-VL processor
        input_ids: (seq_len,) tensor (1D, includes vision pad tokens)
        image_token_range: (start, end) of vision tokens
        num_frames: number of temporal slots
        tokens_per_frame: spatial tokens per temporal slot
    """
    total_len = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else len(input_ids)
    img_start, img_end = image_token_range
    input_id_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

    labels = []
    for i in range(total_len):
        if i < img_start or i >= img_end:
            labels.append(processor.tokenizer.decode([input_id_list[i]]))
        else:
            offset = i - img_start
            if num_frames and num_frames > 1 and tokens_per_frame:
                f_idx = offset // tokens_per_frame
                p_idx = offset % tokens_per_frame
                labels.append(f"[F{f_idx}_p{p_idx}]")
            else:
                labels.append(f"[IMG_{offset}]")

    return labels


@torch.no_grad()
def extract_attention_qwen3vl_fast(
    model, processor, inputs,
    vision_pad_token_id,
    num_frames=None,
    layer_stride=4,
):
    """
    Attention extraction for Qwen3-VL with hook-based per-layer aggregation.

    Uses eager attention (model must be loaded with attn_implementation='eager')
    so the model computes correct M-RoPE attention weights internally.
    Hooks capture and aggregate metrics on CPU per-layer, discarding raw weights
    to keep GPU memory bounded.

    Returns dict compatible with LLaVA extraction output format.
    """
    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]

    # Vision token range
    vision_mask = (input_ids == vision_pad_token_id)
    vision_indices = vision_mask.nonzero(as_tuple=True)[0]
    if len(vision_indices) == 0:
        raise ValueError("No vision tokens found in input")

    img_start = vision_indices[0].item()
    img_end = vision_indices[-1].item() + 1
    image_token_range = (img_start, img_end)
    num_vision = img_end - img_start

    # Qwen3-VL has no inter-frame newline tokens
    if num_frames and num_frames > 1:
        tokens_per_frame = num_vision // num_frames
    else:
        tokens_per_frame = num_vision
    inter_frame_tokens = 0

    # Frame ranges
    frame_ranges = []
    if num_frames and num_frames > 1:
        for f in range(num_frames):
            s = img_start + f * tokens_per_frame
            e = min(s + tokens_per_frame, img_end)
            frame_ranges.append((s, e))

    q_indices = list(range(img_end, seq_len))

    # Collapse groups (same logic as LLaVA fast path)
    collapse_groups = []
    for i in range(img_start):
        collapse_groups.append((i, i + 1))
    if num_frames and num_frames > 1:
        for f in range(num_frames):
            gs = img_start + f * tokens_per_frame
            ge = min(gs + tokens_per_frame, img_end)
            if gs < img_end:
                collapse_groups.append((gs, ge))
    else:
        collapse_groups.append((img_start, img_end))
    for i in range(img_end, seq_len):
        collapse_groups.append((i, i + 1))
    n_collapsed = len(collapse_groups)

    # Precompute weight matrices for vectorized block averaging
    collapse_weights = torch.zeros(n_collapsed, seq_len)
    for ci, (gs, ge) in enumerate(collapse_groups):
        n = ge - gs
        collapse_weights[ci, gs:ge] = 1.0 / n

    frame_weights = None
    if frame_ranges:
        nf = len(frame_ranges)
        frame_weights = torch.zeros(nf, seq_len)
        for fi, (si, ei) in enumerate(frame_ranges):
            frame_weights[fi, si:ei] = 1.0 / (ei - si)

    # Accumulators (CPU, small)
    per_layer_last_row = []
    per_layer_f2f = []
    per_layer_a2f = []
    per_layer_q2f = []
    per_layer_collapsed = []
    hooks = []

    decoder_layers = _get_qwen3vl_decoder_layers(model)
    n_total_layers = len(decoder_layers)
    selected_layers = list(range(0, n_total_layers, layer_stride))
    if selected_layers[-1] != n_total_layers - 1:
        selected_layers.append(n_total_layers - 1)
    selected_set = set(selected_layers)

    print(f"  [hooks] {len(selected_layers)}/{n_total_layers} layers (stride={layer_stride})")

    # Hook self_attn modules (not decoder layers):
    # Qwen3-VL decoder layer returns a single Tensor, but self_attn returns
    # (hidden_states, attn_weights) when output_attentions=True.
    for i in range(n_total_layers):
        attn_module = decoder_layers[i].self_attn

        def _make_hook(layer_idx, compute_metrics):
            def _hook(module, input, output):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    return output

                if compute_metrics:
                    attn = output[1]  # (batch, heads, seq, seq)
                    a = attn.squeeze(0).float().mean(dim=0)  # (S, S) head-averaged

                    if not a.isnan().any():
                        a_cpu = a.cpu()

                        per_layer_last_row.append(a_cpu[-1])

                        if frame_weights is not None:
                            fw = frame_weights
                            nf_local = fw.shape[0]

                            # Answer→Frame
                            last_row = a_cpu[-1]
                            a2f_row = torch.zeros(nf_local)
                            for fi, (si, ei) in enumerate(frame_ranges):
                                a2f_row[fi] = last_row[si:ei].sum()
                            a2f_total = a2f_row.sum().clamp(min=1e-8)
                            per_layer_a2f.append((a2f_row / a2f_total).numpy())

                            # Frame-to-frame
                            f2f = fw @ a_cpu @ fw.T
                            rs = f2f.sum(dim=1, keepdim=True).clamp(min=1e-8)
                            per_layer_f2f.append((f2f / rs).numpy())

                            # Question→Frame
                            if q_indices:
                                q_rows = a_cpu[q_indices]
                                q2f = torch.zeros(len(q_indices), nf_local)
                                for fi, (si, ei) in enumerate(frame_ranges):
                                    q2f[:, fi] = q_rows[:, si:ei].sum(dim=1)
                                qt = q2f.sum(dim=1, keepdim=True).clamp(min=1e-8)
                                per_layer_q2f.append((q2f / qt).numpy())

                        # Collapsed attention
                        collapsed = collapse_weights @ a_cpu @ collapse_weights.T
                        rs = collapsed.sum(dim=1, keepdim=True).clamp(min=1e-8)
                        per_layer_collapsed.append((collapsed / rs).unsqueeze(0).unsqueeze(0))

                        del a_cpu
                    del a

                # Discard attention weights from output to free GPU memory
                return (output[0], None) + output[2:]
            return _hook

        hooks.append(attn_module.register_forward_hook(_make_hook(i, i in selected_set)))

    try:
        import time
        _t = time.time()
        print(f"  [forward] seq_len={seq_len}, layers={n_total_layers}, starting...")

        outputs = model(**inputs, output_attentions=True, return_dict=True)

        print(f"  [forward] done in {time.time()-_t:.1f}s, "
              f"hooks captured {len(per_layer_last_row)}/{len(selected_layers)} layers")

        logits = outputs.logits
        predicted_id = logits[0, -1].argmax().item()
        predicted_token = processor.tokenizer.decode([predicted_id], skip_special_tokens=True)

        del outputs
        torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    # Assemble results (same logic as LLaVA fast path)
    collapsed_attentions = per_layer_collapsed

    if per_layer_last_row:
        rows = torch.stack(per_layer_last_row)
        # Layer average
        vision_avg = rows[:, img_start:img_end].mean(dim=0).numpy()
        v_min, v_max = vision_avg.min(), vision_avg.max()
        if v_max - v_min > 1e-8:
            vision_avg = (vision_avg - v_min) / (v_max - v_min)

        # Rollout approximation: later layers weighted exponentially more
        n_l = len(per_layer_last_row)
        weights = torch.exp(torch.linspace(0, 2, n_l))
        weights = weights / weights.sum()
        vision_rollout = (rows[:, img_start:img_end] * weights.unsqueeze(1)).sum(dim=0).numpy()
        v_min, v_max = vision_rollout.min(), vision_rollout.max()
        if v_max - v_min > 1e-8:
            vision_rollout = (vision_rollout - v_min) / (v_max - v_min)
    else:
        vision_avg = np.zeros(num_vision)
        vision_rollout = vision_avg

    cross_frame = None
    if per_layer_f2f:
        cross_frame = {
            "frame_to_frame": np.array(per_layer_f2f),
            "answer_to_frame": np.array(per_layer_a2f),
            "question_to_frame": np.array(per_layer_q2f) if per_layer_q2f else np.array([]),
            "question_tokens": [],
            "num_valid_layers": len(per_layer_f2f),
        }

    return {
        "attentions": collapsed_attentions,
        "vit_attentions": None,
        "input_ids": inputs["input_ids"].cpu(),
        "inputs_embeds_shape": (1, seq_len, model.config.text_config.hidden_size),
        "image_token_range": image_token_range,
        "predicted_token": predicted_token,
        "predicted_id": predicted_id,
        "vision_attn_rollout": vision_rollout,
        "vision_attn_avg": vision_avg,
        "cross_frame": cross_frame,
    }
