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

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


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

    pool_stride = getattr(config, 'mm_spatial_pool_stride', None)
    if pool_stride and pool_stride > 1:
        grid_h = math.ceil(grid_h / pool_stride)
        grid_w = math.ceil(grid_w / pool_stride)

    return grid_h, grid_w


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
        grid_h, grid_w = get_vision_grid_size(model)
        tokens_per_frame = grid_h * grid_w
        mm_newline = getattr(model.config, "mm_newline_position", "one_token")
        has_newline = mm_newline == "one_token"
        stride = tokens_per_frame + (1 if has_newline else 0)

        vision_labels = []
        for f in range(num_frames):
            for p in range(tokens_per_frame):
                vision_labels.append(f"[F{f}_p{p}]")
            if has_newline and f < num_frames - 1:
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
        grid_h, grid_w = get_vision_grid_size(model)
        tokens_per_frame = grid_h * grid_w
        mm_newline = getattr(model.config, "mm_newline_position", "one_token")
        has_newline = mm_newline == "one_token"
        stride = tokens_per_frame + (1 if has_newline else 0)

        groups = []  # list of (start_in_seq, end_in_seq, label)
        for f in range(num_frames):
            g_start = img_start + f * stride
            g_end = min(g_start + tokens_per_frame, img_end)
            if g_start >= img_end:
                break
            groups.append((g_start, g_end, f"[F{f}]"))
            # Skip newline token (it gets absorbed into the frame group)
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

    R_0 = A_0
    R_l = A_l @ R_{l-1}
    where A_l = 0.5 * attn_l + 0.5 * I  (residual connection)

    Args:
        attentions: list of (1, num_heads, seq_len, seq_len) tensors

    Returns:
        rollout: (seq_len, seq_len) numpy array — effective attention from each position
    """
    result = None
    for layer_attn in attentions:
        # Average over heads: (seq_len, seq_len)
        a = layer_attn.squeeze(0).float().mean(dim=0)
        seq_len = a.shape[0]
        # Add residual connection
        a = 0.5 * a + 0.5 * torch.eye(seq_len)
        # Re-normalize rows
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if result is None:
            result = a
        else:
            result = a @ result

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
        rollout = llm_attention_rollout(attentions)
        # Last row = answer token attending to all positions
        answer_attn = rollout[-1]
        attn_on_vision = answer_attn[img_start:img_end]
    else:
        # Simple average across layers
        per_layer = []
        for layer_attn in attentions:
            a = layer_attn.squeeze(0).float().mean(dim=0)  # (seq, seq)
            per_layer.append(a[-1, img_start:img_end].numpy())
        attn_on_vision = np.mean(per_layer, axis=0)

    # Normalize to [0, 1]
    v_min, v_max = attn_on_vision.min(), attn_on_vision.max()
    if v_max - v_min > 1e-8:
        attn_on_vision = (attn_on_vision - v_min) / (v_max - v_min)

    return attn_on_vision


def attention_to_heatmap(attn_weights, grid_h, grid_w, image_size):
    """
    Convert 1D attention over vision tokens → 2D heatmap at image resolution.

    Args:
        attn_weights: (num_vision_tokens,) array
        grid_h, grid_w: spatial grid dimensions
        image_size: (width, height)

    Returns:
        heatmap: (H, W) float [0, 1]
    """
    import cv2

    num_tokens = grid_h * grid_w
    if len(attn_weights) > num_tokens:
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
                                      include_newline=True):
    """
    Split vision token attention into per-frame attention vectors.

    Returns:
        list of (tokens_per_frame,) arrays, one per frame
    """
    stride = tokens_per_frame + (1 if include_newline else 0)
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
