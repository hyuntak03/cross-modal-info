"""
Attention visualization for LLaVA-NeXT VLMs.

Modes:
  - grid: bertviz-style token-to-token attention lines (layer grid)
  - per_layer: 1 image per layer
  - summary: all-layer average
  - heatmap: answer token → vision token attention projected onto image/video frames
             supports both rollout and simple average

Usage:
  # Token-to-token attention grid
  python Attention_map/visualize_attention.py \
    --attn_path output/attention/video_attn.pt \
    --mode grid

  # Heatmap overlay (rollout, default)
  python Attention_map/visualize_attention.py \
    --attn_path output/attention/video_attn.pt \
    --mode heatmap

  # Heatmap overlay (simple average)
  python Attention_map/visualize_attention.py \
    --attn_path output/attention/video_attn.pt \
    --mode heatmap --heatmap_method avg

  # All modes
  python Attention_map/visualize_attention.py \
    --attn_path output/attention/video_attn.pt \
    --mode all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch

from Attention_map.attention_utils import (
    extract_answer_vision_attention,
    attention_to_heatmap,
    split_vision_attention_by_frames,
    compute_cross_frame_analysis,
)


# ============================================================
# Token coloring: vision tokens vs text tokens
# ============================================================

def _token_color(token):
    """Return color based on token type."""
    if token.startswith("[F") and ("]" in token):
        return "#2196F3"  # blue for video frames
    elif token.startswith("[IMG"):
        return "#4CAF50"  # green for image
    elif token.startswith("[PAD"):
        return "#BDBDBD"  # gray for padding
    else:
        return "#333333"  # dark for text


def _format_token(token, max_len=12):
    """Clean and truncate token for display."""
    token = token.replace('\u0120', ' ').replace('\u2581', ' ').replace('</w>', '')
    token = token.replace('\n', '\\n')
    if len(token) > max_len:
        token = token[:max_len-1] + '..'
    return token


# ============================================================
# Core drawing: attention lines between token columns
# ============================================================

def draw_attention(ax, tokens, attn_matrix, title=None, top_k=None, threshold=0.01):
    """
    Draw bertviz-style attention lines on a matplotlib axes.

    Args:
        ax: matplotlib Axes
        tokens: list of str (length N)
        attn_matrix: (N, N) numpy array, attention weights (row=query, col=key)
        title: optional title string
        top_k: if set, only show top_k attention lines per query token
        threshold: minimum attention weight to draw a line
    """
    n = len(tokens)
    tokens_display = [_format_token(t) for t in tokens]

    left_x = 0.0
    right_x = 1.0
    y_positions = np.linspace(0.95, 0.05, n)

    # Draw tokens on both sides
    for i, (tok, y) in enumerate(zip(tokens_display, y_positions)):
        color = _token_color(tokens[i])
        ax.text(left_x - 0.02, y, tok, ha='right', va='center',
                fontsize=7, color=color, fontfamily='monospace')
        ax.text(right_x + 0.02, y, tok, ha='left', va='center',
                fontsize=7, color=color, fontfamily='monospace')

    # Draw attention lines
    lines = []
    colors = []

    for i in range(n):  # query (left)
        row = attn_matrix[i]

        if top_k is not None:
            indices = np.argsort(row)[-top_k:]
        else:
            indices = np.where(row >= threshold)[0]

        for j in indices:
            weight = float(row[j])
            if weight < threshold:
                continue
            lines.append([(left_x, y_positions[i]), (right_x, y_positions[j])])
            colors.append((0.13, 0.59, 0.95, min(weight * 3, 0.9)))  # blue with alpha

    if lines:
        lc = LineCollection(lines, colors=colors, linewidths=0.8)
        ax.add_collection(lc)

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)


# ============================================================
# Visualization modes
# ============================================================

def visualize_per_layer(attentions, tokens, output_dir, base_name,
                        include_layers=None, top_k=None, threshold=0.01, fmt="png"):
    """Save one image per layer."""
    num_layers = len(attentions)
    if include_layers is None:
        include_layers = list(range(num_layers))

    for layer_idx in include_layers:
        if layer_idx >= num_layers:
            continue
        attn = attentions[layer_idx].squeeze(0).float().mean(dim=0).numpy()

        fig, ax = plt.subplots(figsize=(5, max(3, len(tokens) * 0.25)))
        draw_attention(ax, tokens, attn, title=f"Layer {layer_idx}",
                       top_k=top_k, threshold=threshold)
        path = os.path.join(output_dir, f"{base_name}_layer{layer_idx:02d}.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {path}")


def visualize_grid(attentions, tokens, output_dir, base_name,
                   include_layers=None, top_k=None, threshold=0.01, fmt="png",
                   cols=4):
    """Save a single grid image with multiple layers."""
    num_layers = len(attentions)
    if include_layers is None:
        # Default: evenly spaced 8 layers
        if num_layers <= 8:
            include_layers = list(range(num_layers))
        else:
            step = num_layers // 8
            include_layers = list(range(0, num_layers, step))[:8]

    n_panels = len(include_layers)
    rows = (n_panels + cols - 1) // cols
    cell_h = max(3, len(tokens) * 0.22)
    cell_w = 4.5

    fig, axes = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, layer_idx in enumerate(include_layers):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if layer_idx >= num_layers:
            ax.axis('off')
            continue
        attn = attentions[layer_idx].squeeze(0).float().mean(dim=0).numpy()
        draw_attention(ax, tokens, attn, title=f"Layer {layer_idx}",
                       top_k=top_k, threshold=threshold)

    # Hide unused axes
    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle(f"{base_name}", fontsize=11, fontweight='bold', y=1.0)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{base_name}_grid.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")


def visualize_summary(attentions, tokens, output_dir, base_name,
                      top_k=None, threshold=0.005, fmt="png"):
    """Save a single image with all-layer average attention."""
    # Average across all layers and heads
    all_attn = torch.stack([a.squeeze(0).float().mean(dim=0) for a in attentions])
    avg_attn = all_attn.mean(dim=0).numpy()  # (N, N)

    fig, ax = plt.subplots(figsize=(5, max(3, len(tokens) * 0.25)))
    draw_attention(ax, tokens, avg_attn, title="All Layers (avg)",
                   top_k=top_k, threshold=threshold)

    path = os.path.join(output_dir, f"{base_name}_summary.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ============================================================
# Heatmap overlay: answer → vision attention on image/frames
# ============================================================

def _overlay_heatmap(image_np, heatmap, alpha=0.5, cmap='inferno'):
    """Overlay heatmap on image. Returns (H, W, 3) uint8."""
    import matplotlib.cm as cm
    colormap_fn = cm.get_cmap(cmap)
    hm_rgba = colormap_fn(heatmap)[:, :, :3].astype(np.float32)
    img = image_np.astype(np.float32) / 255.0
    blended = alpha * hm_rgba + (1 - alpha) * img
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def visualize_heatmap(data, output_dir, base_name, method="rollout", fmt="png", cmap="inferno"):
    """
    Overlay answer-token vision attention on original image/video frames.

    Args:
        data: loaded .pt dict
        method: "rollout" or "avg"
    """
    metadata = data.get("metadata", {})
    grid_size = data.get("grid_size", None)
    frames = data.get("frames", None)

    if grid_size is None:
        print("  [SKIP heatmap] No grid_size saved — re-extract with updated extract_attention.py")
        return

    grid_h, grid_w = grid_size

    # Get precomputed vision attention, or recompute from raw
    key = f"vision_attn_{method}"
    if key in data:
        attn_on_vision = data[key]
    elif "attentions_raw" in data:
        raw = [a.float() for a in data["attentions_raw"]]
        attn_on_vision = extract_answer_vision_attention(raw, data["image_token_range"], method=method)
    else:
        print("  [SKIP heatmap] No raw attentions or precomputed vision attention")
        return

    media_type = metadata.get("type", "image")
    num_frames = metadata.get("num_frames", None)
    tokens_per_frame = metadata.get("tokens_per_frame", grid_h * grid_w)
    inter_frame_tokens = metadata.get("inter_frame_tokens",
                                      1 if metadata.get("include_newline", True) else 0)

    if frames is None:
        print("  [SKIP heatmap] No frames saved — re-extract with updated extract_attention.py")
        return

    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        # Single image stored as (H, W, 3)
        frames = [frames]

    method_label = "Rollout" if method == "rollout" else "Avg"

    if media_type == "video" and num_frames and num_frames > 1:
        # Per-frame heatmaps
        per_frame = split_vision_attention_by_frames(
            attn_on_vision, num_frames, tokens_per_frame, inter_frame_tokens
        )

        n_frames = min(len(per_frame), len(frames))
        cols = min(n_frames, 8)
        rows = (n_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
        if rows == 1 and cols * 2 == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols * 2 == 1:
            axes = axes[:, np.newaxis]

        for f_idx in range(n_frames):
            r = f_idx // cols
            c = f_idx % cols

            if isinstance(frames, np.ndarray):
                frame = frames[f_idx]
            else:
                frame = np.array(frames[f_idx])

            h, w = frame.shape[:2]
            heatmap = attention_to_heatmap(per_frame[f_idx], grid_h, grid_w, (w, h))
            overlay = _overlay_heatmap(frame, heatmap, cmap=cmap)

            # Original frame
            ax_orig = axes[r, c * 2]
            ax_orig.imshow(frame)
            ax_orig.set_title(f"F{f_idx}", fontsize=8)
            ax_orig.axis('off')

            # Overlay
            ax_over = axes[r, c * 2 + 1]
            ax_over.imshow(overlay)
            ax_over.set_title(f"F{f_idx} ({method_label})", fontsize=8)
            ax_over.axis('off')

        # Hide unused
        for idx in range(n_frames, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c * 2].axis('off')
            axes[r, c * 2 + 1].axis('off')

        question = metadata.get("question", "")[:60]
        predicted = data.get("predicted_token", "")
        fig.suptitle(f"Answer→Vision ({method_label}) | Q: {question}.. | Pred: {predicted}",
                     fontsize=10, fontweight='bold')
        fig.tight_layout()

        path = os.path.join(output_dir, f"{base_name}_heatmap_{method}.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {path}")

    else:
        # Single image
        if isinstance(frames, np.ndarray):
            image = frames[0] if frames.ndim == 4 else frames
        else:
            image = np.array(frames[0])

        h, w = image.shape[:2]
        heatmap = attention_to_heatmap(attn_on_vision, grid_h, grid_w, (w, h))
        overlay = _overlay_heatmap(image, heatmap, cmap=cmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap=cmap)
        axes[1].set_title(f"Attention ({method_label})", fontsize=9)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=9)
        axes[2].axis('off')

        question = metadata.get("question", "")[:60]
        predicted = data.get("predicted_token", "")
        fig.suptitle(f"Answer→Vision ({method_label}) | Q: {question}.. | Pred: {predicted}",
                     fontsize=10, fontweight='bold')
        fig.tight_layout()

        path = os.path.join(output_dir, f"{base_name}_heatmap_{method}.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {path}")


# ============================================================
# Attention Distribution (Sink Analysis)
# ============================================================

def compute_attention_distribution(attentions, tokens, image_token_range):
    """
    Compute per-layer attention distribution across token regions.

    Returns:
        (num_layers, 4) array — fractions going to [system, vision, question, answer_template]
    """
    img_start, img_end = image_token_range
    n_layers = len(attentions)
    dist = np.zeros((n_layers, 4))  # system, vision, question, answer_template

    for l, layer_attn in enumerate(attentions):
        a = layer_attn.squeeze(0).float().mean(dim=0)  # (C, C) head-averaged
        if a.isnan().any():
            continue
        last_row = a[-1].numpy()  # answer token's attention

        total = last_row.sum()
        if total < 1e-8:
            continue

        # Find collapsed token regions
        n = len(tokens)
        # System: tokens before first [F or [IMG
        sys_end = 0
        for i, t in enumerate(tokens):
            if t.startswith("[F") or t.startswith("[IMG"):
                sys_end = i
                break

        # Vision: [F*] or [IMG*] tokens
        vis_start, vis_end = sys_end, sys_end
        for i in range(sys_end, n):
            if tokens[i].startswith("[F") or tokens[i].startswith("[IMG"):
                vis_end = i + 1
            else:
                break

        # Answer template: last few tokens (assistant\n etc)
        ans_start = n
        for i in range(n - 1, vis_end - 1, -1):
            t = tokens[i].strip()
            if t in ('', '\\n', '\n', 'assistant', '<|im_start|>', '<|im_end|>'):
                ans_start = i
            else:
                break

        dist[l, 0] = last_row[:sys_end].sum() / total           # system/BOS
        dist[l, 1] = last_row[sys_end:vis_end].sum() / total    # vision
        dist[l, 2] = last_row[vis_end:ans_start].sum() / total  # question
        dist[l, 3] = last_row[ans_start:].sum() / total         # answer template

    return dist


def visualize_attention_distribution(attentions, tokens, image_token_range,
                                      output_dir, base_name, fmt="png"):
    """
    Stacked area chart: per-layer attention distribution
    (System/BOS vs Vision vs Question vs Answer template).
    Shows attention sink in the System/BOS region.
    """
    dist = compute_attention_distribution(attentions, tokens, image_token_range)
    n_layers = dist.shape[0]

    fig, ax = plt.subplots(figsize=(8, 4))

    layers = np.arange(n_layers)
    colors = ['#ef5350', '#42a5f5', '#66bb6a', '#ffa726']
    labels = ['System/BOS (sink)', 'Vision (frames)', 'Question', 'Answer template']

    ax.stackplot(layers, dist.T, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Attention fraction", fontsize=10)
    ax.set_title(f"Attention Distribution — {base_name}", fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{base_name}_attn_distribution.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")

    return dist


# ============================================================
# Per-sample Question Token → Frame
# ============================================================

def _visualize_per_sample_question2frame(cf, metadata, predicted, output_dir, base_name, fmt="png"):
    """
    Per-sample: each question token's attention to each frame, per layer + layer average.
    Saves both per-layer grid and averaged version.
    """
    q2f = cf["question_to_frame"]  # (L, Q, F)
    q_tokens = cf.get("question_tokens", [])
    num_frames = metadata.get("num_frames", q2f.shape[-1] if len(q2f) > 0 else 0)

    if len(q2f) == 0 or num_frames == 0:
        return

    n_layers = q2f.shape[0]
    max_q = min(q2f.shape[1], 50)
    q_display = [t.strip().replace('\n', '\\n')[:12] for t in q_tokens[:max_q]]
    if not q_display:
        q_display = [f"q{i}" for i in range(max_q)]
    frame_labels = [f"F{i}" for i in range(num_frames)]

    question = metadata.get("question", "")[:50]

    # --- 1. Layer average (single heatmap) ---
    avg_q2f = q2f.mean(axis=0)[:max_q]  # (Q, F)

    fig, ax = plt.subplots(figsize=(max(4, num_frames * 0.6), max(3, max_q * 0.2)))
    im = ax.imshow(avg_q2f, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=7)
    ax.set_yticks(range(max_q)); ax.set_yticklabels(q_display, fontsize=6)
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_title(f"Q-Token→Frame (avg) | {question}.. | Pred: {predicted}", fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{base_name}_qtoken2frame_avg.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")

    # --- 2. Per-layer grid (selected layers) ---
    if n_layers <= 8:
        show_layers = list(range(n_layers))
    else:
        step = n_layers // 6
        show_layers = list(range(0, n_layers, step))[:6]

    n_panels = len(show_layers)
    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, max(3, max_q * 0.18) * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, layer_idx in enumerate(show_layers):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        layer_q2f = q2f[layer_idx][:max_q]
        ax.imshow(layer_q2f, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=6)
        ax.set_yticks(range(max_q)); ax.set_yticklabels(q_display, fontsize=5)
        ax.set_title(f"Layer {layer_idx}", fontsize=8)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle(f"Q-Token→Frame (per layer) | {question}.. | Pred: {predicted}",
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, f"{base_name}_qtoken2frame_layers.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ============================================================
# Cross-frame Interaction Analysis
# ============================================================

def visualize_cross_frame(data, output_dir, base_name, fmt="png", include_layers=None):
    """
    Visualize cross-frame attention interactions:
      1. Frame↔Frame attention matrix (per layer grid + average)
      2. Answer→Frame attention (per layer line plot)
      3. Question→Frame attention (heatmap per layer)
    """
    metadata = data.get("metadata", {})
    num_frames = metadata.get("num_frames", None)
    tpf = metadata.get("tokens_per_frame", None)
    inter = metadata.get("inter_frame_tokens",
                         1 if metadata.get("include_newline", True) else 0)

    if not num_frames or num_frames <= 1:
        print("  [SKIP cross_frame] Not a multi-frame video")
        return

    # Use precomputed if available, else compute from raw
    if "cross_frame" in data:
        result = data["cross_frame"]
    elif "attentions_raw" in data:
        raw = [a.float() for a in data["attentions_raw"]]
        token_labels = data.get("tokens_full", data["tokens_collapsed"])
        result = compute_cross_frame_analysis(
            raw, token_labels, data["image_token_range"],
            num_frames, tpf, inter,
        )
    else:
        print("  [SKIP cross_frame] No cross_frame data — re-extract")
        return

    n_layers = result["num_valid_layers"]
    if n_layers == 0:
        print("  [SKIP cross_frame] All layers have NaN")
        return

    f2f = result["frame_to_frame"]     # (L, F, F)
    a2f = result["answer_to_frame"]    # (L, F)
    q2f = result["question_to_frame"]  # (L, Q, F)
    q_tokens = result["question_tokens"]

    question = metadata.get("question", "")[:60]
    predicted = data.get("predicted_token", "")

    # ---- 1. Frame-to-Frame: average + selected layers ----
    if include_layers is None:
        if n_layers <= 8:
            show_layers = list(range(n_layers))
        else:
            step = n_layers // 6
            show_layers = list(range(0, n_layers, step))[:6]
    else:
        show_layers = [l for l in include_layers if l < n_layers]

    n_panels = len(show_layers) + 1  # +1 for average
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    frame_labels = [f"F{i}" for i in range(num_frames)]

    # Average
    r, c = 0, 0
    ax = axes[r, c]
    im = ax.imshow(f2f.mean(axis=0), cmap='Blues', vmin=0)
    ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=7)
    ax.set_yticks(range(num_frames)); ax.set_yticklabels(frame_labels, fontsize=7)
    ax.set_title("Avg all layers", fontsize=9, fontweight='bold')
    ax.set_xlabel("Key (attended to)", fontsize=7)
    ax.set_ylabel("Query (from)", fontsize=7)

    for idx, layer_idx in enumerate(show_layers):
        r, c = divmod(idx + 1, cols)
        ax = axes[r, c]
        im = ax.imshow(f2f[layer_idx], cmap='Blues', vmin=0)
        ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=7)
        ax.set_yticks(range(num_frames)); ax.set_yticklabels(frame_labels, fontsize=7)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle(f"Frame↔Frame Attention | {question}.. | Pred: {predicted}", fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, f"{base_name}_frame2frame.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")

    # ---- 2. Answer→Frame per layer ----
    fig, ax = plt.subplots(figsize=(max(6, num_frames * 0.8), 4))
    im = ax.imshow(a2f, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=8)
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    ax.set_title(f"Answer→Frame Attention (per layer) | Pred: {predicted}", fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{base_name}_answer2frame.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")

    # ---- 3. Question→Frame per layer (select a few layers) ----
    # Truncate question tokens for readability
    max_q = min(len(q_tokens), 40)
    q_display = [t.strip().replace('\n', '\\n')[:10] for t in q_tokens[:max_q]]

    sel_layers = show_layers[:min(6, len(show_layers))]
    n_sel = len(sel_layers) + 1  # +1 for avg
    fig, axes = plt.subplots(1, n_sel, figsize=(3.5 * n_sel, max(3, max_q * 0.22)))

    if n_sel == 1:
        axes = [axes]

    # Average
    avg_q2f = q2f.mean(axis=0)[:max_q]
    axes[0].imshow(avg_q2f, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    axes[0].set_xticks(range(num_frames)); axes[0].set_xticklabels(frame_labels, fontsize=7)
    axes[0].set_yticks(range(max_q)); axes[0].set_yticklabels(q_display, fontsize=6)
    axes[0].set_title("Avg", fontsize=9, fontweight='bold')
    axes[0].set_xlabel("Frame", fontsize=8)

    for idx, layer_idx in enumerate(sel_layers):
        ax = axes[idx + 1]
        q2f_layer = q2f[layer_idx][:max_q]
        ax.imshow(q2f_layer, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=7)
        ax.set_yticks(range(max_q)); ax.set_yticklabels(q_display, fontsize=6)
        ax.set_title(f"L{layer_idx}", fontsize=9)

    fig.suptitle(f"Question→Frame Attention | {question}..", fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, f"{base_name}_question2frame.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ============================================================
# Interactive HTML (bertviz-style, d3.js)
# ============================================================

def visualize_interactive(attentions, tokens, output_dir, base_name,
                          include_layers=None, metadata=None):
    """
    Generate interactive bertviz-style HTML with d3.js.
    Token hover → attention lines, layer selector, head checkboxes.
    """
    import json as _json

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    if include_layers is None:
        if num_layers <= 12:
            include_layers = list(range(num_layers))
        else:
            step = num_layers // 8
            include_layers = list(range(0, num_layers, step))[:8]

    # Format: (filtered_layers, heads, seq, seq)
    filtered = []
    for idx in include_layers:
        filtered.append(attentions[idx].squeeze(0).float())  # (H, seq, seq)
    attn_tensor = torch.stack(filtered)  # (L, H, seq, seq)

    tokens_clean = [t.replace('\u0120', ' ').replace('\u2581', ' ').replace('</w>', '').replace('\n', '\\n')
                    for t in tokens]

    attn_data = [{
        'name': None,
        'attn': attn_tensor.tolist(),
        'left_text': tokens_clean,
        'right_text': tokens_clean,
    }]

    vis_id = 'bertviz-head'
    params = {
        'attention': attn_data,
        'default_filter': "0",
        'root_div_id': vis_id,
        'layer': None,
        'heads': None,
        'include_layers': include_layers,
    }

    question = metadata.get("question", "")[:80] if metadata else ""
    predicted = metadata.get("predicted_token", "") if metadata else ""
    title = f"Interactive Attention — Q: {question}.. | Pred: {predicted}"

    params_json = _json.dumps(params)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min.js"></script>
<style>
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 20px; }}
h2 {{ color: #333; font-size: 16px; }}
.info {{ color: #666; font-size: 13px; margin-bottom: 10px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div class="info">Tokens: {len(tokens)} | Layers shown: {len(include_layers)} | Heads: {num_heads}</div>
<div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
    <span style="user-select:none">
        Layer: <select id="layer"></select>
    </span>
    <div id='vis'></div>
</div>
<script>
(function() {{
    const params = {params_json};
    const TEXT_SIZE = 15;
    const BOXWIDTH = 110;
    const BOXHEIGHT = 22.5;
    const MATRIX_WIDTH = 115;
    const CHECKBOX_SIZE = 20;
    const TEXT_TOP = 30;

    let headColors = d3.scaleOrdinal(d3.schemeCategory10);
    let config = {{}};
    initialize();
    renderVis();

    function initialize() {{
        config.attention = params['attention'];
        config.filter = params['default_filter'];
        config.rootDivId = params['root_div_id'];
        config.nLayers = config.attention[config.filter]['attn'].length;
        config.nHeads = config.attention[config.filter]['attn'][0].length;
        config.layers = params['include_layers'];
        config.headVis = new Array(config.nHeads).fill(true);
        config.initialTextLength = config.attention[config.filter].right_text.length;
        config.layer_seq = 0;
        config.layer = config.layers[0];

        let layerEl = document.querySelector('#' + config.rootDivId + ' #layer');
        for (const layer of config.layers) {{
            let opt = document.createElement('option');
            opt.value = layer; opt.text = layer;
            layerEl.appendChild(opt);
        }}
        layerEl.value = config.layer;
        layerEl.addEventListener('change', function(e) {{
            config.layer = +e.target.value;
            config.layer_seq = config.layers.findIndex(l => config.layer === l);
            renderVis();
        }});
    }}

    function renderVis() {{
        const attnData = config.attention[config.filter];
        const leftText = attnData.left_text;
        const rightText = attnData.right_text;
        const layerAttention = attnData.attn[config.layer_seq];

        let visEl = document.querySelector('#' + config.rootDivId + ' #vis');
        visEl.innerHTML = '';

        const height = Math.max(leftText.length, rightText.length) * BOXHEIGHT + TEXT_TOP;
        const svg = d3.select('#' + config.rootDivId + ' #vis')
            .append('svg').attr("width", "100%").attr("height", height + "px");

        renderText(svg, leftText, true, layerAttention, 0);
        renderText(svg, rightText, false, layerAttention, MATRIX_WIDTH + BOXWIDTH);
        renderAttention(svg, layerAttention);
        drawCheckboxes(0, svg);
    }}

    function renderText(svg, text, isLeft, attention, leftPos) {{
        const textContainer = svg.append("svg:g").attr("id", isLeft ? "left" : "right");
        textContainer.append("g").classed("attentionBoxes", true)
            .selectAll("g").data(attention).enter().append("g")
            .attr("head-index", (d, i) => i)
            .selectAll("rect").data(d => isLeft ? d : transpose(d)).enter()
            .append("rect")
            .attr("x", function() {{ return leftPos + boxOffsets(+this.parentNode.getAttribute("head-index")); }})
            .attr("y", BOXHEIGHT).attr("width", BOXWIDTH / activeHeads())
            .attr("height", BOXHEIGHT)
            .attr("fill", function() {{ return headColors(+this.parentNode.getAttribute("head-index")); }})
            .style("opacity", 0.0);

        const tokenContainer = textContainer.append("g").selectAll("g").data(text).enter().append("g");
        tokenContainer.append("rect").classed("background", true).style("opacity", 0.0)
            .attr("fill", "lightgray").attr("x", leftPos)
            .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT)
            .attr("width", BOXWIDTH).attr("height", BOXHEIGHT);

        const textEl = tokenContainer.append("text").text(d => d)
            .attr("font-size", TEXT_SIZE + "px").style("cursor", "default")
            .style("-webkit-user-select", "none").attr("x", leftPos)
            .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT);

        if (isLeft) {{ textEl.style("text-anchor", "end").attr("dx", BOXWIDTH - 0.5 * TEXT_SIZE).attr("dy", TEXT_SIZE); }}
        else {{ textEl.style("text-anchor", "start").attr("dx", 0.5 * TEXT_SIZE).attr("dy", TEXT_SIZE); }}

        tokenContainer.on("mouseover", function(d, index) {{
            textContainer.selectAll(".background").style("opacity", (d, i) => i === index ? 1.0 : 0.0);
            svg.select("#attention").selectAll("line[visibility='visible']").attr("visibility", null);
            svg.select("#attention").attr("visibility", "hidden");
            if (isLeft) {{ svg.select("#attention").selectAll("line[left-token-index='" + index + "']").attr("visibility", "visible"); }}
            else {{ svg.select("#attention").selectAll("line[right-token-index='" + index + "']").attr("visibility", "visible"); }}

            const id = isLeft ? "right" : "left";
            const lp = isLeft ? MATRIX_WIDTH + BOXWIDTH : 0;
            svg.select("#" + id).selectAll(".attentionBoxes").selectAll("g")
                .attr("head-index", (d, i) => i).selectAll("rect")
                .attr("x", function() {{ return lp + boxOffsets(+this.parentNode.getAttribute("head-index")); }})
                .attr("y", (d, i) => TEXT_TOP + i * BOXHEIGHT)
                .attr("width", BOXWIDTH / activeHeads()).attr("height", BOXHEIGHT)
                .style("opacity", function(d) {{
                    const hi = +this.parentNode.getAttribute("head-index");
                    return (config.headVis[hi] && d) ? d[index] : 0.0;
                }});
        }});

        textContainer.on("mouseleave", function() {{
            d3.select(this).selectAll(".background").style("opacity", 0.0);
            svg.select("#attention").selectAll("line[visibility='visible']").attr("visibility", null);
            svg.select("#attention").attr("visibility", "visible");
            svg.selectAll(".attentionBoxes").selectAll("g").selectAll("rect").style("opacity", 0.0);
        }});
    }}

    function renderAttention(svg, attention) {{
        svg.select("#attention").remove();
        svg.append("g").attr("id", "attention")
            .selectAll(".headAttention").data(attention).enter()
            .append("g").classed("headAttention", true).attr("head-index", (d, i) => i)
            .selectAll(".tokenAttention").data(d => d).enter()
            .append("g").classed("tokenAttention", true).attr("left-token-index", (d, i) => i)
            .selectAll("line").data(d => d).enter()
            .append("line")
            .attr("x1", BOXWIDTH)
            .attr("y1", function() {{ return TEXT_TOP + (+this.parentNode.getAttribute("left-token-index")) * BOXHEIGHT + BOXHEIGHT / 2; }})
            .attr("x2", BOXWIDTH + MATRIX_WIDTH)
            .attr("y2", (d, ri) => TEXT_TOP + ri * BOXHEIGHT + BOXHEIGHT / 2)
            .attr("stroke-width", 2)
            .attr("stroke", function() {{ return headColors(+this.parentNode.parentNode.getAttribute("head-index")); }})
            .attr("left-token-index", function() {{ return +this.parentNode.getAttribute("left-token-index"); }})
            .attr("right-token-index", (d, i) => i);
        updateAttention(svg);
    }}

    function updateAttention(svg) {{
        svg.select("#attention").selectAll("line")
            .attr("stroke-opacity", function(d) {{
                const hi = +this.parentNode.parentNode.getAttribute("head-index");
                return config.headVis[hi] ? d / activeHeads() : 0.0;
            }});
    }}

    function boxOffsets(i) {{
        return config.headVis.reduce((acc, val, cur) => val && cur < i ? acc + 1 : acc, 0) * (BOXWIDTH / activeHeads());
    }}
    function activeHeads() {{ return config.headVis.reduce((a, v) => v ? a + 1 : a, 0); }}

    function drawCheckboxes(top, svg) {{
        const cc = svg.append("g");
        const cb = cc.selectAll("rect").data(config.headVis).enter().append("rect")
            .attr("fill", (d, i) => headColors(i))
            .attr("x", (d, i) => i * CHECKBOX_SIZE).attr("y", top)
            .attr("width", CHECKBOX_SIZE).attr("height", CHECKBOX_SIZE);

        function upd() {{ cc.selectAll("rect").data(config.headVis).attr("fill", (d, i) => d ? headColors(i) : lighten(headColors(i))); }}
        upd();
        cb.on("click", function(d, i) {{
            if (config.headVis[i] && activeHeads() === 1) return;
            config.headVis[i] = !config.headVis[i]; upd(); updateAttention(svg);
        }});
        cb.on("dblclick", function(d, i) {{
            if (config.headVis[i] && activeHeads() === 1) config.headVis = new Array(config.nHeads).fill(true);
            else {{ config.headVis = new Array(config.nHeads).fill(false); config.headVis[i] = true; }}
            upd(); updateAttention(svg);
        }});
    }}

    function lighten(color) {{ const c = d3.hsl(color); c.l += (1 - c.l) * 0.6; c.s -= (1 - c.l) * 0.6; return c; }}
    function transpose(mat) {{ return mat[0].map((_, i) => mat.map(row => row[i])); }}
}})();
</script>
</body>
</html>"""

    path = os.path.join(output_dir, f"{base_name}_interactive.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [SAVED] {path}")


# ============================================================
# CLI
# ============================================================

def load_and_visualize(attn_path, output_dir, mode, include_layers, top_k, threshold, fmt, cols,
                       heatmap_method="rollout", cmap="inferno"):
    """Load a .pt file and generate visualization images."""
    print(f"[INFO] Loading: {attn_path}")
    data = torch.load(attn_path, map_location="cpu", weights_only=False)

    attentions = [a.float() for a in data["attentions_collapsed"]]
    tokens = data["tokens_collapsed"]
    metadata = data.get("metadata", {})

    source = metadata.get("source", "")
    base_name = os.path.splitext(os.path.basename(attn_path))[0]

    print(f"  Tokens: {len(tokens)} | Layers: {len(attentions)} | "
          f"Heads: {attentions[0].shape[1]} | Type: {metadata.get('type', '?')}")

    os.makedirs(output_dir, exist_ok=True)

    def _subdir(name):
        d = os.path.join(output_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    if mode in ("per_layer", "all"):
        visualize_per_layer(attentions, tokens, _subdir("per_layer"), base_name,
                            include_layers=include_layers, top_k=top_k,
                            threshold=threshold, fmt=fmt)

    if mode in ("grid", "all"):
        visualize_grid(attentions, tokens, _subdir("grid"), base_name,
                       include_layers=include_layers, top_k=top_k,
                       threshold=threshold, fmt=fmt, cols=cols)

    if mode in ("summary", "all"):
        visualize_summary(attentions, tokens, _subdir("summary"), base_name,
                          top_k=top_k, threshold=threshold, fmt=fmt)

    if mode in ("heatmap", "all"):
        visualize_heatmap(data, _subdir("heatmap"), base_name,
                          method=heatmap_method, fmt=fmt, cmap=cmap)
        if mode == "all" and heatmap_method == "rollout":
            visualize_heatmap(data, _subdir("heatmap"), base_name,
                              method="avg", fmt=fmt, cmap=cmap)

    if mode in ("cross_frame", "all"):
        visualize_cross_frame(data, _subdir("cross_frame"), base_name,
                              fmt=fmt, include_layers=include_layers)

    if mode == "all":
        visualize_attention_distribution(
            attentions, tokens, data["image_token_range"],
            _subdir("attention_distribution"), base_name, fmt=fmt)

    if mode in ("interactive", "all"):
        visualize_interactive(attentions, tokens, _subdir("interactive"), base_name,
                              include_layers=include_layers,
                              metadata={**metadata, "predicted_token": data.get("predicted_token", "")})


def generate_html_report(output_dir, title="Attention Visualization"):
    """
    Scan output_dir (with type subdirectories) and create an HTML gallery.
    Structure: output_dir/{grid,summary,heatmap,cross_frame,interactive,per_layer}/
    """
    import re

    image_exts = (".png", ".jpg", ".pdf", ".svg")
    html_exts = (".html",)

    # Collect files from subdirectories
    type_order = ["attention_distribution", "heatmap", "cross_frame", "question_token_frame", "grid", "summary", "per_layer", "interactive"]
    type_labels = {
        "attention_distribution": "Attention Distribution (Sink)",
        "heatmap": "Heatmap Overlay",
        "cross_frame": "Cross-Frame Analysis (Aggregated)",
        "question_token_frame": "Question Token → Frame (Per Sample)",
        "grid": "Token Attention (Layer Grid)",
        "summary": "Token Attention (Summary)",
        "per_layer": "Token Attention (Per Layer)",
        "interactive": "Interactive (HTML)",
    }

    sections = {}  # type_name → list of (relative_path, filename)
    for type_name in type_order:
        subdir = os.path.join(output_dir, type_name)
        if not os.path.isdir(subdir):
            continue
        files = sorted(os.listdir(subdir))
        entries = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in image_exts or ext in html_exts:
                entries.append((f"{type_name}/{f}", f))
        if entries:
            sections[type_name] = entries

    if not sections:
        return

    total_files = sum(len(v) for v in sections.values())

    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 20px; background: #fafafa; }}
h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 8px; }}
h2 {{ color: #1976D2; margin-top: 24px; }}
.section {{ background: white; margin: 12px 0; padding: 16px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.images {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.images img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
.img-card {{ flex: 1 1 400px; max-width: 900px; }}
.img-card .label {{ font-size: 11px; color: #888; margin-top: 4px; font-family: monospace; }}
a.html-link {{ display: inline-block; padding: 6px 14px; margin: 4px; background: #e3f2fd;
               color: #1565c0; border-radius: 4px; text-decoration: none; font-size: 13px; }}
a.html-link:hover {{ background: #bbdefb; }}
details summary {{ cursor: pointer; font-weight: bold; }}
nav {{ margin-bottom: 16px; }}
nav a {{ margin-right: 12px; color: #1976D2; text-decoration: none; font-size: 14px; }}
nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p style="color:#666">{total_files} files in {len(sections)} categories</p>
<nav>"""]

    for type_name in type_order:
        if type_name in sections:
            label = type_labels.get(type_name, type_name)
            html_parts.append(f'<a href="#{type_name}">{label}</a>')

    html_parts.append("</nav>")

    # Link to correct/incorrect sub-reports if they exist
    for sub in ["correct", "incorrect"]:
        sub_index = os.path.join(output_dir, sub, "index.html")
        if os.path.exists(sub_index):
            color = "#4CAF50" if sub == "correct" else "#f44336"
            count = len([f for f in os.listdir(os.path.join(output_dir, sub))
                        if os.path.isdir(os.path.join(output_dir, sub, f))])
            html_parts.append(
                f'<a href="{sub}/index.html" style="display:inline-block;padding:10px 20px;'
                f'margin:4px;background:{color};color:white;border-radius:6px;'
                f'text-decoration:none;font-weight:bold;font-size:15px">'
                f'{sub.upper()} →</a>'
            )

    for type_name in type_order:
        if type_name not in sections:
            continue
        entries = sections[type_name]
        label = type_labels.get(type_name, type_name)

        html_parts.append(f"""<div class="section" id="{type_name}">
<details open>
<summary><h2 style="display:inline; margin:0">{label} ({len(entries)})</h2></summary>
<div class="images">""")

        for rel_path, filename in entries:
            ext = os.path.splitext(filename)[1].lower()
            if ext in html_exts:
                html_parts.append(f'<a class="html-link" href="{rel_path}" target="_blank">{filename}</a>')
            else:
                short = re.sub(r'^.*?_attn_', '', filename).rsplit('.', 1)[0] or filename
                html_parts.append(f"""<div class="img-card">
<img src="{rel_path}" loading="lazy">
<div class="label">{short}</div>
</div>""")

        html_parts.append("</div></details></div>")

    html_parts.append("</body></html>")

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"[SAVED] {html_path}")


def _check_correct(data):
    """
    Check if prediction matches ground-truth answer.
    Returns True (correct), False (incorrect), or None (no answer available).
    """
    metadata = data.get("metadata", {})
    answer = metadata.get("answer", "")
    predicted = data.get("predicted_token", "")

    if not answer:
        return None

    answer = answer.strip().upper()
    predicted = predicted.strip().upper()

    if predicted == answer:
        return True
    if len(answer) == 1 and answer.isalpha():
        if answer in predicted:
            return True
    return False


def _collect_cross_frame(data):
    """Get cross_frame analysis from precomputed data, or compute from raw if available."""
    # Prefer precomputed (fast)
    if "cross_frame" in data:
        return data["cross_frame"]

    # Fallback: compute from raw attentions (slow, for old .pt files)
    metadata = data.get("metadata", {})
    num_frames = metadata.get("num_frames", None)
    tpf = metadata.get("tokens_per_frame", None)
    inter = metadata.get("inter_frame_tokens",
                         1 if metadata.get("include_newline", True) else 0)

    if not num_frames or num_frames <= 1 or "attentions_raw" not in data:
        return None

    raw = [a.float() for a in data["attentions_raw"]]
    token_labels = data.get("tokens_full", data["tokens_collapsed"])

    return compute_cross_frame_analysis(
        raw, token_labels, data["image_token_range"],
        num_frames, tpf, inter,
    )


def visualize_aggregated_cross_frame(results, output_dir, label, num_frames,
                                      fmt="png", include_layers=None):
    """
    Visualize sample-averaged cross-frame analysis.

    Args:
        results: list of cross_frame result dicts (from compute_cross_frame_analysis)
        output_dir: save directory
        label: prefix label (e.g. "correct", "incorrect", "all")
        num_frames: number of frames
    """
    if not results:
        return

    # Average across samples (only matching num_frames)
    f2f_list = [r["frame_to_frame"] for r in results if r["frame_to_frame"].shape[1] == num_frames]
    a2f_list = [r["answer_to_frame"] for r in results if r["answer_to_frame"].shape[1] == num_frames]

    if not f2f_list:
        return

    # Align layer counts (take min)
    min_layers = min(r.shape[0] for r in f2f_list)
    f2f_avg = np.mean([r[:min_layers] for r in f2f_list], axis=0)  # (L, F, F)
    a2f_avg = np.mean([r[:min_layers] for r in a2f_list], axis=0)  # (L, F)

    n_samples = len(f2f_list)
    n_layers = min_layers

    os.makedirs(output_dir, exist_ok=True)
    frame_labels = [f"F{i}" for i in range(num_frames)]

    if include_layers is None:
        if n_layers <= 8:
            show_layers = list(range(n_layers))
        else:
            step = n_layers // 6
            show_layers = list(range(0, n_layers, step))[:6]
    else:
        show_layers = [l for l in include_layers if l < n_layers]

    # ---- 1. Frame↔Frame (avg + per layer) ----
    n_panels = len(show_layers) + 1
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    ax = axes[0, 0]
    ax.imshow(f2f_avg.mean(axis=0), cmap='Blues', vmin=0)
    ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=7)
    ax.set_yticks(range(num_frames)); ax.set_yticklabels(frame_labels, fontsize=7)
    ax.set_title("Avg all layers", fontsize=9, fontweight='bold')
    ax.set_xlabel("Key", fontsize=7); ax.set_ylabel("Query", fontsize=7)

    for idx, layer_idx in enumerate(show_layers):
        r, c = divmod(idx + 1, cols)
        ax = axes[r, c]
        ax.imshow(f2f_avg[layer_idx], cmap='Blues', vmin=0)
        ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=7)
        ax.set_yticks(range(num_frames)); ax.set_yticklabels(frame_labels, fontsize=7)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle(f"Frame↔Frame — {label} (n={n_samples})", fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, f"agg_{label}_frame2frame.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")

    # ---- 2. Answer→Frame (layer × frame heatmap) ----
    fig, ax = plt.subplots(figsize=(max(6, num_frames * 0.8), 4))
    im = ax.imshow(a2f_avg, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel("Frame", fontsize=10); ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=8)
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    ax.set_title(f"Answer→Frame — {label} (n={n_samples})", fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(output_dir, f"agg_{label}_answer2frame.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")

    # ---- 3. Question→Frame (layer × frame, averaged across question tokens + samples) ----
    q2f_list = [r["question_to_frame"] for r in results
                if r.get("question_to_frame") is not None and len(r["question_to_frame"]) > 0
                and r["question_to_frame"].shape[-1] == num_frames]

    if q2f_list:
        # Average across question tokens first (per sample), then across samples
        q2f_per_sample = []
        for q2f in q2f_list:
            # q2f: (L, Q, F) — average over Q dimension → (L, F)
            q2f_sample = q2f[:min_layers].mean(axis=1)  # (L, F)
            q2f_per_sample.append(q2f_sample)
        q2f_avg = np.mean(q2f_per_sample, axis=0)  # (L, F)

        fig, ax = plt.subplots(figsize=(max(6, num_frames * 0.8), 4))
        im = ax.imshow(q2f_avg, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        ax.set_xlabel("Frame", fontsize=10); ax.set_ylabel("Layer", fontsize=10)
        ax.set_xticks(range(num_frames)); ax.set_xticklabels(frame_labels, fontsize=8)
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
        ax.set_title(f"Question→Frame — {label} (n={n_samples})", fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = os.path.join(output_dir, f"agg_{label}_question2frame.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {path}")


def _render_per_sample(pt_path, sub_dir, base_name, mode, heatmap_method, fmt, cmap):
    """
    Worker function for per-sample visualization (runs in subprocess).
    Loads .pt, renders heatmap and/or cross_frame, saves images.
    """
    import matplotlib
    matplotlib.use("Agg")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    metadata = data.get("metadata", {})

    if mode in ("heatmap", "all"):
        heatmap_dir = os.path.join(sub_dir, "heatmap")
        os.makedirs(heatmap_dir, exist_ok=True)
        visualize_heatmap(data, heatmap_dir, base_name,
                          method=heatmap_method, fmt=fmt, cmap=cmap)
        if mode == "all":
            visualize_heatmap(data, heatmap_dir, base_name,
                              method="avg", fmt=fmt, cmap=cmap)

    if mode in ("cross_frame", "all"):
        cf = data.get("cross_frame", None)
        if cf and cf.get("question_to_frame") is not None and len(cf["question_to_frame"]) > 0:
            q2f_dir = os.path.join(sub_dir, "question_token_frame")
            os.makedirs(q2f_dir, exist_ok=True)
            _visualize_per_sample_question2frame(
                cf, metadata, data.get("predicted_token", ""),
                q2f_dir, base_name, fmt=fmt)

    del data


def main():
    parser = argparse.ArgumentParser(
        description="Attention visualization with correct/incorrect separation + aggregation."
    )

    parser.add_argument("--attn_path", type=str, default=None,
                        help="Path to single .pt attention file")
    parser.add_argument("--attn_dir", type=str, default=None,
                        help="Directory of .pt files (batch mode)")
    parser.add_argument("--output_dir", type=str, default="output/attention_vis")

    parser.add_argument("--mode", type=str, default="grid",
                        choices=["per_layer", "grid", "summary", "heatmap", "cross_frame", "interactive", "all"],
                        help="cross_frame: frame↔frame, answer→frame, question→frame analysis")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to include (default: evenly spaced 8)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Only show top-K attention lines per token")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Min attention weight to draw (default: 0.01)")
    parser.add_argument("--fmt", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output format")
    parser.add_argument("--cols", type=int, default=4,
                        help="Columns in grid mode")

    # Heatmap options
    parser.add_argument("--heatmap_method", type=str, default="rollout",
                        choices=["rollout", "avg"],
                        help="rollout: attention rollout across layers, avg: simple layer average")
    parser.add_argument("--cmap", type=str, default="inferno",
                        help="Colormap for heatmap (inferno, viridis, hot, jet, etc.)")

    # Performance
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Parallel workers for visualization (default: 4)")

    args = parser.parse_args()

    if not args.attn_path and not args.attn_dir:
        parser.error("--attn_path or --attn_dir required")

    if args.attn_path:
        load_and_visualize(args.attn_path, args.output_dir, args.mode,
                           args.layers, args.top_k, args.threshold, args.fmt, args.cols,
                           args.heatmap_method, args.cmap)
        generate_html_report(args.output_dir)

    elif args.attn_dir:
        pt_files = sorted(glob.glob(os.path.join(args.attn_dir, "*_attn.pt")))
        if not pt_files:
            print(f"[WARN] No *_attn.pt files in {args.attn_dir}")
            return
        print(f"[INFO] Found {len(pt_files)} files (workers={args.num_workers})")

        # Collectors for aggregation
        correct_cf, incorrect_cf, all_cf = [], [], []
        num_frames_ref = None
        correct_attn, incorrect_attn, all_attn = [], [], []
        tokens_ref = None
        n_correct, n_incorrect, n_unknown = 0, 0, 0

        # --- Phase 1: Load data + collect for aggregation + queue vis tasks ---
        # Per-sample vis tasks: (pt_path, sub_dir, base_name) for parallel rendering
        per_sample_tasks = []  # (pt_path, sub_dir, base_name)

        for pt_file in pt_files:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            is_correct = _check_correct(data)
            metadata = data.get("metadata", {})

            if is_correct is True:
                sub_dir = os.path.join(args.output_dir, "correct")
                n_correct += 1
            elif is_correct is False:
                sub_dir = os.path.join(args.output_dir, "incorrect")
                n_incorrect += 1
            else:
                sub_dir = args.output_dir
                n_unknown += 1

            base_name = os.path.splitext(os.path.basename(pt_file))[0]

            # Queue for parallel per-sample vis
            needs_vis = (args.mode in ("heatmap", "cross_frame", "all"))
            if needs_vis:
                per_sample_tasks.append((pt_file, sub_dir, base_name))

            # Collect collapsed attention for aggregation
            if args.mode in ("grid", "summary", "per_layer", "all"):
                collapsed = [a.float() for a in data["attentions_collapsed"]]
                tokens = data["tokens_collapsed"]

                if tokens_ref is None:
                    tokens_ref = tokens

                if len(tokens) == len(tokens_ref):
                    all_attn.append(collapsed)
                    if is_correct is True:
                        correct_attn.append(collapsed)
                    elif is_correct is False:
                        incorrect_attn.append(collapsed)

            # Collect cross_frame for aggregation
            if args.mode in ("cross_frame", "all"):
                cf = _collect_cross_frame(data)
                if cf and cf["num_valid_layers"] > 0:
                    nf = metadata.get("num_frames", 0)
                    if num_frames_ref is None:
                        num_frames_ref = nf
                    if nf == num_frames_ref:
                        all_cf.append(cf)
                        if is_correct is True:
                            correct_cf.append(cf)
                        elif is_correct is False:
                            incorrect_cf.append(cf)

            del data

        # --- Phase 2: Per-sample visualizations in parallel (ProcessPoolExecutor) ---
        if per_sample_tasks:
            print(f"[INFO] Running {len(per_sample_tasks)} per-sample visualizations "
                  f"({args.num_workers} workers)...")
            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                futures = {}
                for pt_path, sub_dir, base_name in per_sample_tasks:
                    fut = pool.submit(
                        _render_per_sample, pt_path, sub_dir, base_name,
                        args.mode, args.heatmap_method, args.fmt, args.cmap,
                    )
                    futures[fut] = base_name
                n_done = 0
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"  [WARN] {futures[fut]}: {e}")
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"  [{n_done}/{len(per_sample_tasks)}] done")
            print(f"  [{len(per_sample_tasks)}/{len(per_sample_tasks)}] per-sample visualization complete")

        print(f"\n[SUMMARY] correct: {n_correct}, incorrect: {n_incorrect}, no_answer: {n_unknown}")

        # --- Aggregated grid/summary ---
        def _avg_attentions(attn_list):
            """Average collapsed attentions across samples: list of [list of (1,H,S,S)] → [list of (1,H,S,S)]."""
            n = len(attn_list)
            n_layers = len(attn_list[0])
            averaged = []
            for l in range(n_layers):
                stacked = torch.stack([s[l] for s in attn_list])  # (n, 1, H, S, S)
                averaged.append(stacked.mean(dim=0))
            return averaged

        # --- Phase 3: Aggregated visualizations (sequential, few tasks) ---
        if args.mode in ("grid", "summary", "per_layer", "all") and tokens_ref:
            groups = [
                ("all",       all_attn,       args.output_dir),
                ("correct",   correct_attn,   os.path.join(args.output_dir, "correct")),
                ("incorrect", incorrect_attn, os.path.join(args.output_dir, "incorrect")),
            ]
            for label, attn_list, out_dir in groups:
                if not attn_list:
                    continue
                avg = _avg_attentions(attn_list)
                print(f"  [AGG] {label}: {len(attn_list)} samples averaged")
                if args.mode in ("grid", "all"):
                    grid_dir = os.path.join(out_dir, "grid")
                    os.makedirs(grid_dir, exist_ok=True)
                    visualize_grid(avg, tokens_ref, grid_dir, f"agg_{label}",
                                   include_layers=args.layers, top_k=args.top_k,
                                   threshold=args.threshold, fmt=args.fmt, cols=args.cols)
                if args.mode in ("summary", "all"):
                    sum_dir = os.path.join(out_dir, "summary")
                    os.makedirs(sum_dir, exist_ok=True)
                    visualize_summary(avg, tokens_ref, sum_dir, f"agg_{label}",
                                      top_k=args.top_k, threshold=args.threshold, fmt=args.fmt)
                if args.mode in ("per_layer", "all"):
                    pl_dir = os.path.join(out_dir, "per_layer")
                    os.makedirs(pl_dir, exist_ok=True)
                    visualize_per_layer(avg, tokens_ref, pl_dir, f"agg_{label}",
                                        include_layers=args.layers, top_k=args.top_k,
                                        threshold=args.threshold, fmt=args.fmt)
                if args.mode == "all":
                    sink_dir = os.path.join(out_dir, "attention_distribution")
                    os.makedirs(sink_dir, exist_ok=True)
                    img_range_collapsed = None
                    for ti, t in enumerate(tokens_ref):
                        if t.startswith("[F") or t.startswith("[IMG"):
                            img_range_collapsed = (ti, ti)
                            break
                    if img_range_collapsed:
                        for ti in range(img_range_collapsed[0], len(tokens_ref)):
                            if tokens_ref[ti].startswith("[F") or tokens_ref[ti].startswith("[IMG"):
                                img_range_collapsed = (img_range_collapsed[0], ti + 1)
                            else:
                                break
                        visualize_attention_distribution(
                            avg, tokens_ref, img_range_collapsed,
                            sink_dir, f"agg_{label}", fmt=args.fmt)

        if args.mode in ("cross_frame", "all") and num_frames_ref:
            groups = [
                ("all",       all_cf,       args.output_dir),
                ("correct",   correct_cf,   os.path.join(args.output_dir, "correct")),
                ("incorrect", incorrect_cf, os.path.join(args.output_dir, "incorrect")),
            ]
            for label, cf_list, out_dir in groups:
                if not cf_list:
                    continue
                cf_dir = os.path.join(out_dir, "cross_frame")
                os.makedirs(cf_dir, exist_ok=True)
                visualize_aggregated_cross_frame(
                    cf_list, cf_dir, label, num_frames_ref,
                    fmt=args.fmt, include_layers=args.layers)

        # HTML reports
        for sub in ["correct", "incorrect"]:
            sub_path = os.path.join(args.output_dir, sub)
            if os.path.isdir(sub_path):
                generate_html_report(sub_path, title=f"{sub.capitalize()} Predictions")
        generate_html_report(args.output_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
