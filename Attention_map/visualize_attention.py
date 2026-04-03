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

import argparse
import glob
import os

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
        # Average across heads: (1, H, N, N) → (N, N)
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
    include_newline = metadata.get("include_newline", True)

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
            attn_on_vision, num_frames, tokens_per_frame, include_newline
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

    if mode in ("per_layer", "all"):
        visualize_per_layer(attentions, tokens, output_dir, base_name,
                            include_layers=include_layers, top_k=top_k,
                            threshold=threshold, fmt=fmt)

    if mode in ("grid", "all"):
        visualize_grid(attentions, tokens, output_dir, base_name,
                       include_layers=include_layers, top_k=top_k,
                       threshold=threshold, fmt=fmt, cols=cols)

    if mode in ("summary", "all"):
        visualize_summary(attentions, tokens, output_dir, base_name,
                          top_k=top_k, threshold=threshold, fmt=fmt)

    if mode in ("heatmap", "all"):
        visualize_heatmap(data, output_dir, base_name,
                          method=heatmap_method, fmt=fmt, cmap=cmap)
        # If rollout was requested, also show avg for comparison in "all" mode
        if mode == "all" and heatmap_method == "rollout":
            visualize_heatmap(data, output_dir, base_name,
                              method="avg", fmt=fmt, cmap=cmap)


def main():
    parser = argparse.ArgumentParser(
        description="Bertviz-style attention visualization (static images)."
    )

    parser.add_argument("--attn_path", type=str, default=None,
                        help="Path to single .pt attention file")
    parser.add_argument("--attn_dir", type=str, default=None,
                        help="Directory of .pt files (batch mode)")
    parser.add_argument("--output_dir", type=str, default="output/attention_vis")

    parser.add_argument("--mode", type=str, default="grid",
                        choices=["per_layer", "grid", "summary", "heatmap", "all"],
                        help="grid: token-to-token lines, heatmap: answer→vision on image/frames")
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

    args = parser.parse_args()

    if not args.attn_path and not args.attn_dir:
        parser.error("--attn_path or --attn_dir required")

    if args.attn_path:
        load_and_visualize(args.attn_path, args.output_dir, args.mode,
                           args.layers, args.top_k, args.threshold, args.fmt, args.cols,
                           args.heatmap_method, args.cmap)
    elif args.attn_dir:
        pt_files = sorted(glob.glob(os.path.join(args.attn_dir, "*_attn.pt")))
        if not pt_files:
            print(f"[WARN] No *_attn.pt files in {args.attn_dir}")
            return
        print(f"[INFO] Found {len(pt_files)} files")
        for pt_file in pt_files:
            load_and_visualize(pt_file, args.output_dir, args.mode,
                               args.layers, args.top_k, args.threshold, args.fmt, args.cols,
                               args.heatmap_method, args.cmap)

    print("[DONE]")


if __name__ == "__main__":
    main()
