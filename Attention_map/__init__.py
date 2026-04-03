"""
Attention_map: attention visualization for VLMs (LLaVA-NeXT, Qwen3-VL).

Pipeline:
  1. extract_attention.py — Run inference, save attention matrices + frames as .pt
  2. visualize_attention.py — Generate images: token-to-token lines + heatmap overlays
"""

from Attention_map.attention_utils import (
    # LLaVA (available when llava is installed)
    hook_vision_tower,
    extract_attention,
    build_token_labels,
    collapse_vision_tokens,
    build_prompt,
    get_vision_grid_size,
    get_tokens_per_frame,
    llm_attention_rollout,
    extract_answer_vision_attention,
    attention_to_heatmap,
    split_vision_attention_by_frames,
    # Qwen3-VL (always available)
    extract_attention_qwen3vl_fast,
    build_token_labels_qwen3vl,
    get_qwen3vl_vision_token_id,
    get_qwen3vl_grid_size,
)

try:
    from Attention_map.visualize_attention import (
        draw_attention,
        visualize_per_layer,
        visualize_grid,
        visualize_summary,
        visualize_heatmap,
    )
except ImportError:
    pass  # visualization dependencies optional
