"""
Q-K Space Analysis: Hypothesis testing for attention sink phenomenon.

Hypothesis 1: Sink token K-norm dominance
  → BOS/System K vectors have abnormally large norms, dominating QK dot product.

Hypothesis 2: Vision-Text K-vector subspace mismatch (post-RoPE)
  → Vision K vectors are angular-distant from answer Q in the actual attention space.

Counterfactual: Recompute attention with K norms normalized to 1.
  → Isolates norm effect from directional effect.

All cosine similarity and counterfactual metrics use POST-RoPE Q, K vectors,
matching what the model actually computes for attention.

Usage:
  python Attention_map/analyze_qk_space.py \
    --model_args "pretrained=...,device_map=auto" \
    --task direction_testbed_ablation_4way \
    --limit 50 --output_dir output/qk_analysis/model_name
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

torch.set_grad_enabled(False)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _import_module_direct(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dataset_loader = _import_module_direct(
    "core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py")
)
load_dataset_as_questions = _dataset_loader.load_dataset_as_questions

try:
    from llava.constants import IMAGE_TOKEN_INDEX
    _LLAVA_AVAILABLE = True
except (ImportError, Exception):
    _LLAVA_AVAILABLE = False
    IMAGE_TOKEN_INDEX = -200


# ============================================================
# Shared: compute stats from post-RoPE Q, K
# ============================================================

def _compute_stats(q, k, img_start, img_end, seq_len, head_dim, frame_ranges=None):
    """
    Compute Q-K metrics for 3 query groups: answer, question, vision.
    Optionally computes per-frame intra/cross similarity.

    Args:
        q: (n_heads, seq_len, head_dim) — post-RoPE
        k: (n_heads, seq_len, head_dim) — post-RoPE, GQA-expanded
        frame_ranges: list of (start, end) tuples for each frame (optional)

    Query groups:
        answer:   last token (generates the answer)
        question: tokens after vision (img_end:seq_len-1, excluding answer)
        vision:   vision tokens (img_start:img_end)

    For each query group, computes:
        cos_sim_{query}_{key}: mean cosine similarity to each key type
        {key}_attn_{query}_original: attention fraction (original)
        {key}_attn_{query}_knorm: attention fraction (K-normalized counterfactual)
    """
    scale = head_dim ** 0.5
    n_heads = q.shape[0]

    system_idx = list(range(0, img_start))
    vision_idx = list(range(img_start, img_end))
    question_idx = list(range(img_end, seq_len))

    def _mean(t, idx):
        return t[idx].mean().item() if idx else 0.0

    # --- H1: K norms (shared, query-independent) ---
    k_norms = k.norm(dim=-1).mean(dim=0)  # (seq_len,)
    k_unit = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    stats = {
        'k_norm_system': _mean(k_norms, system_idx),
        'k_norm_vision': _mean(k_norms, vision_idx),
        'k_norm_question': _mean(k_norms, question_idx),
    }

    # --- Per query group: cosine similarity + attention ---
    question_end = seq_len - 1  # exclude answer (last token)
    query_groups = {
        'answer': q[:, -1:, :],                                          # (H, 1, D)
        'question': q[:, img_end:question_end, :] if img_end < question_end else None,  # (H, Q, D)
        'vision': q[:, img_start:img_end, :],                            # (H, V, D)
    }

    for qname, q_group in query_groups.items():
        if q_group is None or q_group.shape[1] == 0:
            for kname in ['system', 'vision', 'question']:
                stats[f'cos_{qname}_{kname}'] = 0.0
                stats[f'attn_{qname}_{kname}_orig'] = 0.0
                stats[f'attn_{qname}_{kname}_knorm'] = 0.0
            continue

        # Mean Q across positions in group → (H, D)
        q_mean = q_group.mean(dim=1)

        # Causal mask: query group's LAST position can attend to positions 0..last_pos
        if qname == 'answer':
            last_pos = seq_len - 1
        elif qname == 'question':
            last_pos = seq_len - 2  # question ends before answer
        else:  # vision
            last_pos = img_end - 1

        # Cosine similarity: q_mean vs causally accessible K only
        cos_sim = F.cosine_similarity(
            q_mean.unsqueeze(1).expand(-1, seq_len, -1), k, dim=-1
        ).mean(dim=0)  # (seq_len,)

        # Attention (original)
        scores = torch.matmul(q_mean.unsqueeze(1), k.transpose(-1, -2)).squeeze(1) / scale  # (H, S)
        if last_pos < seq_len - 1:
            scores[:, last_pos + 1:] = float('-inf')
        attn_orig = torch.softmax(scores, dim=-1).mean(dim=0)

        # Attention (K-normalized counterfactual)
        scores_cf = torch.matmul(q_mean.unsqueeze(1), k_unit.transpose(-1, -2)).squeeze(1) / scale
        if last_pos < seq_len - 1:
            scores_cf[:, last_pos + 1:] = float('-inf')
        attn_cf = torch.softmax(scores_cf, dim=-1).mean(dim=0)

        for kname, kidx in [('system', system_idx), ('vision', vision_idx), ('question', question_idx)]:
            # Cosine similarity: only for causally accessible keys
            causal_idx = [i for i in kidx if i <= last_pos]
            stats[f'cos_{qname}_{kname}'] = _mean(cos_sim, causal_idx) if causal_idx else float('nan')
            stats[f'attn_{qname}_{kname}_orig'] = attn_orig[causal_idx].sum().item() if causal_idx else 0.0
            stats[f'attn_{qname}_{kname}_knorm'] = attn_cf[causal_idx].sum().item() if causal_idx else 0.0

    # Backward-compatible aliases for answer query (used by existing plot code)
    stats['cos_sim_system'] = stats['cos_answer_system']
    stats['cos_sim_vision'] = stats['cos_answer_vision']
    stats['cos_sim_question'] = stats['cos_answer_question']
    stats['vision_attn_original'] = stats['attn_answer_vision_orig']
    stats['vision_attn_knorm'] = stats['attn_answer_vision_knorm']
    stats['system_attn_original'] = stats['attn_answer_system_orig']
    stats['system_attn_knorm'] = stats['attn_answer_system_knorm']
    stats['question_attn_original'] = stats['attn_answer_question_orig']
    stats['question_attn_knorm'] = stats['attn_answer_question_knorm']

    # --- Per-frame Q-K similarity: intra vs cross-frame ---
    if frame_ranges and len(frame_ranges) > 1:
        nf = len(frame_ranges)

        # Precompute per-frame mean Q (unit-normed) and per-frame K (raw)
        q_frames_unit = []  # (nf, H, D) — unit normed for cosine sim
        k_frames_raw = []   # (nf, H, T, D) — raw for dot product
        for fs, fe in frame_ranges:
            q_f = q[:, fs:fe, :].mean(dim=1)  # (H, D)
            q_frames_unit.append(q_f / q_f.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            k_frames_raw.append(k[:, fs:fe, :])  # (H, T, D)

        # F×F cosine similarity matrix (causal: frame i can attend to frames 0..i)
        # Uses angular similarity only (K also normalized per-vector)
        frame_cos = torch.full((nf, nf), float('nan'))
        for fi in range(nf):
            for fj in range(fi + 1):  # causal
                k_fj_unit = k_frames_raw[fj] / k_frames_raw[fj].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos_val = (q_frames_unit[fi].unsqueeze(1) * k_fj_unit).sum(dim=-1).mean().item()
                frame_cos[fi, fj] = cos_val

        # Intra-frame: diagonal
        diag_vals = frame_cos.diag()
        stats['cos_intra_frame'] = diag_vals[~diag_vals.isnan()].mean().item()

        # Cross-frame: off-diagonal lower triangle (causal)
        mask_cross = torch.tril(torch.ones(nf, nf, dtype=torch.bool), diagonal=-1)
        cross_vals = frame_cos[mask_cross]
        cross_vals = cross_vals[~cross_vals.isnan()]
        stats['cos_cross_frame'] = cross_vals.mean().item() if len(cross_vals) > 0 else float('nan')

        # Adjacent frames: frame i → frame i-1
        adj_vals = []
        for fi in range(1, nf):
            v = frame_cos[fi, fi - 1].item()
            if not np.isnan(v):
                adj_vals.append(v)
        stats['cos_adj_frame'] = np.mean(adj_vals) if adj_vals else float('nan')

        # Distant frames: frame i → frame 0 (for i >= 2)
        distant_vals = []
        for fi in range(2, nf):
            v = frame_cos[fi, 0].item()
            if not np.isnan(v):
                distant_vals.append(v)
        stats['cos_distant_frame'] = np.mean(distant_vals) if distant_vals else float('nan')
    else:
        stats['cos_intra_frame'] = float('nan')
        stats['cos_cross_frame'] = float('nan')
        stats['cos_adj_frame'] = float('nan')
        stats['cos_distant_frame'] = float('nan')

    return stats


# ============================================================
# LLaVA analysis (Qwen2 RoPE)
# ============================================================

@torch.no_grad()
def analyze_sample_llava(model, inputs_embeds, attention_mask, position_ids,
                         img_start, img_end, layer_stride=1, frame_ranges=None):
    """Run LLaVA forward with hooks to get post-RoPE Q, K per layer."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    decoder_layers = model.model.layers
    n_total = len(decoder_layers)
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    n_groups = n_heads // n_kv_heads
    seq_len = inputs_embeds.shape[1]

    selected = list(range(0, n_total, layer_stride))
    if selected[-1] != n_total - 1:
        selected.append(n_total - 1)
    selected_set = set(selected)

    # Detect RoPE API once (old: seq_len=int, new: position_ids=tensor)
    import inspect as _inspect
    _first_attn = decoder_layers[0].self_attn
    _rope_uses_position_ids = 'position_ids' in _inspect.signature(_first_attn.rotary_emb.forward).parameters
    _pos_default = position_ids if position_ids is not None else torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

    layer_stats = []
    hooks = []

    for i in range(n_total):
        attn_mod = decoder_layers[i].self_attn
        captured = {}

        def _qh(cap):
            def h(mod, inp, out): cap['q'] = out.detach(); return out
            return h
        def _kh(cap):
            def h(mod, inp, out): cap['k'] = out.detach(); return out
            return h

        def _lh(layer_idx, cap, attn_mod, compute):
            def h(module, input, output):
                if not compute or 'q' not in cap or 'k' not in cap:
                    cap.clear()
                    return output

                q_raw = cap.pop('q')[0].float()
                k_raw = cap.pop('k')[0].float()

                q = q_raw.view(seq_len, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
                k = k_raw.view(seq_len, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

                # Apply RoPE
                if _rope_uses_position_ids:
                    cos, sin = attn_mod.rotary_emb(k, _pos_default)
                else:
                    cos, sin = attn_mod.rotary_emb(k, seq_len=seq_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, _pos_default)

                # GQA expand
                if n_groups > 1:
                    k = k.repeat_interleave(n_groups, dim=1)

                stats = _compute_stats(q[0], k[0], img_start, img_end, seq_len, head_dim,
                                      frame_ranges=frame_ranges)
                stats['layer_idx'] = layer_idx
                layer_stats.append(stats)

                del q, k, q_raw, k_raw
                return output
            return h

        hooks.append(attn_mod.q_proj.register_forward_hook(_qh(captured)))
        hooks.append(attn_mod.k_proj.register_forward_hook(_kh(captured)))
        hooks.append(decoder_layers[i].register_forward_hook(_lh(i, captured, attn_mod, i in selected_set)))

    try:
        model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
              position_ids=position_ids, output_attentions=False, return_dict=True)
    finally:
        for h in hooks:
            h.remove()

    return layer_stats


def run_llava(args):
    from core.model_loader import parse_model_args as parse_model_args_llava, load_model_from_args
    from core.data_pipeline import create_data_loader

    model_args_dict = parse_model_args_llava(args.model_args)
    tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
    model.eval()

    if "max_frames_num" in model_args_dict and args.frames_upbound == 32:
        args.frames_upbound = int(model_args_dict["max_frames_num"])

    questions, _ = load_dataset_as_questions(
        task_name=args.task, video_folder=args.video_folder,
        image_folder=args.image_folder, limit=args.limit,
    )
    data_loader = create_data_loader(
        questions, args.image_folder, 1, 2,
        tokenizer, image_processor, model.config, args.task, conv_template,
        video_folder=args.video_folder, video_fps=1,
        frames_upbound=args.frames_upbound, force_sample=True,
    )

    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"[INFO] Model: {model_name}, heads={n_heads}, head_dim={head_dim}")
    print(f"[INFO] Samples: {len(questions)}")

    all_stats = []
    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(data_loader, questions), total=len(questions), desc="Q-K Analysis (LLaVA)"
    ):
        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]
        eff_mod = "image" if "v1.6" in model_name.lower() or "v1.5" in model_name.lower() else modality

        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None,
                image_tensor, [eff_mod], image_sizes=image_sizes
            )
        seq_len = inputs_embeds.shape[1]
        image_dim = seq_len - (input_ids.shape[-1] - 1)
        img_pos = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
        img_start = img_pos[0] if img_pos else 0
        img_end = img_start + image_dim

        # Compute frame ranges for per-frame analysis
        from Attention_map.attention_utils import get_tokens_per_frame
        tpf, inter = get_tokens_per_frame(model)
        stride_f = tpf + inter
        num_vision = img_end - img_start
        num_frames = max(1, (num_vision + stride_f - 1) // stride_f)
        fr = []
        for f in range(num_frames):
            fs = img_start + f * stride_f
            fe = min(fs + tpf, img_end)
            if fs < img_end:
                fr.append((fs, fe))

        stats = analyze_sample_llava(
            model, inputs_embeds, attention_mask, position_ids,
            img_start, img_end, args.layer_stride,
            frame_ranges=fr if len(fr) > 1 else None,
        )
        all_stats.append(stats)
        torch.cuda.empty_cache()

    return all_stats, model_name


# ============================================================
# Qwen3-VL analysis (M-RoPE via rotary_emb hook)
# ============================================================

@torch.no_grad()
def analyze_sample_qwen3vl(model, inputs, img_start, img_end, layer_stride=1, frame_ranges=None):
    """Run Qwen3-VL forward with hooks to get post-RoPE Q, K per layer."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    language_model = model.model.language_model
    decoder_layers = language_model.layers
    n_total = len(decoder_layers)
    tc = model.config.text_config
    n_heads = tc.num_attention_heads
    n_kv_heads = tc.num_key_value_heads
    head_dim = tc.head_dim
    n_groups = n_heads // n_kv_heads
    seq_len = inputs["input_ids"].shape[1]

    selected = list(range(0, n_total, layer_stride))
    if selected[-1] != n_total - 1:
        selected.append(n_total - 1)
    selected_set = set(selected)

    # Capture cos, sin from rotary_emb (called once, shared across layers)
    rope_cache = {}
    def _rope_hook(module, input, output):
        rope_cache['cos'] = output[0].detach()
        rope_cache['sin'] = output[1].detach()

    layer_stats = []
    hooks = []
    hooks.append(language_model.rotary_emb.register_forward_hook(_rope_hook))

    for i in range(n_total):
        attn_mod = decoder_layers[i].self_attn
        captured = {}

        def _qh(cap):
            def h(mod, inp, out): cap['q'] = out.detach(); return out
            return h
        def _kh(cap):
            def h(mod, inp, out): cap['k'] = out.detach(); return out
            return h

        def _lh(layer_idx, cap, attn_mod, compute):
            def h(module, input, output):
                if not compute or 'q' not in cap or 'k' not in cap:
                    cap.clear()
                    return output
                if 'cos' not in rope_cache:
                    cap.clear()
                    return output

                q_raw = cap.pop('q')[0].float()
                k_raw = cap.pop('k')[0].float()

                q = q_raw.view(seq_len, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
                k = k_raw.view(seq_len, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

                # Apply q_norm, k_norm (Qwen3-VL specific)
                q = attn_mod.q_norm(q)
                k = attn_mod.k_norm(k)

                # Apply M-RoPE using captured cos, sin (move to same device as Q/K)
                cos = rope_cache['cos'].to(device=q.device, dtype=torch.float32)
                sin = rope_cache['sin'].to(device=q.device, dtype=torch.float32)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # GQA expand
                if n_groups > 1:
                    k = k.repeat_interleave(n_groups, dim=1)

                stats = _compute_stats(q[0], k[0], img_start, img_end, seq_len, head_dim,
                                      frame_ranges=frame_ranges)
                stats['layer_idx'] = layer_idx
                layer_stats.append(stats)

                del q, k, q_raw, k_raw
                return output
            return h

        hooks.append(attn_mod.q_proj.register_forward_hook(_qh(captured)))
        hooks.append(attn_mod.k_proj.register_forward_hook(_kh(captured)))
        hooks.append(decoder_layers[i].register_forward_hook(_lh(i, captured, attn_mod, i in selected_set)))

    try:
        model(**inputs, output_attentions=False, return_dict=True)
    finally:
        for h in hooks:
            h.remove()

    return layer_stats


def run_qwen3vl(args):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_args_dict = _parse_model_args(args.model_args)
    pretrained = _resolve_pretrained_path(model_args_dict.get("pretrained", ""))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pretrained, dtype="auto",
        device_map=model_args_dict.get("device_map", "auto"),
        attn_implementation="eager",
    )
    proc_kw = {}
    if "min_pixels" in model_args_dict: proc_kw["min_pixels"] = int(model_args_dict["min_pixels"])
    if "max_pixels" in model_args_dict: proc_kw["max_pixels"] = int(model_args_dict["max_pixels"])
    processor = AutoProcessor.from_pretrained(pretrained, **proc_kw)
    model.eval()

    model_name = os.path.basename(pretrained.rstrip("/"))
    video_pad_id = processor.tokenizer.encode("<|video_pad|>", add_special_tokens=False)[0]

    if args.frames_upbound == 32:
        for key in ("max_frames_num", "max_num_frames"):
            if key in model_args_dict:
                args.frames_upbound = int(model_args_dict[key])
                break

    questions, _ = load_dataset_as_questions(
        task_name=args.task, video_folder=args.video_folder,
        image_folder=args.image_folder, limit=args.limit,
    )

    tc = model.config.text_config
    print(f"[INFO] Model: {model_name}, heads={tc.num_attention_heads}, head_dim={tc.head_dim}")
    print(f"[INFO] Samples: {len(questions)}")

    all_stats = []
    for line in tqdm(questions, desc="Q-K Analysis (Qwen3-VL)"):
        video_rel = line.get("video", "")
        if not video_rel:
            continue
        video_path = os.path.join(args.video_folder, video_rel) if args.video_folder and not os.path.isabs(video_rel) else video_rel
        if not os.path.exists(video_path):
            continue

        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        n_fr = min(total, args.frames_upbound)
        indices = np.linspace(0, total - 1, n_fr, dtype=int).tolist() if total > n_fr else list(range(total))
        frames = [Image.fromarray(f) for f in vr.get_batch(indices).asnumpy()]

        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": line.get("question", "")},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"][0]
        vis_idx = (input_ids == video_pad_id).nonzero(as_tuple=True)[0]
        if len(vis_idx) == 0:
            continue
        img_start, img_end = vis_idx[0].item(), vis_idx[-1].item() + 1

        # Compute frame ranges for per-frame analysis
        temporal_patch_size = getattr(model.config.vision_config, 'temporal_patch_size', 2)
        num_temporal_slots = (len(indices) + temporal_patch_size - 1) // temporal_patch_size
        num_vision = img_end - img_start
        tpf_q = num_vision // max(num_temporal_slots, 1)
        fr = []
        for f in range(num_temporal_slots):
            fs = img_start + f * tpf_q
            fe = min(fs + tpf_q, img_end)
            if fs < img_end:
                fr.append((fs, fe))

        stats = analyze_sample_qwen3vl(model, inputs, img_start, img_end, args.layer_stride,
                                       frame_ranges=fr if len(fr) > 1 else None)
        all_stats.append(stats)

        del inputs
        torch.cuda.empty_cache()

    return all_stats, model_name


# ============================================================
# Utility
# ============================================================

def _parse_model_args(s):
    if not s: return {}
    r = {}
    for item in s.split(","):
        item = item.strip()
        if "=" not in item: continue
        k, v = item.split("=", 1)
        for cast in [lambda x: {"true":True,"false":False,"none":None}[x.lower()], int, float]:
            try: v = cast(v); break
            except: continue
        r[k.strip()] = v
    return r

def _resolve_pretrained_path(p):
    if os.path.exists(os.path.join(p, "config.json")): return p
    for d in [os.path.join(p, "snapshots")]:
        if os.path.isdir(d):
            for h in os.listdir(d): return os.path.join(d, h)
    if "/" in p and not os.path.isdir(p):
        hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cs = os.path.join(hf, "models--" + p.replace("/","--"), "snapshots")
        if os.path.isdir(cs):
            for h in os.listdir(cs): return os.path.join(cs, h)
    return p

def _detect_model_type(pretrained):
    if "qwen3-vl" in pretrained.lower() or "qwen3_vl" in pretrained.lower():
        return "qwen3_vl"
    resolved = _resolve_pretrained_path(pretrained)
    cp = os.path.join(resolved, "config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            if "qwen3_vl" in json.load(f).get("model_type", ""):
                return "qwen3_vl"
    return "llava"


# ============================================================
# Aggregation & Visualization
# ============================================================

def aggregate_stats(all_stats):
    if not all_stats or not all_stats[0]: return {}
    layer_indices = sorted(set(s['layer_idx'] for sample in all_stats for s in sample))
    keys = [k for k in all_stats[0][0].keys() if k != 'layer_idx']
    agg = {k: [] for k in keys}
    agg['layer_idx'] = layer_indices
    for li in layer_indices:
        vals = {k: [] for k in keys}
        for sample in all_stats:
            for s in sample:
                if s['layer_idx'] == li:
                    for k in keys: vals[k].append(s[k])
                    break
        for k in keys:
            agg[k].append(np.nanmean(vals[k]) if vals[k] else float('nan'))
    return {k: np.array(v) for k, v in agg.items()}


def plot_results(agg, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    layers = agg['layer_idx']
    kcolors = {'system': '#ef5350', 'vision': '#42a5f5', 'question': '#66bb6a'}

    # ---- Figure 1: K-norms + Answer query analysis (backward-compatible) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for r in ['system', 'vision', 'question']:
        ax.plot(layers, agg[f'k_norm_{r}'], color=kcolors[r], marker='o', markersize=3, label=r.capitalize())
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean ||K||')
    ax.set_title('H1: K-vector Norms'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for r in ['system', 'vision', 'question']:
        ax.plot(layers, agg[f'cos_answer_{r}'], color=kcolors[r], marker='o', markersize=3, label=f'K({r})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Similarity')
    ax.set_title('H2: Q(answer) vs K alignment'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(layers, agg['attn_answer_vision_orig'], 'b-o', label='Original', markersize=3)
    ax.plot(layers, agg['attn_answer_vision_knorm'], 'b--s', label='K-normalized', markersize=3)
    ax.fill_between(layers, agg['attn_answer_vision_orig'], agg['attn_answer_vision_knorm'],
                    alpha=0.2, color='blue', label='Norm effect')
    ax.set_xlabel('Layer'); ax.set_ylabel('Vision Attn Fraction')
    ax.set_title('Counterfactual: Answer→Vision'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    w = 0.35; x = np.arange(3)
    orig = [agg[f'attn_answer_{r}_orig'].mean() for r in ['system','vision','question']]
    norm = [agg[f'attn_answer_{r}_knorm'].mean() for r in ['system','vision','question']]
    ax.bar(x - w/2, orig, w, label='Original', color=list(kcolors.values()))
    ax.bar(x + w/2, norm, w, label='K-normalized', color=list(kcolors.values()), alpha=0.5, edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['System','Vision','Question'])
    ax.set_ylabel('Mean Attn Fraction'); ax.set_title('Answer Attn Redistribution')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f'Q-K Analysis — {model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'qk_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Figure 2: Multi-query comparison (answer vs question vs vision) ----
    query_groups = ['answer', 'question', 'vision']
    query_labels = {'answer': 'Q(Answer)', 'question': 'Q(Question)', 'vision': 'Q(Vision)'}
    query_styles = {'answer': '-', 'question': '--', 'vision': ':'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Row 0: Cosine similarity per query group
    for col, qname in enumerate(query_groups):
        ax = axes[0, col]
        for kname in ['system', 'vision', 'question']:
            key = f'cos_{qname}_{kname}'
            if key in agg:
                ax.plot(layers, agg[key], color=kcolors[kname], marker='o', markersize=2,
                        label=f'K({kname})')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer'); ax.set_title(f'Cosine Sim: {query_labels[qname]}', fontsize=10)
        if col == 0: ax.set_ylabel('Cosine Similarity')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 1: Vision attention fraction (original vs K-normalized) per query group
    for col, qname in enumerate(query_groups):
        ax = axes[1, col]
        orig_key = f'attn_{qname}_vision_orig'
        norm_key = f'attn_{qname}_vision_knorm'
        if orig_key in agg and norm_key in agg:
            ax.plot(layers, agg[orig_key], 'b-o', label='Original', markersize=2)
            ax.plot(layers, agg[norm_key], 'b--s', label='K-normalized', markersize=2)
            ax.fill_between(layers, agg[orig_key], agg[norm_key], alpha=0.15, color='blue')
        ax.set_xlabel('Layer'); ax.set_title(f'Vision Attn: {query_labels[qname]}', fontsize=10)
        if col == 0: ax.set_ylabel('Vision Attn Fraction')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f'Multi-Query Q-K Analysis — {model_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'qk_multi_query.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Figure 3: Cross-query vision attention comparison (overlay) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    qcolors = {'answer': '#e53935', 'question': '#1E88E5', 'vision': '#43A047'}
    for qname in query_groups:
        orig_key = f'attn_{qname}_vision_orig'
        norm_key = f'attn_{qname}_vision_knorm'
        if orig_key in agg:
            ax.plot(layers, agg[orig_key], color=qcolors[qname], linestyle='-', marker='o',
                    markersize=3, label=f'{query_labels[qname]} (orig)')
            ax.plot(layers, agg[norm_key], color=qcolors[qname], linestyle='--', marker='s',
                    markersize=3, alpha=0.6, label=f'{query_labels[qname]} (K-norm)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Vision Attention Fraction')
    ax.set_title(f'Vision Attention by Query Type — {model_name}')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'qk_vision_by_query.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Figure 4: Intra-frame vs Cross-frame cosine similarity ----
    if 'cos_intra_frame' in agg and not np.all(np.isnan(agg['cos_intra_frame'])):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(layers, agg['cos_intra_frame'], 'r-o', markersize=3, label='Intra-frame (Q_fi · K_fi)')
        ax.plot(layers, agg['cos_cross_frame'], 'b-s', markersize=3, label='Cross-frame (Q_fi · K_fj, j<i)')
        if 'cos_adj_frame' in agg:
            ax.plot(layers, agg['cos_adj_frame'], 'g-^', markersize=3, label='Adjacent (Q_fi · K_f(i-1))')
        if 'cos_distant_frame' in agg:
            ax.plot(layers, agg['cos_distant_frame'], 'm-d', markersize=3, label='Distant (Q_fi · K_f0)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'Intra vs Cross-Frame Q-K Similarity — {model_name}')
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'qk_frame_similarity.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[SAVED] {output_dir}/qk_frame_similarity.png')

    print(f'[SAVED] {output_dir}/qk_analysis.png, qk_multi_query.png, qk_vision_by_query.png')

    np.savez(os.path.join(output_dir, 'qk_stats.npz'), **{k: np.array(v) for k, v in agg.items()})
    print(f'[SAVED] {os.path.join(output_dir, "qk_stats.npz")}')


def plot_comparison(model_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models = {}
    for d in sorted(model_dirs):
        p = os.path.join(d, 'qk_stats.npz')
        if os.path.exists(p):
            models[os.path.basename(d)] = dict(np.load(p))
    if not models:
        print("[WARN] No qk_stats.npz found"); return

    n = len(models)
    fig, axes = plt.subplots(3, n, figsize=(5*n, 12))
    if n == 1: axes = axes[:, np.newaxis]
    colors = {'system':'#ef5350','vision':'#42a5f5','question':'#66bb6a'}

    for col, (name, agg) in enumerate(models.items()):
        layers = agg['layer_idx']
        ax = axes[0, col]
        for r in ['system','vision','question']:
            ax.plot(layers, agg[f'k_norm_{r}'], color=colors[r], label=r.capitalize(), linewidth=1.5, marker='o', markersize=2)
        ax.set_title(name, fontsize=9, fontweight='bold')
        if col==0: ax.set_ylabel('||K|| (H1)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1, col]
        for r in ['system','vision','question']:
            ax.plot(layers, agg[f'cos_sim_{r}'], color=colors[r], label=r.capitalize(), linewidth=1.5, marker='o', markersize=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if col==0: ax.set_ylabel('cos(Q,K) (H2)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.plot(layers, agg['vision_attn_original'], 'b-o', label='Original', markersize=2)
        ax.plot(layers, agg['vision_attn_knorm'], 'b--s', label='K-normalized', markersize=2)
        ax.fill_between(layers, agg['vision_attn_original'], agg['vision_attn_knorm'], alpha=0.15, color='blue')
        ax.set_xlabel('Layer')
        if col==0: ax.set_ylabel('Vision Attn Frac')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle('Q-K Space Analysis — Model Comparison (post-RoPE)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, 'qk_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'[SAVED] {path}')


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Q-K Space Analysis (post-RoPE)")
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto","llava","qwen3_vl"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/qk_analysis")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--layer_stride", type=int, default=1)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--compare_dirs", type=str, nargs="+", default=None)
    args = parser.parse_args()

    if args.compare_dirs:
        plot_comparison(args.compare_dirs, args.output_dir)
        return

    mt = args.model_type
    if mt == "auto":
        mt = _detect_model_type(_parse_model_args(args.model_args).get("pretrained",""))
        print(f"[INFO] Detected: {mt}")

    if mt == "qwen3_vl":
        all_stats, model_name = run_qwen3vl(args)
    else:
        all_stats, model_name = run_llava(args)

    agg = aggregate_stats(all_stats)
    if agg:
        plot_results(agg, args.output_dir, model_name)
        print(f"\n{'='*60}")
        print(f"  Q-K Analysis (post-RoPE): {model_name}")
        print(f"  Samples: {len(all_stats)}")
        print(f"{'='*60}")
        print(f"\n  [H1] K-norm (layer avg):")
        print(f"    System:   {agg['k_norm_system'].mean():.3f}")
        print(f"    Vision:   {agg['k_norm_vision'].mean():.3f}")
        print(f"    Question: {agg['k_norm_question'].mean():.3f}")
        print(f"    System/Vision: {agg['k_norm_system'].mean()/max(agg['k_norm_vision'].mean(),1e-8):.2f}x")

        for qname, qlabel in [('answer','Answer'), ('question','Question'), ('vision','Vision')]:
            print(f"\n  [H2] cos(Q({qlabel}), K) — layer avg:")
            for kname in ['system', 'vision', 'question']:
                key = f'cos_{qname}_{kname}'
                if key in agg:
                    print(f"    K({kname:8s}): {agg[key].mean():.4f}")

        print(f"\n  [Counterfactual] Vision attn fraction (layer avg):")
        for qname, qlabel in [('answer','Answer'), ('question','Question'), ('vision','Vision')]:
            orig_key = f'attn_{qname}_vision_orig'
            norm_key = f'attn_{qname}_vision_knorm'
            if orig_key in agg:
                o, c = agg[orig_key].mean(), agg[norm_key].mean()
                print(f"    {qlabel:8s} → Vision: {o:.4f} → {c:.4f} (delta {c-o:+.4f})")

        if 'cos_intra_frame' in agg and not np.all(np.isnan(agg['cos_intra_frame'])):
            intra = np.nanmean(agg['cos_intra_frame'])
            cross = np.nanmean(agg['cos_cross_frame'])
            print(f"\n  [Frame-level] Q-K cosine similarity (layer avg):")
            print(f"    Intra-frame (Q_fi·K_fi):     {intra:.4f}")
            print(f"    Cross-frame (Q_fi·K_fj):     {cross:.4f}")
            print(f"    Intra/Cross ratio:            {intra/cross:.2f}x" if abs(cross) > 1e-6 else "")
            if 'cos_adj_frame' in agg:
                print(f"    Adjacent (Q_fi·K_f(i-1)):    {np.nanmean(agg['cos_adj_frame']):.4f}")
            if 'cos_distant_frame' in agg:
                print(f"    Distant (Q_fi·K_f0):         {np.nanmean(agg['cos_distant_frame']):.4f}")
        print()


if __name__ == "__main__":
    main()
