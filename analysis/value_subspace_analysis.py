"""
Value Subspace Tracing: Vision Token → V Projection → Last Token.

Direction information flow를 attention chain의 각 stage에서 추적:
  Stage 1: Vision token h_vision (layer l 입력, vision position)
  Stage 2: Value vector v = V_proj(LayerNorm(h_vision))
  Stage 3: Attention output = Σ α_i · v_i (self_attn이 last token에 쓰는 값)
  Stage 4: Last token h_last (layer l 출력)

각 stage에서:
  (a) Direction 4-class linear probe
  (b) Fisher Discriminant Ratio (direction)
  (c) Top-k Fisher subspace 추출

Stage 간 subspace alignment:
  vision direction subspace ↔ value direction subspace ↔ last token direction subspace

Usage:
    # 단일
    CUDA_VISIBLE_DEVICES=0 python analysis/value_subspace_analysis.py \
        --model llava-video-7b_lora_4combo_v2_baseline --task obj_place

    # 전체
    CUDA_VISIBLE_DEVICES=0 python analysis/value_subspace_analysis.py \
        --model all --task all
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"
LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline":
        "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "llava-video-7b_lora_4combo_v2_delta":
        "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
}

ALL_MODELS = ["llava-video-7b", "llava-video-7b_lora_4combo_v2_baseline", "llava-video-7b_lora_4combo_v2_delta"]
ALL_TASKS = ["shape_color", "obj_place"]
IDENTITY_ATTRS = {"shape_color": "shape", "obj_color": "obj_class", "shape_place": "place_class", "obj_place": "obj_class"}
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
DIRECTIONS = ["up", "down", "left", "right"]


def model_short(name):
    return name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")


# ============================================================
#  GPU Probe
# ============================================================

def gpu_probe(X, y, nc, seed=42, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32) if isinstance(X, np.ndarray) else X.clone().to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device) if isinstance(y, np.ndarray) else y.clone().long().to(device)
    m = X_t.mean(0); X_t -= m; s = X_t.std(0); s[s < 1e-8] = 1; X_t /= s
    nt = max(1, int(N * 0.3))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    p = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    Xtr, ytr, Xte, yte = X_t[p[:-nt]], y_t[p[:-nt]], X_t[p[-nt:]], y_t[p[-nt:]]
    del X_t, y_t
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(D, nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 64):
            b = idx[i:i + 64]; loss = crit(model(Xtr[b]), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
    del model, opt; torch.cuda.empty_cache()
    return acc


# ============================================================
#  Fisher Subspace
# ============================================================

def compute_fisher_subspace(X_np, y_np, top_k=50, num_classes=4):
    """Fisher discriminant: top-k dimensions + FDR."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(device, dtype=torch.float32)
    y = torch.from_numpy(y_np).long().to(device)
    gm = X.mean(0)

    var_b = torch.zeros(X.shape[1], device=device)
    var_w = torch.zeros(X.shape[1], device=device)
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() == 0: continue
        Xc = X[mask]
        cm = Xc.mean(0)
        var_b += mask.sum().float() * (cm - gm) ** 2
        var_w += (Xc - cm).pow(2).sum(dim=0)

    fisher = var_b / (var_w + 1e-8)
    fdr = fisher.mean().item()
    topk_idx = fisher.argsort(descending=True)[:top_k].cpu().numpy()
    return topk_idx, fdr, fisher.cpu().numpy()


def subspace_overlap(idx_a, idx_b):
    """두 Fisher top-k index set의 overlap ratio."""
    set_a = set(idx_a.tolist())
    set_b = set(idx_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(len(set_a), 1), inter / max(union, 1)  # recall, jaccard


# ============================================================
#  Model Loading
# ============================================================

def load_model(model_name):
    from core.model_loader import parse_model_args, load_model_from_args
    if model_name == "llava-video-7b":
        a = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    else:
        lp = LORA_PATHS[model_name]
        a = f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    ma = parse_model_args(a)
    tok, model, ip, cl, mn, ct = load_model_from_args(ma)
    model.eval()
    return tok, model, ip, mn, ct


def load_metadata(task):
    with open(os.path.join(META_ROOT, f"{task}_metadata.json")) as f:
        return json.load(f)


def get_labels(metadata, qids, attr):
    mb = {m['id']: m for m in metadata}
    le = LabelEncoder()
    raw = [str(mb[int(str(q).split('_')[0])][attr]) for q in qids]
    return le.fit_transform(raw), len(le.classes_)


# ============================================================
#  Extraction with Hooks
# ============================================================

@torch.no_grad()
def extract_all_stages(model, tokenizer, task, conv_template, image_processor, metadata, limit=200):
    """
    각 layer에서 4개 stage의 last-token/vision-token features 추출:
      1. vision_input: layer 입력의 vision token 위치 hidden state (mean-pooled)
      2. value_vector: V_proj 출력의 vision token 위치 (mean-pooled)
      3. attn_output: self_attn 출력의 last token 위치
      4. last_token: layer 출력의 last token 위치
    """
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions
    from llava.constants import IMAGE_TOKEN_INDEX

    questions, _ = load_dataset_as_questions(task_name=TASK_FULL(task), limit=limit)
    dl = create_data_loader(
        questions, "", 1, 2, tokenizer, image_processor, model.config,
        TASK_FULL(task), conv_template, video_folder="", video_fps=1,
        frames_upbound=8, force_sample=True
    )

    n_layers = model.config.num_hidden_layers

    # Storage
    all_vision_input = {l: [] for l in range(n_layers)}
    all_value_vector = {l: [] for l in range(n_layers)}
    all_attn_output = {l: [] for l in range(n_layers)}
    all_last_token = {l: [] for l in range(n_layers)}
    all_qids = []

    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(dl, questions), total=len(questions), desc=f"  {task}"
    ):
        all_qids.append(line['q_id'])
        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]

        # Find vision token positions
        img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        n_text_before = img_pos[0].item() if len(img_pos) > 0 else 0
        n_text_after = input_ids.shape[1] - n_text_before - 1  # 1 = IMAGE_TOKEN_INDEX

        # Prepare inputs (expands image tokens)
        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor, [modality], image_sizes=image_sizes
            )

        seq_len = inputs_embeds.shape[1]
        n_vision = seq_len - n_text_before - n_text_after
        vision_start = n_text_before
        vision_end = vision_start + n_vision

        # Hooks
        intermediate = {}
        hooks = []

        for li in range(n_layers):
            layer_module = model.model.layers[li]

            # Hook 1: V_proj output (value vectors)
            def make_vproj_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output shape: (1, seq_len, num_heads * head_dim)
                    v_vision = output[0, vision_start:vision_end, :].detach()
                    # Mean pool across vision tokens → (D,)
                    intermediate[f"value_{layer_idx}"] = v_vision.mean(dim=0).cpu().to(torch.float16)
                return hook_fn
            hooks.append(layer_module.self_attn.v_proj.register_forward_hook(make_vproj_hook(li)))

            # Hook 2: self_attn output (attention-weighted value at last position)
            def make_attn_hook(layer_idx):
                def hook_fn(module, input, output):
                    attn_out = output[0] if isinstance(output, tuple) else output
                    intermediate[f"attn_{layer_idx}"] = attn_out[0, -1, :].detach().cpu().to(torch.float16)
                return hook_fn
            hooks.append(layer_module.self_attn.register_forward_hook(make_attn_hook(li)))

            # Hook 3: Layer input (vision tokens) and output (last token)
            def make_layer_hook(layer_idx):
                def hook_fn(module, input, output):
                    h_in = input[0] if isinstance(input, tuple) else input
                    h_out = output[0] if isinstance(output, tuple) else output

                    # Vision token input (mean-pooled)
                    h_vision = h_in[0, vision_start:vision_end, :].detach()
                    all_vision_input[layer_idx].append(h_vision.mean(dim=0).cpu().to(torch.float16))

                    # Value vector (from v_proj hook)
                    v = intermediate.get(f"value_{layer_idx}")
                    if v is not None:
                        all_value_vector[layer_idx].append(v)

                    # Attn output at last position (from attn hook)
                    a = intermediate.get(f"attn_{layer_idx}")
                    if a is not None:
                        all_attn_output[layer_idx].append(a)

                    # Last token output
                    all_last_token[layer_idx].append(h_out[0, -1, :].detach().cpu().to(torch.float16))
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_layer_hook(li)))

        # Forward pass
        outputs = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, output_attentions=False, return_dict=True
        )
        del outputs

        for h in hooks:
            h.remove()
        intermediate.clear()

        if len(all_qids) % 50 == 0:
            torch.cuda.empty_cache()

    return all_vision_input, all_value_vector, all_attn_output, all_last_token, all_qids, n_layers


# ============================================================
#  Analysis
# ============================================================

def run_analysis(all_vision_input, all_value_vector, all_attn_output, all_last_token,
                 all_qids, n_layers, task, metadata, output_dir, model_name):
    """4 stages × per layer: direction probe + FDR + subspace overlap."""

    id_attr = IDENTITY_ATTRS[task]
    dir_labels, dir_nc = get_labels(metadata, all_qids, "direction")
    id_labels, id_nc = get_labels(metadata, all_qids, id_attr)

    results = {
        "model": model_name, "task": task,
        "layers": [],
        # Direction probe per stage
        "vision_dir": [], "value_dir": [], "attn_out_dir": [], "last_token_dir": [],
        # Identity probe per stage
        "vision_id": [], "value_id": [], "attn_out_id": [], "last_token_id": [],
        # FDR per stage
        "vision_fdr": [], "value_fdr": [], "attn_out_fdr": [], "last_token_fdr": [],
        # Subspace overlap (Fisher top-50)
        "overlap_vision_value": [], "overlap_value_last": [], "overlap_vision_last": [],
    }

    FISHER_K = 50

    print(f"\n  {'Ly':>3} | {'Vis→dir':>8} {'Val→dir':>8} {'Attn→dir':>9} {'Last→dir':>9} | "
          f"{'Vis→id':>7} {'Val→id':>7} {'Last→id':>8} | "
          f"{'FDR_vis':>8} {'FDR_val':>8} {'FDR_last':>9} | "
          f"{'V↔Val':>6} {'Val↔L':>6}")
    print("  " + "-" * 120)

    for l in range(n_layers):
        if not all_vision_input[l] or not all_value_vector[l]:
            continue

        feat_vis = torch.stack(all_vision_input[l]).numpy().astype(np.float32)
        feat_val = torch.stack(all_value_vector[l]).numpy().astype(np.float32)
        feat_attn = torch.stack(all_attn_output[l]).numpy().astype(np.float32)
        feat_last = torch.stack(all_last_token[l]).numpy().astype(np.float32)

        # Direction probes
        vis_dir = gpu_probe(feat_vis.copy(), dir_labels, dir_nc)
        val_dir = gpu_probe(feat_val.copy(), dir_labels, dir_nc)
        attn_dir = gpu_probe(feat_attn.copy(), dir_labels, dir_nc)
        last_dir = gpu_probe(feat_last.copy(), dir_labels, dir_nc)

        # Identity probes
        vis_id = gpu_probe(feat_vis.copy(), id_labels, id_nc)
        val_id = gpu_probe(feat_val.copy(), id_labels, id_nc)
        attn_id = gpu_probe(feat_attn.copy(), id_labels, id_nc)
        last_id = gpu_probe(feat_last.copy(), id_labels, id_nc)

        # Fisher subspaces
        vis_topk, vis_fdr, _ = compute_fisher_subspace(feat_vis, dir_labels, FISHER_K)
        val_topk, val_fdr, _ = compute_fisher_subspace(feat_val, dir_labels, FISHER_K)
        _, attn_fdr, _ = compute_fisher_subspace(feat_attn, dir_labels, FISHER_K)
        last_topk, last_fdr, _ = compute_fisher_subspace(feat_last, dir_labels, FISHER_K)

        # Subspace overlap
        ov_vis_val, _ = subspace_overlap(vis_topk, val_topk)
        ov_val_last, _ = subspace_overlap(val_topk, last_topk)
        ov_vis_last, _ = subspace_overlap(vis_topk, last_topk)

        results["layers"].append(l)
        results["vision_dir"].append(vis_dir)
        results["value_dir"].append(val_dir)
        results["attn_out_dir"].append(attn_dir)
        results["last_token_dir"].append(last_dir)
        results["vision_id"].append(vis_id)
        results["value_id"].append(val_id)
        results["attn_out_id"].append(attn_id)
        results["last_token_id"].append(last_id)
        results["vision_fdr"].append(vis_fdr)
        results["value_fdr"].append(val_fdr)
        results["attn_out_fdr"].append(attn_fdr)
        results["last_token_fdr"].append(last_fdr)
        results["overlap_vision_value"].append(ov_vis_val)
        results["overlap_value_last"].append(ov_val_last)
        results["overlap_vision_last"].append(ov_vis_last)

        print(f"  {l:>3} | {vis_dir:>7.1f}% {val_dir:>7.1f}% {attn_dir:>8.1f}% {last_dir:>8.1f}% | "
              f"{vis_id:>6.1f}% {val_id:>6.1f}% {last_id:>7.1f}% | "
              f"{vis_fdr:>8.3f} {val_fdr:>8.3f} {last_fdr:>9.3f} | "
              f"{ov_vis_val:>5.1%} {ov_val_last:>5.1%}")

        del feat_vis, feat_val, feat_attn, feat_last

    # Save
    os.makedirs(output_dir, exist_ok=True)
    short = model_short(model_name)
    sp = os.path.join(output_dir, f"value_subspace_{short}_{task}.json")
    with open(sp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] {sp}")
    return results


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task", default="obj_place")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output_dir", default="analysis/value_subspace_results")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Loading: {model_name}")
        print(f"{'='*60}")
        tok, model, ip, mn, ct = load_model(model_name)

        for task in tasks:
            print(f"\n{'#'*60}")
            print(f"  {model_name} / {task}")
            print(f"{'#'*60}")
            metadata = load_metadata(task)
            vis, val, attn, last, qids, nl = extract_all_stages(
                model, tok, task, ct, ip, metadata, args.limit
            )
            run_analysis(vis, val, attn, last, qids, nl, task, metadata, args.output_dir, model_name)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
