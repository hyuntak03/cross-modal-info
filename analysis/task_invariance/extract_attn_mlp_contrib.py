"""
Decompose last-token direction amplification into attn vs MLP contributions per layer.

For each decoder layer L:
  residual_update_L = attn_out_L + mlp_out_L  (before residual add)

Hook self_attn.output and mlp.output at each target layer (L10..L25),
capture the LAST-TOKEN slice, save per-sample.

Usage:
  CUDA_VISIBLE_DEVICES=0 python extract_attn_mlp_contrib.py \
      --task vlm_direction_testbed_R2R_4way_1500_obj_place \
      --model baseline --limit 2000
"""
import argparse, os, sys, json, gc, importlib.util
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info"
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT")
os.environ.setdefault("HF_HOME", "/data/datasets/LLaVA-Video-100K-Subset/")
os.environ.setdefault("HF_DATASETS_CACHE", "/local_datasets/vlm_direction/")

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

VIDEO_FOLDER = "/local_datasets/vlm_direction/"
VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"
DELTA_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

LAYERS = list(range(10, 26))  # L10..L25
OUT_ROOT = "/local_datasets/vlm_direction/attn_mlp_contrib"


def _import_module_direct(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_letter_ids(tokenizer):
    ids = {}
    for ltr in ["A", "B", "C", "D"]:
        for cand in [ltr, " " + ltr]:
            tids = tokenizer.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids[ltr] = tids[0]; break
    return ids


@torch.no_grad()
def run(args):
    _model_loader = _import_module_direct(
        "core.model_loader", os.path.join(_PROJECT_ROOT, "core", "model_loader.py")
    )
    _data_pipeline = _import_module_direct(
        "core.data_pipeline", os.path.join(_PROJECT_ROOT, "core", "data_pipeline.py")
    )
    _dataset_loader = _import_module_direct(
        "core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py")
    )
    parse_model_args_llava = _model_loader.parse_model_args
    load_model_from_args = _model_loader.load_model_from_args
    create_data_loader = _data_pipeline.create_data_loader
    load_dataset_as_questions = _dataset_loader.load_dataset_as_questions

    if args.model == "vanilla":
        model_args_str = VANILLA_ARGS
    elif args.model == "baseline":
        model_args_str = f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    else:
        model_args_str = f"lora_pretrained={DELTA_LORA},{VANILLA_ARGS}"
    print(f"[load model] {args.model}")
    model_args_dict = parse_model_args_llava(model_args_str)
    tokenizer, model, image_processor, _, model_name, conv_template = load_model_from_args(model_args_dict)
    model.eval()
    model.tie_weights()

    questions, _ = load_dataset_as_questions(
        task_name=args.task,
        video_folder=VIDEO_FOLDER,
        image_folder="",
        hf_cache_dir=os.environ.get("HF_HOME"),
        limit=args.limit if args.limit > 0 else -1,
    )
    if args.offset > 0:
        questions = questions[args.offset:]
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"[data] {len(questions)} samples")

    data_loader = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        args.task, conv_template, video_folder=VIDEO_FOLDER, video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    decoder_layers = model.model.layers if hasattr(model.model, "layers") else model.language_model.model.layers
    letter_ids = get_letter_ids(tokenizer)
    id_to_letter = {v: k for k, v in letter_ids.items()}
    letter_tok_ids = list(letter_ids.values())

    captured = {}

    def make_attn_hook(L):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[(L, "attn")] = t.detach()
        return hook

    def make_mlp_hook(L):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[(L, "mlp")] = t.detach()
        return hook

    hooks = []
    for L in LAYERS:
        hooks.append(decoder_layers[L].self_attn.register_forward_hook(make_attn_hook(L)))
        hooks.append(decoder_layers[L].mlp.register_forward_hook(make_mlp_hook(L)))

    n = len(questions)
    hidden_dim = model.config.hidden_size
    nL = len(LAYERS)
    attn_c = np.zeros((n, nL, hidden_dim), dtype=np.float16)
    mlp_c = np.zeros((n, nL, hidden_dim), dtype=np.float16)
    final_h = np.zeros((n, hidden_dim), dtype=np.float16)
    preds, expects, corrects, dirs = [], [], [], []

    idx = 0
    for batch, line in tqdm(zip(data_loader, questions), total=len(questions)):
        if batch is None: continue
        input_ids, image_tensor, image_sizes, _, _, modality = batch
        input_ids = input_ids.to("cuda")
        image_tensor = [t.to("cuda") for t in image_tensor]
        try:
            (_, position_ids, attention_mask, _, inputs_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, image_tensor,
                    modalities=[modality], image_sizes=image_sizes)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          position_ids=position_ids, return_dict=True)
        except Exception as e:
            print(f"[ERR] {line.get('q_id','?')}: {e}")
            captured.clear()
            continue

        for li, L in enumerate(LAYERS):
            a = captured.get((L, "attn"))
            m = captured.get((L, "mlp"))
            if a is None or m is None: continue
            if a.dim() == 3: a = a[0]
            if m.dim() == 3: m = m[0]
            attn_c[idx, li] = a[-1].to(torch.float16).cpu().numpy()
            mlp_c[idx, li] = m[-1].to(torch.float16).cpu().numpy()
        captured.clear()

        logits = out.logits[0, -1, :]
        sub = logits[letter_tok_ids]
        pred = id_to_letter[letter_tok_ids[int(sub.argmax())]]
        expected = str(line["answer"]).strip()
        # expected may be full text; resolve letter if needed
        if len(expected) != 1:
            cands = line.get("candidates", [])
            if isinstance(cands, str):
                import ast as _ast
                try: cands = _ast.literal_eval(cands)
                except Exception: cands = []
            for ci, c in enumerate(cands):
                if str(c).strip() == expected:
                    expected = chr(65 + ci); break
        preds.append(pred); expects.append(expected); corrects.append(int(pred == expected))
        dirs.append(str(line.get("direction") or line.get("answer_direction") or ""))
        idx += 1

        del out
        if idx % 50 == 0:
            torch.cuda.empty_cache()

    for h in hooks: h.remove()

    acc = float(np.mean(corrects) * 100) if corrects else 0.0
    print(f"\n[{args.model}/{args.task}] n={idx} acc={acc:.2f}%")

    os.makedirs(OUT_ROOT, exist_ok=True)
    task_short = args.task.replace("vlm_direction_testbed_R2R_4way_1500_", "")
    suffix = f"_off{args.offset}_lim{args.limit}" if args.limit > 0 else ""
    out_path = os.path.join(OUT_ROOT, f"{args.model}_{task_short}{suffix}.npz")
    np.savez_compressed(out_path,
                        attn_contribs=attn_c[:idx],
                        mlp_contribs=mlp_c[:idx],
                        directions=np.array(dirs),
                        preds=np.array(preds),
                        expects=np.array(expects),
                        corrects=np.array(corrects),
                        layers=np.array(LAYERS))
    print(f"[SAVED] {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="e.g. vlm_direction_testbed_R2R_4way_1500_obj_place")
    ap.add_argument("--model", default="baseline", choices=["baseline", "vanilla", "delta"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0)
    run(ap.parse_args())
