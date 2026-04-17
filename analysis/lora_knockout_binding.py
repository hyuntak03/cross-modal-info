"""
LoRA Knockout × Binding Gap 분석.

Binding gap = Direction probe acc − Letter probe acc (answer token last layer).
  - Baseline 원래: direction 91%, letter 79% → gap 12
  - Layer L knockout 후 gap이 증가하면 → L의 LoRA가 binding에 중요

각 layer knockout 별로:
  1. Baseline weight에서 layer L delta 제거 (→ Vanilla 수준 복원)
  2. Forward pass on N samples → last layer answer token features
  3. Direction probe 학습 + Letter probe 학습 → binding gap 측정
  4. Restore

Usage:
  CUDA_VISIBLE_DEVICES=0 python analysis/lora_knockout_binding.py \
    --layers "0-3,4-7,8-11,12-15,16-19,20-23,24-27" \
    --n_per_task 500
"""

import os, sys, json, argparse, gc, re, ast, string
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

VANILLA_ARGS_GPU = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
VANILLA_ARGS_CPU = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=cpu,force_sample=True"

BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"

MCQ_JSON_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way_1500"

TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"
DIRECTIONS = ["Down", "Left", "Right", "Up"]  # alphabetical (LabelEncoder default)

TARGET_MODULE_SUFFIXES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def resolve_answer(line):
    """letter answer (A/B/C/D) → direction text via candidates."""
    answer = str(line["answer"]).strip()
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        candidates_raw = line.get("candidates", [])
        if isinstance(candidates_raw, str):
            candidates_raw = ast.literal_eval(candidates_raw)
        idx = ord(answer.upper()) - ord('A')
        if idx < len(candidates_raw):
            answer = str(candidates_raw[idx]).strip()
    return answer


# ============================================================
#  Delta 계산 & Knockout (lora_knockout.py와 동일)
# ============================================================

def compute_lora_deltas(vanilla_model, baseline_model):
    deltas = {}
    vanilla_params = dict(vanilla_model.named_parameters())
    for name, p_b in baseline_model.named_parameters():
        if "model.layers." not in name:
            continue
        if not any(s in name for s in TARGET_MODULE_SUFFIXES):
            continue
        if name in vanilla_params:
            p_v = vanilla_params[name]
            # Vanilla(CPU) vs Baseline(GPU) → compute on CPU
            delta = p_b.data.float().cpu() - p_v.data.float().cpu()
            if delta.abs().max() > 1e-5:
                deltas[name] = delta
    return deltas


def get_layer_idx(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def knockout_layers(baseline_model, deltas, layer_indices):
    layer_set = set(layer_indices)
    applied = {}
    for name, p in baseline_model.named_parameters():
        li = get_layer_idx(name)
        if li is None or li not in layer_set:
            continue
        if name not in deltas:
            continue
        delta_gpu = deltas[name].to(p.device, p.dtype)
        p.data -= delta_gpu
        applied[name] = delta_gpu
    return applied


def restore(baseline_model, applied):
    for name, p in baseline_model.named_parameters():
        if name in applied:
            p.data += applied[name]


# ============================================================
#  Feature 추출 + Probing
# ============================================================

@torch.no_grad()
def extract_features_and_labels(model, tokenizer, image_processor, conv_template, task, n_samples):
    """Last layer answer token features + direction / letter labels."""
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions

    # Load MCQ for letter labels
    mcq_by_id = {m["id"]: m for m in json.load(open(os.path.join(MCQ_JSON_ROOT, f"{task}.json")))}

    questions, _ = load_dataset_as_questions(task_name=TASK_FULL(task), limit=n_samples)
    dl = create_data_loader(
        questions, "", 1, 4, tokenizer, image_processor, model.config,
        TASK_FULL(task), conv_template, video_folder="", video_fps=1,
        frames_upbound=8, force_sample=True,
    )

    feats = []
    dir_labels = []
    letter_labels = []

    for (input_ids, image_tensor, image_sizes, prompts, mask_tensor, modality), line in tqdm(
        zip(dl, questions), total=len(questions), desc="  feat", leave=False
    ):
        sid = int(str(line["q_id"]).split("_")[0])
        mcq = mcq_by_id.get(sid)
        if mcq is None:
            continue
        letter = str(mcq["answer"]).strip().upper()
        # Direction text는 resolve_answer로: line["answer"] (A/B/C/D) → candidates 매핑 → "Right" 등
        direction = resolve_answer(line).strip().capitalize()
        if direction not in DIRECTIONS or letter not in ["A", "B", "C", "D"]:
            continue

        input_ids = input_ids.to('cuda')
        image_tensor = [t.to('cuda') for t in image_tensor]

        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor,
                modalities=[modality], image_sizes=image_sizes,
            )
        output = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, output_hidden_states=True, return_dict=True,
        )
        # last layer's last token
        h_last = output.hidden_states[-1][0, -1, :].cpu().to(torch.float16)
        feats.append(h_last)
        dir_labels.append(DIRECTIONS.index(direction))
        letter_labels.append("ABCD".index(letter))

    feats = torch.stack(feats).float().numpy()  # (N, D)
    dir_labels = np.array(dir_labels)
    letter_labels = np.array(letter_labels)
    return feats, dir_labels, letter_labels


def gpu_probe(X, y, nc, seed=42, epochs=50, lr=1e-3, weight_decay=1e-2, test_ratio=0.3):
    device = torch.device("cuda")
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    m = X_t.mean(0); s = X_t.std(0); s[s < 1e-8] = 1
    X_t = (X_t - m) / s
    nt = max(1, int(len(X) * test_ratio))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    p = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    Xtr, ytr = X_t[p[:-nt]], y_t[p[:-nt]]
    Xte, yte = X_t[p[-nt:]], y_t[p[-nt:]]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = nn.Linear(X.shape[1], nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 128):
            b = idx[i:i+128]
            loss = crit(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
    return acc


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0-3,4-7,8-11,12-15,16-19,20-23,24-27",
                        help="comma-separated layer specs ('0-3' or '14')")
    parser.add_argument("--tasks", default="shape_color,obj_place")
    parser.add_argument("--n_per_task", type=int, default=500)
    parser.add_argument("--output", default="analysis/lora_knockout_binding.json")
    args = parser.parse_args()

    tasks = args.tasks.split(",")

    # Parse layer specs
    specs = []
    for spec in args.layers.split(","):
        spec = spec.strip()
        if "-" in spec:
            a, b = spec.split("-")
            specs.append((f"L{a}-{b}", list(range(int(a), int(b) + 1))))
        else:
            specs.append((f"L{spec}", [int(spec)]))

    # Load Vanilla (CPU, temporary)
    print("Loading Vanilla (CPU)...")
    _, vanilla_model, _, _, _, _ = load_model(VANILLA_ARGS_CPU)
    vanilla_model.eval()

    # Load Baseline (GPU)
    print("Loading Baseline (GPU)...")
    b_args_str = f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS_GPU}"
    tokenizer, baseline_model, image_processor, _, _, conv_template = load_model(b_args_str)
    baseline_model.eval()

    # Compute deltas
    print("Computing deltas...")
    deltas = compute_lora_deltas(vanilla_model, baseline_model)
    print(f"  {len(deltas)} params with non-zero delta")
    del vanilla_model; gc.collect(); torch.cuda.empty_cache()

    results = {"reference": {}, "knockouts": {}}

    # Reference (no knockout)
    print("\n[Reference (no knockout)]")
    for task in tasks:
        feats, dl, ll = extract_features_and_labels(
            baseline_model, tokenizer, image_processor, conv_template, task, args.n_per_task
        )
        dir_acc = gpu_probe(feats, dl, 4)
        letter_acc = gpu_probe(feats, ll, 4)
        gap = dir_acc - letter_acc
        results["reference"][task] = {"dir_acc": dir_acc, "letter_acc": letter_acc, "gap": gap, "n": len(feats)}
        print(f"  {task}: dir={dir_acc:.1f}% letter={letter_acc:.1f}% gap={gap:+.1f} (n={len(feats)})")

    # Knockouts
    for label, layers_list in specs:
        print(f"\n[Knockout {label} (layers={layers_list})]")
        applied = knockout_layers(baseline_model, deltas, layers_list)
        print(f"  {len(applied)} params zeroed")

        task_accs = {}
        for task in tasks:
            feats, dl, ll = extract_features_and_labels(
                baseline_model, tokenizer, image_processor, conv_template, task, args.n_per_task
            )
            dir_acc = gpu_probe(feats, dl, 4)
            letter_acc = gpu_probe(feats, ll, 4)
            gap = dir_acc - letter_acc

            ref = results["reference"][task]
            dir_drop = ref["dir_acc"] - dir_acc
            letter_drop = ref["letter_acc"] - letter_acc
            gap_delta = gap - ref["gap"]

            task_accs[task] = {
                "dir_acc": dir_acc, "letter_acc": letter_acc, "gap": gap,
                "dir_drop": dir_drop, "letter_drop": letter_drop, "gap_delta": gap_delta,
            }
            print(f"  {task}: dir={dir_acc:.1f}% ({dir_drop:+.1f}) | "
                  f"letter={letter_acc:.1f}% ({letter_drop:+.1f}) | "
                  f"gap={gap:+.1f} (Δ{gap_delta:+.1f})")

        results["knockouts"][label] = {"layers": layers_list, "tasks": task_accs}
        restore(baseline_model, applied)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY — Binding Gap Change per Layer Knockout\n{'='*70}")
    print(f"  {'Layer':>10} ", end="")
    for task in tasks:
        print(f" {task[:15]:>32}", end="")
    print()
    print(f"  {'':>10} ", end="")
    for task in tasks:
        print(f" {'dir(Δ) let(Δ) gap(Δ)':>32}", end="")
    print()
    print(f"  {'ref':>10} ", end="")
    for task in tasks:
        r = results["reference"][task]
        print(f" {r['dir_acc']:>7.1f}      {r['letter_acc']:>7.1f}      {r['gap']:>+6.1f}     ", end="")
    print()
    for label, entry in results["knockouts"].items():
        print(f"  {label:>10} ", end="")
        for task in tasks:
            r = entry["tasks"][task]
            print(f" {r['dir_acc']:>5.1f}({r['dir_drop']:+4.1f}) {r['letter_acc']:>5.1f}({r['letter_drop']:+4.1f}) {r['gap']:>+5.1f}({r['gap_delta']:+4.1f})", end="")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
