"""
Answer token에서 Direction probe vs Letter probe 비교.

Vanilla의 direction probe는 높은데 MCQ acc는 낮다 → letter mapping을 못 하는가?
Letter probe를 직접 학습해서 확인.

Direction: Up/Down/Left/Right (4-class, chance=25%)
Letter: A/B/C/D/E/F/G/H (현재 8-way feature이므로 8-class, chance=12.5%)
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

FEAT_ROOTS = {
    "Vanilla": "/data3/local_datasets/vlm_direction/linear_probing_4way_1500/llava-video-7b",
    "Baseline": "/data3/local_datasets/vlm_direction/linear_probing_4way_1500/llava-video-7b_lora_4combo_v2_baseline",
    "Delta": "/data3/local_datasets/vlm_direction/linear_probing_4way_1500/llava-video-7b_lora_4combo_v2_delta",
}

MCQ_JSON_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way_1500"
TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_1500_{t}"

KEY_LAYERS = list(range(29))  # 전 layer


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
        for i in range(0, len(Xtr), 256):
            b = idx[i:i+256]
            loss = crit(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
    return acc


def load_letter_labels(task):
    """MCQ JSON에서 각 sample의 letter answer 로드."""
    with open(os.path.join(MCQ_JSON_ROOT, f"{task}.json")) as f:
        data = json.load(f)
    return {m["id"]: str(m["answer"]).strip().upper() for m in data}


def main():
    results = {}
    for model_name, feat_root in FEAT_ROOTS.items():
        if not os.path.exists(feat_root):
            continue
        results[model_name] = {}
        print(f"\n{'='*80}\n  {model_name}\n{'='*80}")

        for task in TASKS:
            print(f"\n  [{task}]")
            answer_dir = os.path.join(feat_root, "answer_token", TASK_FULL(task))
            qids = np.load(os.path.join(answer_dir, "qids.npy"))
            dir_labels_raw = np.load(os.path.join(answer_dir, "labels.npy"))  # 이미 direction label encoded

            # Letter labels
            letter_map = load_letter_labels(task)
            letters = []
            valid_idx = []
            for i, q in enumerate(qids):
                sid = int(str(q).split("_")[0])
                if sid in letter_map:
                    letters.append(letter_map[sid])
                    valid_idx.append(i)
            letters = np.array(letters)
            valid_idx = np.array(valid_idx)

            le = LabelEncoder()
            letter_enc = le.fit_transform(letters)
            n_letter_classes = len(le.classes_)

            print(f"    {len(qids)} samples, {n_letter_classes} unique letters: {list(le.classes_)}")

            task_results = {}
            for layer in KEY_LAYERS:
                feat = np.load(os.path.join(answer_dir, f"features_layer_{layer}.npy")).astype(np.float32)
                feat = feat[valid_idx]
                dl = dir_labels_raw[valid_idx]
                ll = letter_enc

                acc_dir = gpu_probe(feat, dl, 4)
                acc_letter = gpu_probe(feat, ll, n_letter_classes)
                gap = acc_dir - acc_letter

                task_results[f"L{layer}"] = {
                    "direction_acc": acc_dir, "letter_acc": acc_letter,
                    "gap": gap, "n_letter_classes": n_letter_classes,
                }
                chance_dir = 25.0
                chance_letter = 100 / n_letter_classes
                print(f"    L{layer:2d}: dir={acc_dir:5.1f}% (chance {chance_dir:.1f}) | "
                      f"letter={acc_letter:5.1f}% (chance {chance_letter:.1f}) | "
                      f"gap={gap:+5.1f}")

            results[model_name][task] = task_results

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/letter_vs_direction_probing.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] analysis/letter_vs_direction_probing.json")


if __name__ == "__main__":
    main()
