"""
Measure lm_head alignment with canonical direction axis Δ̂_d_L21.

Tests the claim: "canonical axis at L21 is the readout axis" — i.e., hidden
direction content on Δ̂_d_L21 translates to letter logits via lm_head.

If `cos(W_lm["A"], Δ̂_up_L21)` is high and positive, and other letters have
low/negative cos, then yes — lm_head directly reads from canonical direction axis.

Canonical MCQ: Up=A, Right=B, Down=C, Left=D.
"""
import numpy as np
import torch, sys, os, glob
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/cross-modal-info')
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

BASELINE_LORA = "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5"
VANILLA_ARGS = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"


def load_model(args_str):
    from core.model_loader import parse_model_args, load_model_from_args
    a = parse_model_args(args_str)
    return load_model_from_args(a)


def load_factorial(cond):
    arr = {'hiddens':[], 'directions':[]}
    for f in sorted(glob.glob(f'/local_datasets/vlm_direction/factorial_dataset/hiddens/baseline_{cond}_4variants*.npz')):
        d = np.load(f, allow_pickle=True)
        for k in arr: arr[k].append(d[k])
    return {k: np.concatenate(v) for k, v in arr.items()}


def main():
    print("[load Baseline model]")
    tok, model, _, _, _, _ = load_model(f'lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    print(f"lm_head weight shape: {W.shape}")

    # Letter token IDs
    letter_ids = {}
    for ltr in ['A', 'B', 'C', 'D']:
        for cand in [ltr, ' ' + ltr]:
            tids = tok.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                letter_ids[ltr] = tids[0]; break
    print(f"Letter token IDs: {letter_ids}")

    # Compute Δ̂_d_L21 from factorial OP
    OP = load_factorial('obj_place')
    H = OP['hiddens'].astype(np.float32)
    dirs = OP['directions']
    g_L21 = H.mean(0)[21]
    Delta = {dn: H[dirs == dn].mean(0)[21] - g_L21 for dn in ['up','right','down','left']}
    Delta_hat = {dn: v/(np.linalg.norm(v)+1e-9) for dn,v in Delta.items()}

    # Also from SC (should be similar at L21)
    SC = load_factorial('shape_color')
    Hs = SC['hiddens'].astype(np.float32)
    dirs_s = SC['directions']
    g_L21_sc = Hs.mean(0)[21]
    Delta_sc = {dn: Hs[dirs_s == dn].mean(0)[21] - g_L21_sc for dn in ['up','right','down','left']}
    Delta_hat_sc = {dn: v/(np.linalg.norm(v)+1e-9) for dn,v in Delta_sc.items()}

    canon = {'up':'A', 'right':'B', 'down':'C', 'left':'D'}
    letters = ['A','B','C','D']
    directions_list = ['up','right','down','left']

    print("\n" + "=" * 90)
    print("cos(W_lm[letter], Δ̂_d_L21_OP)  — raw letter token weight vs OP L21 direction axis")
    print("=" * 90)
    print(f"{'Direction':>10} | {'A':>10} | {'B':>10} | {'C':>10} | {'D':>10}")
    for d in directions_list:
        row = f"{d:>10} |"
        for l in letters:
            w = W[letter_ids[l]]
            c = float(Delta_hat[d] @ w / (np.linalg.norm(w) + 1e-9))
            mark = '*' if canon[d] == l else ' '
            row += f" {mark}{c:+8.4f}{mark} |"
        print(row)

    print("\n" + "=" * 90)
    print("cos(W_lm[letter] - mean(W_lm[ABCD]), Δ̂_d_L21_OP)  — centered letter weight")
    print("=" * 90)
    letter_tokens = [letter_ids[l] for l in letters]
    g_W = W[letter_tokens].mean(0)
    print(f"{'Direction':>10} | {'A':>10} | {'B':>10} | {'C':>10} | {'D':>10}")
    for d in directions_list:
        row = f"{d:>10} |"
        for l in letters:
            w = W[letter_ids[l]] - g_W
            c = float(Delta_hat[d] @ w / (np.linalg.norm(w) + 1e-9))
            mark = '*' if canon[d] == l else ' '
            row += f" {mark}{c:+8.4f}{mark} |"
        print(row)

    print("\n" + "=" * 90)
    print("Direction contrast axis: W_A-W_C vs Δ̂_up-Δ̂_down, W_B-W_D vs Δ̂_right-Δ̂_left")
    print("=" * 90)
    W_ud = W[letter_ids['A']] - W[letter_ids['C']]
    D_ud = Delta_hat['up'] - Delta_hat['down']
    c1 = float(W_ud @ D_ud / (np.linalg.norm(W_ud) * np.linalg.norm(D_ud) + 1e-9))
    print(f"cos(W[A]-W[C], Δ̂_up - Δ̂_down) = {c1:+.4f}")

    W_rl = W[letter_ids['B']] - W[letter_ids['D']]
    D_rl = Delta_hat['right'] - Delta_hat['left']
    c2 = float(W_rl @ D_rl / (np.linalg.norm(W_rl) * np.linalg.norm(D_rl) + 1e-9))
    print(f"cos(W[B]-W[D], Δ̂_right - Δ̂_left) = {c2:+.4f}")

    # Same with SC axes (should be similar since cos(OP, SC)=0.94)
    print("\nUsing SC axes for cross-check:")
    D_ud_sc = Delta_hat_sc['up'] - Delta_hat_sc['down']
    c1s = float(W_ud @ D_ud_sc / (np.linalg.norm(W_ud) * np.linalg.norm(D_ud_sc) + 1e-9))
    print(f"cos(W[A]-W[C], Δ̂_up_SC - Δ̂_down_SC) = {c1s:+.4f}")
    D_rl_sc = Delta_hat_sc['right'] - Delta_hat_sc['left']
    c2s = float(W_rl @ D_rl_sc / (np.linalg.norm(W_rl) * np.linalg.norm(D_rl_sc) + 1e-9))
    print(f"cos(W[B]-W[D], Δ̂_right_SC - Δ̂_left_SC) = {c2s:+.4f}")

    # Also: lm_head read via RMSNorm first, let's check post-norm alignment
    # Qwen2 applies model.norm before lm_head. The norm is RMSNorm.
    norm = model.model.norm
    print(f"\nRMSNorm weight: {norm.weight.detach().cpu().float().numpy()[:5]}...  (mean={norm.weight.detach().cpu().float().numpy().mean():.3f}, std={norm.weight.detach().cpu().float().numpy().std():.3f})")

    # Effective readout for direction axis is cos(W_lm, norm(Δ̂))
    # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    # For unit vector Δ̂: it gets element-wise scaled by weight/||Δ̂||_RMS
    norm_w = norm.weight.detach().cpu().float().numpy()
    print(f"\n" + "=" * 90)
    print("Effective direction-to-letter projection via RMSNorm * lm_head")
    print("=" * 90)
    print(f"{'Direction':>10} | {'A':>10} | {'B':>10} | {'C':>10} | {'D':>10}")
    for d in directions_list:
        # Apply RMSNorm approximation: scale by weight, normalize
        v = Delta[d]  # use unnormalized Delta (has magnitude info)
        v_rms = np.sqrt(np.mean(v ** 2) + 1e-6)
        v_norm = v * norm_w / v_rms
        row = f"{d:>10} |"
        for l in letters:
            w = W[letter_ids[l]]
            # This represents the actual logit contribution: v_norm @ w
            dot = float(v_norm @ w)
            mark = '*' if canon[d] == l else ' '
            row += f" {mark}{dot:+10.2f}{mark} |"
        print(row)


if __name__ == "__main__":
    main()
