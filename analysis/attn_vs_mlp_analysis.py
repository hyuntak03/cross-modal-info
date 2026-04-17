"""
Attention vs MLP Contribution 분리.

각 layer에서:
  h_after_attn = h_{l-1} + self_attn(layernorm(h_{l-1}))  ← attention만
  h_l = h_after_attn + mlp(layernorm(h_after_attn))       ← MLP 추가

Last token position에서 h_after_attn과 h_l 각각 direction/identity probe.

If direction jumps at h_after_attn → V projection이 direction 추출
If direction jumps at h_l but not h_after_attn → MLP가 direction 생성

Usage:
    # 단일 모델/task
    CUDA_VISIBLE_DEVICES=0 python analysis/attn_vs_mlp_analysis.py \
        --model llava-video-7b_lora_4combo_v2_baseline --task obj_place

    # 전체 (3모델 × 2task)
    CUDA_VISIBLE_DEVICES=0 python analysis/attn_vs_mlp_analysis.py \
        --model all --task all
"""

import os, sys, json, argparse, math
import numpy as np
import torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"
LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "llava-video-7b_lora_4combo_v2_delta": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
}
ALL_MODELS = ["llava-video-7b", "llava-video-7b_lora_4combo_v2_baseline", "llava-video-7b_lora_4combo_v2_delta"]
ALL_TASKS = ["shape_color", "obj_place"]

def model_short(name):
    """결과 파일명용 약칭."""
    return name.replace("llava-video-7b_lora_", "").replace("llava-video-7b", "vanilla")
IDENTITY_ATTRS = {"shape_color":"shape","obj_color":"obj_class","shape_place":"place_class","obj_place":"obj_class"}
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
DIRECTIONS = ["up","down","left","right"]

def gpu_probe(X, y, nc, seed=42, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = X.shape
    X_t = torch.from_numpy(X).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y).long().to(device)
    m=X_t.mean(0);X_t-=m;s=X_t.std(0);s[s<1e-8]=1;X_t/=s
    nt=max(1,int(N*0.3))
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    p=torch.randperm(N,generator=torch.Generator().manual_seed(seed))
    Xtr,ytr,Xte,yte=X_t[p[:-nt]],y_t[p[:-nt]],X_t[p[-nt:]],y_t[p[-nt:]]
    del X_t,y_t
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=nn.Linear(D,nc).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    crit=nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        idx=torch.randperm(len(Xtr),device=device)
        for i in range(0,len(Xtr),64):
            b=idx[i:i+64];loss=crit(model(Xtr[b]),ytr[b]);opt.zero_grad();loss.backward();opt.step()
    model.eval()
    with torch.no_grad(): acc=(model(Xte).argmax(1)==yte).float().mean().item()*100
    del model,opt;torch.cuda.empty_cache()
    return acc

def load_metadata(task):
    with open(os.path.join(META_ROOT,f"{task}_metadata.json")) as f: return json.load(f)

def get_labels(metadata, qids, attr):
    mb={m['id']:m for m in metadata};le=LabelEncoder()
    raw=[str(mb[int(str(q).split('_')[0])][attr]) for q in qids]
    return le.fit_transform(raw),len(le.classes_)

def load_model(model_name):
    from core.model_loader import parse_model_args, load_model_from_args
    if model_name=="llava-video-7b":
        a="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    else:
        lp=LORA_PATHS[model_name]
        a=f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    ma=parse_model_args(a);tok,model,ip,cl,mn,ct=load_model_from_args(ma)
    model.eval();return tok,model,ip,mn,ct

@torch.no_grad()
def extract_attn_vs_mlp(model, tokenizer, task, conv_template, image_processor, metadata, limit=200):
    """각 layer에서 post-attn / post-mlp last token hidden state 추출."""
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions
    from llava.constants import IMAGE_TOKEN_INDEX

    questions,_=load_dataset_as_questions(task_name=TASK_FULL(task),limit=limit)
    dl=create_data_loader(questions,"",1,2,tokenizer,image_processor,model.config,
                          TASK_FULL(task),conv_template,video_folder="",video_fps=1,
                          frames_upbound=8,force_sample=True)

    n_layers=model.config.num_hidden_layers
    meta_by_id={m['id']:m for m in metadata}

    # Storage: per layer, per sample → (after_attn, after_mlp) last token
    all_after_attn={l:[] for l in range(n_layers)}
    all_after_mlp={l:[] for l in range(n_layers)}
    all_qids=[]

    for (input_ids,image_tensor,image_sizes,prompts,mask_tensor,modality),line in tqdm(
        zip(dl,questions),total=len(questions),desc=f"  {task}"):

        all_qids.append(line['q_id'])
        input_ids=input_ids.to('cuda')
        image_tensor=[t.to('cuda') for t in image_tensor]

        (_,position_ids,attention_mask,_,inputs_embeds,_)=\
            model.prepare_inputs_labels_for_multimodal(
                input_ids,None,None,None,None,image_tensor,[modality],image_sizes=image_sizes)

        # Hook: capture intermediate state after self_attn (before MLP)
        intermediate={}
        hooks=[]

        for li in range(n_layers):
            layer_module=model.model.layers[li]

            def make_attn_hook(layer_idx):
                def hook_fn(module, input, output):
                    # self_attn returns (attn_output, attn_weights, past_kv) or tuple
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # h_after_attn = residual + attn_out
                    # But we need the residual (input to the layer)
                    # The layer's forward: residual=hidden, hidden=layernorm(hidden), hidden=self_attn(hidden), hidden=residual+hidden
                    # So at self_attn hook, output[0] is the attn output (before residual add)
                    intermediate[f"attn_{layer_idx}"] = attn_out[0, -1, :].detach().cpu().to(torch.float16)
                return hook_fn

            hooks.append(layer_module.self_attn.register_forward_hook(make_attn_hook(li)))

            def make_layer_hook(layer_idx):
                def hook_fn(module, input, output):
                    # Layer input = h_{l-1}, layer output = h_l
                    h_in = input[0] if isinstance(input, tuple) else input
                    h_out = output[0] if isinstance(output, tuple) else output

                    # h_after_attn = h_in + attn_output (we have attn_output from attn hook)
                    attn_out = intermediate.get(f"attn_{layer_idx}")
                    if attn_out is not None:
                        h_after_attn = h_in[0, -1, :].detach().cpu().to(torch.float16) + attn_out
                        all_after_attn[layer_idx].append(h_after_attn)

                    # h_l = full layer output
                    all_after_mlp[layer_idx].append(h_out[0, -1, :].detach().cpu().to(torch.float16))
                return hook_fn

            hooks.append(model.model.layers[li].register_forward_hook(make_layer_hook(li)))

        # Forward
        outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                      position_ids=position_ids,output_attentions=False,return_dict=True)
        del outputs

        for h in hooks: h.remove()
        intermediate.clear()

        if len(all_qids)%50==0: torch.cuda.empty_cache()

    return all_after_attn, all_after_mlp, all_qids, n_layers

def run_analysis(all_after_attn, all_after_mlp, all_qids, n_layers, task, metadata, output_dir, model_name=""):
    """Post-attn vs post-MLP direction/identity probe."""
    id_attr = IDENTITY_ATTRS[task]
    dl, dnc = get_labels(metadata, all_qids, "direction")
    il, inc = get_labels(metadata, all_qids, id_attr)

    results = {"model": model_name, "task": task, "layers":[], "after_attn_dir":[], "after_mlp_dir":[], "after_attn_id":[], "after_mlp_id":[]}

    print(f"\n  {'Layer':>6} {'Attn→dir':>10} {'MLP→dir':>10} {'Δ(MLP-Attn)':>12} {'Attn→id':>10} {'MLP→id':>10}")

    for l in range(n_layers):
        if not all_after_attn[l] or not all_after_mlp[l]: continue

        feat_attn = torch.stack(all_after_attn[l]).numpy().astype(np.float32)
        feat_mlp = torch.stack(all_after_mlp[l]).numpy().astype(np.float32)

        attn_dir = gpu_probe(feat_attn.copy(), dl, dnc)
        mlp_dir = gpu_probe(feat_mlp.copy(), dl, dnc)
        attn_id = gpu_probe(feat_attn.copy(), il, inc)
        mlp_id = gpu_probe(feat_mlp.copy(), il, inc)

        delta = mlp_dir - attn_dir
        results["layers"].append(l)
        results["after_attn_dir"].append(attn_dir)
        results["after_mlp_dir"].append(mlp_dir)
        results["after_attn_id"].append(attn_id)
        results["after_mlp_id"].append(mlp_id)

        marker = " ★★★" if delta > 10 else (" ★" if delta > 5 else "")
        print(f"  {l:>6} {attn_dir:>9.1f}% {mlp_dir:>9.1f}% {delta:>+11.1f}%p {attn_id:>9.1f}% {mlp_id:>9.1f}%{marker}")

        del feat_attn, feat_mlp

    os.makedirs(output_dir, exist_ok=True)
    short = model_short(model_name)
    sp = os.path.join(output_dir, f"attn_vs_mlp_{short}_{task}.json")
    with open(sp,"w") as f: json.dump(results, f, indent=2)
    print(f"\n  [SAVED] {sp}")
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task",default="obj_place")
    parser.add_argument("--limit",type=int,default=200)
    parser.add_argument("--output_dir",default="analysis/attn_vs_mlp_results")
    args=parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    os.makedirs(args.output_dir,exist_ok=True)

    for model_name in models:
        print(f"\nLoading model: {model_name}")
        tok,model,ip,mn,ct=load_model(model_name)

        for task in tasks:
            print(f"\n{'#'*60}\n  {model_name} / {task}\n{'#'*60}")
            metadata=load_metadata(task)
            aa,am,qids,nl=extract_attn_vs_mlp(model,tok,task,ct,ip,metadata,args.limit)
            run_analysis(aa,am,qids,nl,task,metadata,args.output_dir,model_name=model_name)

        del model
        torch.cuda.empty_cache()

if __name__=="__main__":main()
