"""
Identity Filtering 파이프라인 + Decoding Gap 원인 분석.

Part 1: Vision Token per-layer identity filtering
Part 2A: FDR ratio <-> Logit Lens 상관
Part 2B: Direction subspace projection -> logit lens
Part 2C: Logit margin analysis

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/decoding_gap_analysis.py \
        --model llava-video-7b_lora_4combo_v2_baseline
"""

import os, sys, json, argparse
import numpy as np
import torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"
META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"
MCQ_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way"

FEAT_ROOTS = {
    "llava-video-7b": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b",
    "llava-video-7b_lora_syn_v4_baseline": "/data3/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_syn_v4_baseline",
    "llava-video-7b_lora_4combo_v2_baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
}
LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
    "llava-video-7b_lora_syn_v4_baseline": "/data/jong980812/project/LLaVA-NeXT/work_dirs/LLaVA-Video-7B-Qwen2/llava-video-7b-qwen2_lora-r64_motion_v4_100K_baseline_rope1d_f8_ep1_lr1e-5",
}
LETTERS = ["A","B","C","D"]
IDENTITY_ATTRS = {"shape_color":"shape","obj_color":"obj_class","shape_place":"place_class","obj_place":"obj_class"}

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

def compute_fdr(X, y):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t=torch.from_numpy(X.astype(np.float32)).to(device);y_t=torch.from_numpy(y).long().to(device)
    cs=torch.unique(y_t);gm=X_t.mean(0)
    bv=torch.zeros(X_t.shape[1],device=device);wv=torch.zeros(X_t.shape[1],device=device)
    for c in cs:
        Xc=X_t[y_t==c];cm=Xc.mean(0);bv+=Xc.shape[0]*(cm-gm)**2;wv+=((Xc-cm)**2).sum(0)
    wv=wv.clamp(min=1e-10);fdr=(bv/wv).cpu().numpy()
    del X_t,y_t;torch.cuda.empty_cache()
    return fdr

def load_metadata(task):
    with open(os.path.join(META_ROOT,f"{task}_metadata.json")) as f: return json.load(f)

def load_mcq(task):
    with open(os.path.join(MCQ_ROOT,f"{task}.json")) as f: return json.load(f)

def get_labels(metadata, qids, attr):
    mb={m['id']:m for m in metadata};le=LabelEncoder()
    raw=[str(mb[int(str(q).split('_')[0])][attr]) for q in qids]
    return le.fit_transform(raw),len(le.classes_)

def load_vision_layer(feat_root, task, layer_idx):
    d=os.path.join(feat_root,"vision_token",TASK_FULL(task))
    meta=np.load(os.path.join(d,"meta.npy"),allow_pickle=True).item()
    feat=np.array(np.load(os.path.join(d,f"features_layer_{layer_idx}.npy"),mmap_mode='r'))
    qids=np.load(os.path.join(d,"qids.npy"))
    nf=meta.get("num_frames",8);tpf=meta.get("tokens_per_frame_post",196);hd=meta.get("hidden_dim",3584)
    return feat.reshape(feat.shape[0],nf,tpf,hd).mean(axis=(1,2)),qids,meta

def load_answer_layer(feat_root, task, layer_idx):
    d=os.path.join(feat_root,"answer_token",TASK_FULL(task))
    feat=np.array(np.load(os.path.join(d,f"features_layer_{layer_idx}.npy"),mmap_mode='r'))
    qids=np.load(os.path.join(d,"qids.npy"))
    meta=np.load(os.path.join(d,"meta.npy"),allow_pickle=True).item()
    return feat,qids,meta

def apply_rmsnorm(x,w,eps=1e-6):
    rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+eps);return x/rms*w

def load_model_weights(model_name):
    sys.path.insert(0,'/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
    os.environ['HF_HOME']='/data/datasets/LLaVA-Video-100K-Subset/'
    from core.model_loader import parse_model_args,load_model_from_args
    if model_name=="llava-video-7b":
        a="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"
    else:
        lp=LORA_PATHS[model_name]
        a=f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,device_map=cpu"
    ma=parse_model_args(a);tok,model,_,_,_,_=load_model_from_args(ma)
    lm_w=model.lm_head.weight.data.float().clone();norm_w=model.model.norm.weight.data.float().clone()
    lid={lt:tok.encode(lt,add_special_tokens=False)[0] for lt in LETTERS}
    del model;return lm_w,norm_w,lid,tok

# Part 1
def part1_vision_filtering(feat_root,task,num_layers):
    print(f"\n  Part1: Vision Token Filtering — {task}")
    metadata=load_metadata(task);id_attr=IDENTITY_ATTRS[task]
    results={"dir":[],"id":[],"layers":[]}
    for l in range(num_layers):
        feat,qids,_=load_vision_layer(feat_root,task,l);feat=feat.astype(np.float32)
        dl,dnc=get_labels(metadata,qids,"direction");il,inc=get_labels(metadata,qids,id_attr)
        da=gpu_probe(feat.copy(),dl,dnc);ia=gpu_probe(feat.copy(),il,inc)
        results["dir"].append(da);results["id"].append(ia);results["layers"].append(l)
        print(f"    L{l:2d}: dir={da:5.1f}%  {id_attr}={ia:5.1f}%")
    return results

# Part 2A
def part2a(feat_root,task,lm_w,norm_w,lid,num_layers):
    print(f"\n  Part2A: FDR vs Logit — {task}")
    metadata=load_metadata(task);mcq=load_mcq(task);mb={m['id']:m for m in mcq}
    id_attr=IDENTITY_ATTRS[task];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abcd=torch.tensor([lid[lt] for lt in LETTERS]).to(device)
    results={"fdr_ratio":[],"logit_acc":[],"layers":[]}
    for l in range(num_layers):
        feat,qids,_=load_answer_layer(feat_root,task,l);feat=feat.astype(np.float32)
        dl,_=get_labels(metadata,qids,"direction");il,_=get_labels(metadata,qids,id_attr)
        fd=compute_fdr(feat,dl);fi=compute_fdr(feat,il);ratio=fd.mean()/max(fi.mean(),1e-10)
        X=torch.from_numpy(feat).to(device);normed=apply_rmsnorm(X,norm_w.to(device))
        logits=normed@lm_w.to(device).T;al=logits[:,abcd]
        correct=0
        for i,qid in enumerate(qids):
            sid=int(str(qid).split('_')[0]);q=mb.get(sid)
            if not q:continue
            gt=LETTERS.index(q['answer']) if q['answer'] in LETTERS else -1
            if gt<0:continue
            if al[i].argmax().item()==gt:correct+=1
        acc=correct/len(qids)*100
        results["fdr_ratio"].append(float(ratio));results["logit_acc"].append(acc);results["layers"].append(l)
        print(f"    L{l:2d}: FDR={ratio:6.2f}x  logit={acc:5.1f}%")
        del X,logits;torch.cuda.empty_cache()
    return results

# Part 2B
def part2b(feat_root,task,lm_w,norm_w,lid):
    print(f"\n  Part2B: Subspace Projection — {task}")
    metadata=load_metadata(task);mcq=load_mcq(task);mb={m['id']:m for m in mcq}
    id_attr=IDENTITY_ATTRS[task];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abcd=torch.tensor([lid[lt] for lt in LETTERS]).to(device)
    _,_,meta=load_answer_layer(feat_root,task,0);nl=meta["num_layers"];li=nl-1
    feat,qids,_=load_answer_layer(feat_root,task,li);feat=feat.astype(np.float32)
    dl,_=get_labels(metadata,qids,"direction");il,_=get_labels(metadata,qids,id_attr)
    fd=compute_fdr(feat,dl);fi=compute_fdr(feat,il);dr=fd/np.maximum(fi,1e-10)
    results={}
    for top_k in [50,100,200,500,1000,3584]:
        dims=np.argsort(dr)[-min(top_k,len(dr)):]
        fp=np.zeros_like(feat);fp[:,dims]=feat[:,dims]
        X=torch.from_numpy(fp).to(device);normed=apply_rmsnorm(X,norm_w.to(device))
        logits=normed@lm_w.to(device).T;al=logits[:,abcd]
        correct=0
        for i,qid in enumerate(qids):
            sid=int(str(qid).split('_')[0]);q=mb.get(sid)
            if not q:continue
            gt=LETTERS.index(q['answer']) if q['answer'] in LETTERS else -1
            if gt<0:continue
            if al[i].argmax().item()==gt:correct+=1
        acc=correct/len(qids)*100;results[f"top_{top_k}"]=acc
        print(f"    Top-{top_k:>4}: logit={acc:5.1f}%")
        del X,logits;torch.cuda.empty_cache()
    return results

# Part 2C
def part2c(feat_root,task,lm_w,norm_w,lid):
    print(f"\n  Part2C: Logit Margin — {task}")
    mcq=load_mcq(task);mb={m['id']:m for m in mcq}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abcd=torch.tensor([lid[lt] for lt in LETTERS]).to(device)
    _,_,meta=load_answer_layer(feat_root,task,0);li=meta["num_layers"]-1
    feat,qids,_=load_answer_layer(feat_root,task,li)
    X=torch.from_numpy(feat.astype(np.float32)).to(device)
    normed=apply_rmsnorm(X,norm_w.to(device));logits=normed@lm_w.to(device).T;al=logits[:,abcd]
    mc,mw=[],[]
    for i,qid in enumerate(qids):
        sid=int(str(qid).split('_')[0]);q=mb.get(sid)
        if not q:continue
        gt=LETTERS.index(q['answer']) if q['answer'] in LETTERS else -1
        if gt<0:continue
        gl=al[i,gt].item();om=al[i].clone();om[gt]=float('-inf');margin=gl-om.max().item()
        if al[i].argmax().item()==gt:mc.append(margin)
        else:mw.append(margin)
    mc=np.array(mc) if mc else np.array([0]);mw=np.array(mw) if mw else np.array([0])
    print(f"    Correct({len(mc):>3}): margin={mc.mean():.3f}±{mc.std():.3f}")
    print(f"    Wrong  ({len(mw):>3}): margin={mw.mean():.3f}±{mw.std():.3f}")
    del X,logits;torch.cuda.empty_cache()
    return {"correct_mean":float(mc.mean()),"correct_std":float(mc.std()),"n_correct":len(mc),
            "wrong_mean":float(mw.mean()),"wrong_std":float(mw.std()),"n_wrong":len(mw)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task",default="all")
    parser.add_argument("--output_dir",default="analysis/decoding_gap_results")
    args=parser.parse_args()
    feat_root=FEAT_ROOTS.get(args.model)
    if not feat_root:print(f"[ERROR] {args.model}");return
    tasks=TASKS if args.task=="all" else [args.task]
    os.makedirs(args.output_dir,exist_ok=True)
    print("Loading model weights...")
    lm_w,norm_w,lid,tok=load_model_weights(args.model)
    all_r={}
    for task in tasks:
        print(f"\n{'#'*60}\n  {args.model} / {task}\n{'#'*60}")
        _,_,meta=load_answer_layer(feat_root,task,0);nl=meta["num_layers"]
        tr={}
        tr["part1"]=part1_vision_filtering(feat_root,task,nl)
        tr["part2a"]=part2a(feat_root,task,lm_w,norm_w,lid,nl)
        tr["part2b"]=part2b(feat_root,task,lm_w,norm_w,lid)
        tr["part2c"]=part2c(feat_root,task,lm_w,norm_w,lid)
        all_r[task]=tr
    sp=os.path.join(args.output_dir,f"gap_analysis_{args.model}.json")
    def conv(o):
        if isinstance(o,(np.floating,np.integer)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        if isinstance(o,dict):return{k:conv(v) for k,v in o.items()}
        if isinstance(o,list):return[conv(v) for v in o]
        return o
    with open(sp,"w") as f:json.dump(conv(all_r),f,indent=2)
    print(f"\n[SAVED] {sp}")

if __name__=="__main__":main()
