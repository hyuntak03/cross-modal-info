"""
Delta_direct 효과 분석: Baseline vs Delta 비교.

Exp1: Projector Temporal Delta — Direction Discriminability
Exp2: Projector Weight Delta 비교
Exp3: Layer-wise Identity Filtering 비교 (Answer Token)
Exp4: Delta Direction Head 예측 정확도

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/delta_effect_analysis.py
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

FEAT_ROOTS = {
    "baseline": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_baseline",
    "delta": "/data2/local_datasets/vlm_direction/linear_probing/llava-video-7b_lora_4combo_v2_delta",
}

NLT_PATHS = {
    "baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5/non_lora_trainables.bin",
    "delta": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5/non_lora_trainables.bin",
}

LORA_PATHS = {
    "baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5/adapter_model.safetensors",
    "delta": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5/adapter_model.safetensors",
}

IDENTITY_ATTRS = {"shape_color": "shape", "obj_color": "obj_class", "shape_place": "place_class", "obj_place": "obj_class"}

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

def get_labels(metadata, qids, attr):
    mb={m['id']:m for m in metadata};le=LabelEncoder()
    raw=[str(mb[int(str(q).split('_')[0])][attr]) for q in qids]
    return le.fit_transform(raw),len(le.classes_)

# Exp1
def exp1_projector_temporal(task):
    print(f"\n  Exp1: Projector Temporal Delta — {task}")
    metadata = load_metadata(task); id_attr = IDENTITY_ATTRS[task]
    results = {}
    for mkey, feat_root in FEAT_ROOTS.items():
        ap_dir = os.path.join(feat_root, "after_projector", TASK_FULL(task))
        if not os.path.exists(os.path.join(ap_dir, "features.npy")):
            print(f"    [{mkey}] SKIP"); continue
        feat = np.array(np.load(os.path.join(ap_dir, "features.npy"), mmap_mode='r'))
        qids = np.load(os.path.join(feat_root, "vision_token", TASK_FULL(task), "qids.npy"))
        vt_meta = np.load(os.path.join(feat_root, "vision_token", TASK_FULL(task), "meta.npy"), allow_pickle=True).item()
        nf=vt_meta.get("num_frames",8);tpf=vt_meta.get("tokens_per_frame_post",196);hd=vt_meta.get("hidden_dim",3584)
        N=feat.shape[0]; feat_4d=feat.reshape(N,nf,tpf,hd)
        fm=feat_4d.mean(axis=2); deltas=(fm[:,1:,:]-fm[:,:-1,:]).mean(axis=1).astype(np.float32)
        dl,dnc=get_labels(metadata,qids,"direction"); il,inc=get_labels(metadata,qids,id_attr)
        da=gpu_probe(deltas.copy(),dl,dnc); ia=gpu_probe(deltas.copy(),il,inc)
        fdr_d=compute_fdr(deltas,dl);fdr_i=compute_fdr(deltas,il)
        ratio=fdr_d.mean()/max(fdr_i.mean(),1e-10)
        results[mkey]={"dir":da,"id":ia,"fdr":float(ratio)}
        print(f"    [{mkey}] dir={da:.1f}% id={ia:.1f}% FDR={ratio:.2f}x")
        del feat
    return results

# Exp2
def exp2_projector_weights():
    print(f"\n  Exp2: Projector Weight Comparison")
    weights={}
    for mkey, path in NLT_PATHS.items():
        if not os.path.exists(path): print(f"    [{mkey}] NOT FOUND"); continue
        weights[mkey]=torch.load(path,map_location='cpu')
    if len(weights)<2: return {}
    results={}
    for key in weights["baseline"]:
        if 'mm_projector' not in key and 'delta' not in key: continue
        wb=weights["baseline"][key].float();wd=weights["delta"].get(key)
        if wd is None:
            print(f"    {key}: only in baseline")
            continue
        wd=wd.float();diff=(wd-wb).norm().item();bn=wb.norm().item()
        results[key]={"diff":diff,"norm":bn,"rel":diff/max(bn,1e-10)*100}
        print(f"    {key}: diff={diff:.4f} norm={bn:.4f} rel={diff/max(bn,1e-10)*100:.2f}%")
    # Delta-only keys
    for key in weights["delta"]:
        if key not in weights["baseline"]:
            print(f"    {key}: DELTA ONLY, norm={weights['delta'][key].float().norm():.4f}")
            results[key+"_delta_only"]={"norm":weights["delta"][key].float().norm().item()}
    return results

# Exp3
def exp3_filtering(task):
    print(f"\n  Exp3: Layer-wise Filtering — {task}")
    metadata=load_metadata(task);id_attr=IDENTITY_ATTRS[task]
    meta=np.load(os.path.join(FEAT_ROOTS["baseline"],"answer_token",TASK_FULL(task),"meta.npy"),allow_pickle=True).item()
    nl=meta["num_layers"]
    results={mk:{"dir":[],"id":[],"fdr":[]} for mk in FEAT_ROOTS}
    for l in range(nl):
        vals={}
        for mkey,feat_root in FEAT_ROOTS.items():
            d=os.path.join(feat_root,"answer_token",TASK_FULL(task))
            feat=np.array(np.load(os.path.join(d,f"features_layer_{l}.npy"),mmap_mode='r')).astype(np.float32)
            qids=np.load(os.path.join(d,"qids.npy"))
            dl,dnc=get_labels(metadata,qids,"direction");il,inc=get_labels(metadata,qids,id_attr)
            da=gpu_probe(feat.copy(),dl,dnc);ia=gpu_probe(feat.copy(),il,inc)
            fd=compute_fdr(feat,dl);fi=compute_fdr(feat,il);r=fd.mean()/max(fi.mean(),1e-10)
            results[mkey]["dir"].append(da);results[mkey]["id"].append(ia);results[mkey]["fdr"].append(float(r))
            vals[mkey]=(da,ia,r)
        bd,bi,bf=vals["baseline"];dd,di,df_=vals["delta"]
        print(f"    L{l:2d} B: dir={bd:5.1f} id={bi:5.1f} FDR={bf:5.2f}x | D: dir={dd:5.1f} id={di:5.1f} FDR={df_:5.2f}x")
    return results

# Exp4
def exp4_aux_head(task):
    print(f"\n  Exp4: Auxiliary Head — {task}")
    nlt=torch.load(NLT_PATHS["delta"],map_location='cpu')
    head_keys=[k for k in nlt if 'delta_direction_head' in k]
    if not head_keys: print("    No head weights"); return {}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw=hb=None
    for k in head_keys:
        if 'head.weight' in k: hw=nlt[k].float().to(device)
        if 'head.bias' in k: hb=nlt[k].float().to(device)
    if hw is None: print("    No head.weight"); return {}
    feat_root=FEAT_ROOTS["delta"]
    ap_dir=os.path.join(feat_root,"after_projector",TASK_FULL(task))
    feat=np.array(np.load(os.path.join(ap_dir,"features.npy"),mmap_mode='r'))
    qids=np.load(os.path.join(feat_root,"vision_token",TASK_FULL(task),"qids.npy"))
    metadata=load_metadata(task);mb={m['id']:m for m in metadata}
    vt_meta=np.load(os.path.join(feat_root,"vision_token",TASK_FULL(task),"meta.npy"),allow_pickle=True).item()
    nf=vt_meta.get("num_frames",8);tpf=vt_meta.get("tokens_per_frame_post",196);hd=vt_meta.get("hidden_dim",3584)
    N=feat.shape[0];feat_4d=feat.reshape(N,nf,tpf,hd)
    fm=feat_4d.mean(axis=2);deltas=fm[:,1:,:]-fm[:,:-1,:]
    dir_to_vec={"up":[0,1],"down":[0,-1],"left":[-1,0],"right":[1,0]}
    correct=0;total=0
    for i,qid in enumerate(qids):
        sid=int(str(qid).split('_')[0]);d=mb[sid]["direction"];gv=torch.tensor(dir_to_vec[d],dtype=torch.float32).to(device)
        for t in range(nf-1):
            dt=torch.from_numpy(deltas[i,t].astype(np.float32)).to(device)
            pred=dt@hw.T;
            if hb is not None: pred=pred+hb
            pn=pred/pred.norm().clamp(min=1e-8);gn=gv/gv.norm().clamp(min=1e-8)
            if (pn*gn).sum()>0.5: correct+=1
            total+=1
    acc=correct/total*100 if total>0 else 0
    print(f"    Accuracy: {acc:.1f}% ({correct}/{total})")
    del feat;torch.cuda.empty_cache()
    return {"acc":acc,"correct":correct,"total":total}

# LoRA weight comparison
def exp5_lora_comparison():
    print(f"\n  Exp5: LoRA Weight Comparison (per layer)")
    from safetensors.torch import load_file
    results={}
    for mkey,path in LORA_PATHS.items():
        if not os.path.exists(path): continue
        w=load_file(path)
        norms={}
        for l in range(28):
            total=0
            for proj in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
                module='self_attn' if proj in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp'
                ka=f'base_model.model.model.layers.{l}.{module}.{proj}.lora_A.weight'
                kb=f'base_model.model.model.layers.{l}.{module}.{proj}.lora_B.weight'
                if ka in w and kb in w:
                    total+=(w[kb].float()@w[ka].float()).norm().item()
            norms[l]=total
        results[mkey]=norms
    if len(results)==2:
        print(f"  Layer   Baseline   Delta    Diff")
        for l in range(28):
            bn=results["baseline"][l];dn=results["delta"][l]
            print(f"    {l:2d}    {bn:7.3f}   {dn:7.3f}   {dn-bn:+.3f}")
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",default="all")
    parser.add_argument("--output_dir",default="analysis/dimension_selection_results")
    args=parser.parse_args()
    tasks=TASKS if args.task=="all" else [args.task]
    os.makedirs(args.output_dir,exist_ok=True)
    all_r={}
    all_r["exp2"]=exp2_projector_weights()
    all_r["exp5"]=exp5_lora_comparison()
    for task in tasks:
        print(f"\n{'#'*60}\n  {task}\n{'#'*60}")
        tr={}
        tr["exp1"]=exp1_projector_temporal(task)
        tr["exp3"]=exp3_filtering(task)
        tr["exp4"]=exp4_aux_head(task)
        all_r[task]=tr
    sp=os.path.join(args.output_dir,"delta_analysis.json")
    def conv(o):
        if isinstance(o,(np.floating,np.integer)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        if isinstance(o,dict):return{k:conv(v) for k,v in o.items()}
        if isinstance(o,list):return[conv(v) for v in o]
        return o
    with open(sp,"w") as f:json.dump(conv(all_r),f,indent=2)
    print(f"\n[SAVED] {sp}")

if __name__=="__main__":main()
