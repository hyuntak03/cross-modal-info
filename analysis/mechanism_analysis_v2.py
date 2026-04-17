"""
Attention Mechanism Analysis v2: Per-head attention from last token to vision tokens.

Exp A: Per-head temporal pattern — 어떤 head가 어떤 frame을 attend하는가
Exp B: Per-head direction discrimination — direction별로 attention이 다른 head 식별
Exp C: Spatial centroid per direction — Up/Down/Left/Right별 spatial attention 차이

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/mechanism_analysis_v2.py \
        --model llava-video-7b_lora_4combo_v2_baseline --task obj_place
"""

import os, sys, json, argparse, math
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, '/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT')
os.environ['HF_HOME'] = '/data/datasets/LLaVA-Video-100K-Subset/'
os.environ['HF_DATASETS_CACHE'] = '/local_datasets/vlm_direction/'

META_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/vlm_direction_testbed/R2R_4way_video"
LORA_PATHS = {
    "llava-video-7b_lora_4combo_v2_baseline": "/data/takhyun03/project/2026/vlm_direction/LLaVA-NeXT/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
}
DIRECTIONS = ["up","down","left","right"]
DIR_TO_IDX = {d:i for i,d in enumerate(DIRECTIONS)}
TASK_FULL = lambda t: f"vlm_direction_testbed_R2R_4way_{t}"

def load_metadata(task):
    with open(os.path.join(META_ROOT,f"{task}_metadata.json")) as f: return json.load(f)

def load_model(model_name):
    from core.model_loader import parse_model_args, load_model_from_args
    if model_name=="llava-video-7b":
        a="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    else:
        lp=LORA_PATHS[model_name]
        a=f"lora_pretrained={lp},pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,mm_spatial_pool_mode=bilinear,max_frames_num=8,device_map=auto,force_sample=True"
    ma=parse_model_args(a)
    tok,model,ip,cl,mn,ct=load_model_from_args(ma)
    model.eval()
    return tok,model,ip,mn,ct

@torch.no_grad()
def run_analysis(model,tokenizer,task,conv_template,image_processor,metadata,limit=200):
    from core.data_pipeline import create_data_loader
    from core.dataset_loader import load_dataset_as_questions
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    import inspect

    questions,_=load_dataset_as_questions(task_name=TASK_FULL(task),limit=limit)
    dl=create_data_loader(questions,"",1,2,tokenizer,image_processor,model.config,
                          TASK_FULL(task),conv_template,video_folder="",video_fps=1,
                          frames_upbound=8,force_sample=True)

    meta_by_id={m['id']:m for m in metadata}
    n_heads=model.config.num_attention_heads
    n_kv=model.config.num_key_value_heads
    hd=model.config.hidden_size//n_heads
    ng=n_heads//n_kv

    _fa=model.model.layers[0].self_attn
    _rp='position_ids' in inspect.signature(_fa.rotary_emb.forward).parameters

    target_layers=[0,5,10,13,15,17,20,25,27]

    # Accumulate per-direction, per-layer, per-head frame attention
    # shape: {dir: {layer: (n_samples, H, T)}}
    dir_frame_attn={d:{l:[] for l in target_layers} for d in DIRECTIONS}
    dir_pos_attn={d:{l:[] for l in target_layers} for d in DIRECTIONS}

    for (input_ids,image_tensor,image_sizes,prompts,mask_tensor,modality),line in tqdm(
        zip(dl,questions),total=len(questions),desc=f"Analyzing {task}"):

        sid=int(str(line['q_id']).split('_')[0])
        direction=meta_by_id[sid]['direction']

        input_ids=input_ids.to('cuda')
        image_tensor=[t.to('cuda') for t in image_tensor]

        (_,position_ids,attention_mask,_,inputs_embeds,_)=\
            model.prepare_inputs_labels_for_multimodal(
                input_ids,None,None,None,None,image_tensor,[modality],image_sizes=image_sizes)

        seq_len=inputs_embeds.shape[1]
        img_dim=seq_len-(input_ids.shape[-1]-1)
        img_pos=torch.where(input_ids[0]==IMAGE_TOKEN_INDEX)[0].tolist()
        img_start=img_pos[0] if img_pos else 0
        img_end=img_start+img_dim

        stride_val=getattr(model.config,"mm_spatial_pool_stride",2)
        pm=getattr(model.config,"mm_spatial_pool_mode","bilinear")
        nps=model.get_vision_tower().num_patches_per_side
        pps=math.ceil(nps/stride_val) if pm=="bilinear" else nps//stride_val
        tpf=pps*pps
        num_frames=max(1,img_dim//tpf)

        pos_def=position_ids if position_ids is not None else torch.arange(seq_len,device=inputs_embeds.device).unsqueeze(0)

        captured={}
        hooks=[]
        for li in target_layers:
            am=model.model.layers[li].self_attn
            cap={}
            def _qh(c):
                def h(m,i,o):c['q']=o.detach()
                return h
            def _kh(c):
                def h(m,i,o):c['k']=o.detach()
                return h
            def _lh(layer_idx,c,attn_mod):
                def h(module,input,output):
                    if 'q' not in c or 'k' not in c:return
                    qr=c.pop('q')[0].float();kr=c.pop('k')[0].float()
                    q=qr.view(seq_len,n_heads,hd).transpose(0,1).unsqueeze(0)
                    k=kr.view(seq_len,n_kv,hd).transpose(0,1).unsqueeze(0)
                    if _rp:cs,sn=attn_mod.rotary_emb(k,pos_def)
                    else:cs,sn=attn_mod.rotary_emb(k,seq_len=seq_len)
                    q,k=apply_rotary_pos_emb(q,k,cs,sn,pos_def)
                    if ng>1:k=k.repeat_interleave(ng,dim=1)
                    ql=q[0,:,-1,:]  #(H,D)
                    kv=k[0,:,img_start:img_end,:]  #(H,V,D)
                    sc=torch.matmul(ql.unsqueeze(1),kv.transpose(-1,-2)).squeeze(1)/(hd**0.5)
                    aw=torch.softmax(sc,dim=-1)  #(H,V)
                    nv=aw.shape[1]
                    pfa=torch.zeros(n_heads,num_frames,device=aw.device)
                    ppa=torch.zeros(n_heads,tpf,device=aw.device)
                    for f in range(num_frames):
                        fs=f*tpf;fe=min(fs+tpf,nv)
                        if fs<nv:
                            pfa[:,f]=aw[:,fs:fe].sum(dim=1)
                            if fe-fs==tpf:ppa+=aw[:,fs:fe]
                    ppa/=max(num_frames,1)
                    captured[layer_idx]={"frame":pfa.cpu().numpy(),"pos":ppa.cpu().numpy()}
                    del q,k
                return h
            hooks.append(am.q_proj.register_forward_hook(_qh(cap)))
            hooks.append(am.k_proj.register_forward_hook(_kh(cap)))
            hooks.append(model.model.layers[li].register_forward_hook(_lh(li,cap,am)))

        outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                      position_ids=position_ids,output_attentions=False,return_dict=True)
        del outputs
        for h in hooks:h.remove()

        for li in target_layers:
            if li in captured:
                dir_frame_attn[direction][li].append(captured[li]["frame"])
                dir_pos_attn[direction][li].append(captured[li]["pos"])

        if len(dir_frame_attn["up"][target_layers[0]])%50==0:
            torch.cuda.empty_cache()

    return dir_frame_attn,dir_pos_attn,target_layers,n_heads,tpf,num_frames

def print_analysis(dir_frame_attn,dir_pos_attn,target_layers,n_heads,tpf,num_frames,task,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    grid=int(np.sqrt(tpf))

    print(f"\n{'='*80}")
    print(f"  Per-Head Temporal Attention — {task}")
    print(f"  (Last Token → Vision Token, softmax over vision tokens only)")
    print(f"{'='*80}")

    analysis={}
    for li in target_layers:
        # Mean across all samples
        all_frame=np.stack([v for d in DIRECTIONS for v in dir_frame_attn[d][li]])  #(N,H,T)
        mean_frame=all_frame.mean(axis=0)  #(H,T)

        # Per direction
        dir_means={}
        for d in DIRECTIONS:
            if dir_frame_attn[d][li]:
                dir_means[d]=np.stack(dir_frame_attn[d][li]).mean(axis=0)  #(H,T)

        # Temporal gradient per head
        tg=mean_frame[:,-1]-mean_frame[:,0]

        # Direction discrimination: variance of per-frame attention across directions
        if len(dir_means)==4:
            ds=np.stack([dir_means[d] for d in DIRECTIONS])  #(4,H,T)
            dv=ds.var(axis=0).sum(axis=1)  #(H,)
        else:
            dv=np.zeros(n_heads)

        analysis[f"L{li}"]={"temporal_gradient":tg.tolist(),"dir_variance":dv.tolist(),
                            "mean_per_frame":mean_frame.tolist(),
                            "dir_per_frame":{d:v.tolist() for d,v in dir_means.items()}}

        print(f"\n  Layer {li}:")
        # Top direction-discriminative heads
        top5=np.argsort(dv)[-5:][::-1]
        print(f"    Direction-discriminative heads (attn pattern differs by direction):")
        for h in top5:
            parts=" ".join(f"{d}:[F0={dir_means[d][h,0]:.3f},F7={dir_means[d][h,-1]:.3f}]" for d in DIRECTIONS)
            print(f"      Head {h:2d} (var={dv[h]:.5f}): {parts}")

    # Spatial centroid analysis at layer 17
    print(f"\n{'='*80}")
    print(f"  Spatial Attention Centroid per Direction — Layer 17")
    print(f"  (Weighted average (row,col) on {grid}×{grid} grid)")
    print(f"{'='*80}")

    li=17
    if li in target_layers:
        y_coords=np.arange(grid).reshape(1,grid,1)
        x_coords=np.arange(grid).reshape(1,1,grid)
        print(f"\n  {'Head':>6}",end="")
        for d in DIRECTIONS:print(f"  {d:>14}",end="")
        print(f"  {'shift':>10}")

        centroids_by_dir={}
        for d in DIRECTIONS:
            if dir_pos_attn[d][li]:
                mean_pos=np.stack(dir_pos_attn[d][li]).mean(axis=0)  #(H,196)
                g=mean_pos.reshape(n_heads,grid,grid)
                cy=(g*y_coords).sum(axis=(1,2))/g.sum(axis=(1,2)).clip(1e-10)
                cx=(g*x_coords).sum(axis=(1,2))/g.sum(axis=(1,2)).clip(1e-10)
                centroids_by_dir[d]=np.stack([cy,cx],axis=1)  #(H,2)

        if len(centroids_by_dir)==4:
            for h in range(n_heads):
                print(f"  {h:>6}",end="")
                positions=[]
                for d in DIRECTIONS:
                    cy,cx=centroids_by_dir[d][h]
                    print(f"  ({cy:.1f},{cx:.1f})",end="")
                    positions.append((cy,cx))
                # Shift: up-down y差, left-right x差
                ud_shift=abs(positions[0][0]-positions[1][0])  # up vs down: y should differ
                lr_shift=abs(positions[2][1]-positions[3][1])  # left vs right: x should differ
                total_shift=ud_shift+lr_shift
                marker=" ★" if total_shift>1.0 else ""
                print(f"  {total_shift:>8.2f}{marker}")

            analysis[f"spatial_centroid_L{li}"]={d:centroids_by_dir[d].tolist() for d in DIRECTIONS}

    sp=os.path.join(output_dir,f"mechanism_v2_{task}.json")
    def conv(o):
        if isinstance(o,(np.floating,np.integer)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        if isinstance(o,dict):return{k:conv(v) for k,v in o.items()}
        if isinstance(o,list):return[conv(v) for v in o]
        return o
    with open(sp,"w") as f:json.dump(conv(analysis),f,indent=2)
    print(f"\n[SAVED] {sp}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="llava-video-7b_lora_4combo_v2_baseline")
    parser.add_argument("--task",default="obj_place")
    parser.add_argument("--limit",type=int,default=200)
    parser.add_argument("--output_dir",default="analysis/mechanism_results")
    args=parser.parse_args()

    tasks=[args.task] if args.task!="all" else ["shape_color","obj_place"]
    os.makedirs(args.output_dir,exist_ok=True)

    print("Loading model...")
    tok,model,ip,mn,ct=load_model(args.model)

    for task in tasks:
        metadata=load_metadata(task)
        dfa,dpa,tl,nh,tpf,nf=run_analysis(model,tok,task,ct,ip,metadata,args.limit)
        print_analysis(dfa,dpa,tl,nh,tpf,nf,task,args.output_dir)

if __name__=="__main__":main()
