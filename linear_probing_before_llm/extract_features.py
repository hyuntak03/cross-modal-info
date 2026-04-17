"""
Vision Encoder / Projector 출력에 대한 Feature 추출 스크립트.

LLM에 들어가기 전 단계의 feature를 추출하여 linear probing:
  - Stage 1 (pre_projector): Vision encoder 출력 (SigLIP 등)
  - Stage 2 (post_projector): Projector 통과 후, spatial pooling 후 (LLM 입력 직전)

LLM hidden states probing (linear_probing_per_layer/)과 비교하여
정보가 어느 단계에서 사라지는지 특정 가능.

Usage:
    # Single GPU
    python linear_probing_before_llm/extract_features.py \
        --model_args "pretrained=...,device_map=auto" \
        --task identity_testbed_realobj_realbg \
        --output_dir output/before_llm_features/model/task \
        --frames_upbound 2 --force_sample

    # Multi GPU (4 GPUs)
    python linear_probing_before_llm/extract_features.py \
        --model_args "pretrained=...,device_map=auto" \
        --task identity_testbed_realobj_realbg \
        --output_dir output/before_llm_features/model/task \
        --frames_upbound 2 --force_sample --num_gpus 4
"""

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import ast
import importlib.util
import math
import string
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

torch.set_grad_enabled(False)

# dataset loader 직접 import
def _import_module_direct(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dataset_loader = _import_module_direct(
    "core.dataset_loader", os.path.join(_PROJECT_ROOT, "core", "dataset_loader.py")
)
load_dataset_as_questions = _dataset_loader.load_dataset_as_questions


# ============================================================
#  공통 유틸
# ============================================================

def build_label_set(questions):
    unique_answers = set()
    for q in questions:
        ans = str(q["answer"]).strip()
        if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
            candidates_raw = q.get("candidates", [])
            if isinstance(candidates_raw, str):
                candidates_raw = ast.literal_eval(candidates_raw)
            idx = ord(ans.upper()) - ord('A')
            if idx < len(candidates_raw):
                ans = str(candidates_raw[idx]).strip()
        unique_answers.add(ans)
    label_list = sorted(unique_answers)
    answer_to_idx = {a: i for i, a in enumerate(label_list)}
    return label_list, answer_to_idx


def resolve_answer(line):
    answer = str(line["answer"]).strip()
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        candidates_raw = line.get("candidates", [])
        if isinstance(candidates_raw, str):
            candidates_raw = ast.literal_eval(candidates_raw)
        idx = ord(answer.upper()) - ord('A')
        if idx < len(candidates_raw):
            answer = str(candidates_raw[idx]).strip()
    return answer


def _load_video_frames(video_path, image_processor, frames_upbound, force_sample):
    """비디오를 로드하고 image_processor로 전처리하여 tensor 반환."""
    from decord import VideoReader, cpu as decord_cpu
    vr = VideoReader(video_path, ctx=decord_cpu(0), num_threads=1)
    total = len(vr)
    if force_sample or total > frames_upbound:
        indices = np.linspace(0, total - 1, frames_upbound, dtype=int).tolist()
    else:
        indices = list(range(total))
    video = vr.get_batch(indices).asnumpy()
    video = np.stack(video)
    tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
    return tensor, len(indices)


def _prepare_valid_samples(questions, answer_to_idx, video_folder):
    """유효한 샘플 목록 준비."""
    valid_samples = []
    for line in questions:
        answer = resolve_answer(line)
        if answer not in answer_to_idx:
            continue
        video_rel = line["video"]
        if video_folder and not os.path.isabs(video_rel):
            video_path = os.path.join(video_folder, video_rel)
        else:
            video_path = video_rel
        if not os.path.exists(video_path):
            hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface"))
            video_path = os.path.join(hf_cache, video_rel)
        valid_samples.append({
            "q_id": line["q_id"],
            "label_idx": answer_to_idx[answer],
            "video_path": video_path,
        })
    return valid_samples


def _load_model_on_device(model_args_str, device):
    """지정된 device에 vision encoder + projector만 로드."""
    _model_loader = _import_module_direct(
        "core.model_loader", os.path.join(_PROJECT_ROOT, "core", "model_loader.py")
    )
    parse_model_args_llava = _model_loader.parse_model_args
    load_model_from_args = _model_loader.load_model_from_args

    model_args_dict = parse_model_args_llava(model_args_str)
    # device_map을 해당 device로 고정
    model_args_dict["device_map"] = str(device)
    tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
    model.eval()
    model.tie_weights()
    return model, image_processor, model_name


def _get_model_info(model):
    """모델 구조 정보 추출."""
    vision_tower = model.get_vision_tower()
    num_patches_per_side = vision_tower.num_patches_per_side
    vision_hidden_dim = vision_tower.config.hidden_size
    llm_hidden_dim = model.config.hidden_size
    stride = getattr(model.config, "mm_spatial_pool_stride", 2)
    pool_mode = getattr(model.config, "mm_spatial_pool_mode", "bilinear")

    if pool_mode == "bilinear":
        pooled_per_side = math.ceil(num_patches_per_side / stride)
    else:
        pooled_per_side = num_patches_per_side // stride

    return {
        "num_patches_per_side": num_patches_per_side,
        "vision_hidden_dim": vision_hidden_dim,
        "llm_hidden_dim": llm_hidden_dim,
        "stride": stride,
        "tokens_per_frame_pre": num_patches_per_side * num_patches_per_side,
        "tokens_per_frame_post": pooled_per_side * pooled_per_side,
    }


def _process_samples(model, image_processor, samples, args, device, info, rank=0, num_gpus=1):
    """샘플 리스트에 대해 feature 추출."""
    stride = info["stride"]
    vision_hidden_dim = info["vision_hidden_dim"]
    llm_hidden_dim = info["llm_hidden_dim"]

    pre_projector_features = []
    post_projector_features = []
    all_labels = []
    all_qids = []
    num_frames = args.frames_upbound

    desc = f"[GPU {rank}] Extracting" if num_gpus > 1 else "Extracting before-LLM features"
    for batch_start in tqdm(range(0, len(samples), args.batch_size),
                            desc=desc,
                            total=math.ceil(len(samples) / args.batch_size),
                            position=rank, leave=True):
        batch = samples[batch_start:batch_start + args.batch_size]

        batch_tensors = []
        for sample in batch:
            tensor, nf = _load_video_frames(
                sample["video_path"], image_processor,
                args.frames_upbound, args.force_sample,
            )
            batch_tensors.append(tensor)
            num_frames = nf

        all_frames = torch.cat(batch_tensors, dim=0).to(device=device, dtype=torch.float16)

        with torch.inference_mode():
            vision_out = model.get_model().get_vision_tower()(all_frames)
            projected = model.get_model().mm_projector(vision_out)
            if stride > 1:
                pooled = model.get_2dPool(projected, stride)
            else:
                pooled = projected

        vision_out_split = vision_out.view(len(batch), num_frames, -1, vision_hidden_dim)
        pooled_split = pooled.view(len(batch), num_frames, -1, llm_hidden_dim)

        for i, sample in enumerate(batch):
            pre_feat = vision_out_split[i].reshape(-1).cpu().to(torch.float16)
            post_feat = pooled_split[i].reshape(-1).cpu().to(torch.float16)
            pre_projector_features.append(pre_feat)
            post_projector_features.append(post_feat)
            all_labels.append(sample["label_idx"])
            all_qids.append(sample["q_id"])

        del all_frames, vision_out, projected, pooled
        torch.cuda.empty_cache()

    return pre_projector_features, post_projector_features, all_labels, all_qids, num_frames


# ============================================================
#  Multi-GPU worker
# ============================================================

def _gpu_worker(rank, num_gpus, args, valid_samples, label_list, answer_to_idx, output_dir):
    """각 GPU에서 실행되는 worker. 결과를 최종 디렉토리에 직접 저장."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 데이터 분할
    chunk_size = math.ceil(len(valid_samples) / num_gpus)
    start = rank * chunk_size
    end = min(start + chunk_size, len(valid_samples))
    my_samples = valid_samples[start:end]

    if len(my_samples) == 0:
        return

    print(f"[GPU {rank}] Loading model on {device}, processing {len(my_samples)} samples")

    model, image_processor, model_name = _load_model_on_device(args.model_args, device)
    info = _get_model_info(model)

    pre_feats, post_feats, labels, qids, num_frames = _process_samples(
        model, image_processor, my_samples, args, device, info, rank=rank, num_gpus=num_gpus
    )

    # 최종 디렉토리에 rank별 파일로 직접 저장 (병렬 I/O)
    os.makedirs(output_dir, exist_ok=True)

    def _save_pre():
        np.save(os.path.join(output_dir, f"features_pre_projector_rank{rank}.npy"),
                torch.stack(pre_feats, dim=0).numpy())

    def _save_post():
        np.save(os.path.join(output_dir, f"features_post_projector_rank{rank}.npy"),
                torch.stack(post_feats, dim=0).numpy())

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_save_pre)
        f2 = ex.submit(_save_post)
        # labels, qids, meta는 작으니까 그냥 저장
        np.save(os.path.join(output_dir, f"labels_rank{rank}.npy"), np.array(labels))
        np.save(os.path.join(output_dir, f"qids_rank{rank}.npy"), np.array(qids))
        np.save(os.path.join(output_dir, f"meta_rank{rank}.npy"), {
            "num_frames": num_frames, "model_name": model_name, "info": info,
            "num_classes": len(label_list), "label_list": label_list, "task": args.task,
        })
        f1.result()
        f2.result()

    print(f"[GPU {rank}] Done. Saved {len(labels)} samples to {output_dir}")

    del model, pre_feats, post_feats
    torch.cuda.empty_cache()


# ============================================================
#  메인 추출 함수
# ============================================================

def extract_features_llava(args):
    cache_dir = os.environ.get("HF_HOME", None)

    questions, dataset_dict = load_dataset_as_questions(
        task_name=args.task,
        video_folder=args.video_folder,
        image_folder=args.image_folder,
        hf_cache_dir=cache_dir,
        limit=args.limit,
    )

    label_list, answer_to_idx = build_label_set(questions)
    num_classes = len(label_list)
    print(f"[INFO] Classes ({num_classes}): {label_list}")
    print(f"[INFO] Total samples: {len(questions)}, Batch size: {args.batch_size}, GPUs: {args.num_gpus}")

    valid_samples = _prepare_valid_samples(questions, answer_to_idx, args.video_folder)
    print(f"[INFO] Valid samples: {len(valid_samples)}")

    if args.num_gpus <= 1:
        # Single GPU
        device = torch.device("cuda:0")
        model, image_processor, model_name = _load_model_on_device(args.model_args, device)
        info = _get_model_info(model)

        print(f"[INFO] Vision encoder: {info['tokens_per_frame_pre']} tokens/frame, dim={info['vision_hidden_dim']}")
        print(f"[INFO] After projector + pool: {info['tokens_per_frame_post']} tokens/frame, dim={info['llm_hidden_dim']}")

        pre_feats, post_feats, labels, qids, num_frames = _process_samples(
            model, image_processor, valid_samples, args, device, info
        )
        del model
        torch.cuda.empty_cache()
    else:
        # Multi GPU — 각 worker가 최종 디렉토리에 rank별 파일로 직접 저장
        os.makedirs(args.output_dir, exist_ok=True)

        mp.spawn(
            _gpu_worker,
            args=(args.num_gpus, args, valid_samples, label_list, answer_to_idx, args.output_dir),
            nprocs=args.num_gpus,
            join=True,
        )

        # meta에서 정보 읽기 + 통합 meta 저장
        total_samples = 0
        for rank in range(args.num_gpus):
            meta_path = os.path.join(args.output_dir, f"meta_rank{rank}.npy")
            if not os.path.exists(meta_path):
                continue
            meta_r = np.load(meta_path, allow_pickle=True).item()
            lab_arr = np.load(os.path.join(args.output_dir, f"labels_rank{rank}.npy"))
            total_samples += len(lab_arr)
            num_frames = meta_r["num_frames"]
            model_name = meta_r["model_name"]
            info = meta_r["info"]

        meta = {
            "num_frames": num_frames,
            "tokens_per_frame_pre": info["tokens_per_frame_pre"],
            "tokens_per_frame_post": info["tokens_per_frame_post"],
            "vision_hidden_dim": info["vision_hidden_dim"],
            "llm_hidden_dim": info["llm_hidden_dim"],
            "num_samples": total_samples,
            "num_classes": num_classes,
            "label_list": label_list,
            "model_name": model_name,
            "task": args.task,
            "num_ranks": args.num_gpus,
        }
        np.save(os.path.join(args.output_dir, "meta.npy"), meta)

        print(f"[DONE] Saved {total_samples} samples across {args.num_gpus} ranks to {args.output_dir}")
        return None  # 이미 저장 완료

    return {
        "pre_projector": pre_feats,
        "post_projector": post_feats,
        "labels": labels,
        "qids": qids,
        "num_classes": num_classes,
        "label_list": label_list,
        "model_name": model_name,
        "task": args.task,
        "meta": {
            "num_frames": num_frames,
            "tokens_per_frame_pre": info["tokens_per_frame_pre"],
            "tokens_per_frame_post": info["tokens_per_frame_post"],
            "vision_hidden_dim": info["vision_hidden_dim"],
            "llm_hidden_dim": info["llm_hidden_dim"],
        },
    }


# ============================================================
#  저장 (single GPU용)
# ============================================================

def save_results(output_dir, results):
    os.makedirs(output_dir, exist_ok=True)

    labels_array = np.array(results["labels"], dtype=np.int64)
    np.save(os.path.join(output_dir, "labels.npy"), labels_array)
    np.save(os.path.join(output_dir, "qids.npy"), np.array(results["qids"]))

    def _save(name, feat_list):
        features = torch.stack(feat_list, dim=0).numpy()
        np.save(os.path.join(output_dir, f"features_{name}.npy"), features)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_save, "pre_projector", results["pre_projector"])
        f2 = executor.submit(_save, "post_projector", results["post_projector"])
        f1.result()
        f2.result()

    meta = results["meta"].copy()
    meta.update({
        "num_samples": len(results["labels"]),
        "num_classes": results["num_classes"],
        "label_list": results["label_list"],
        "model_name": results["model_name"],
        "task": results["task"],
    })
    np.save(os.path.join(output_dir, "meta.npy"), meta)

    m = results["meta"]
    print(f"[DONE] Saved {len(results['labels'])} samples to {output_dir}")
    print(f"  pre_projector:  {results['pre_projector'][0].shape[0]} dim "
          f"[{m['num_frames']}f x {m['tokens_per_frame_pre']}tok x {m['vision_hidden_dim']}d]")
    print(f"  post_projector: {results['post_projector'][0].shape[0]} dim "
          f"[{m['num_frames']}f x {m['tokens_per_frame_post']}tok x {m['llm_hidden_dim']}d]")
    print(f"  Labels distribution: {np.bincount(labels_array, minlength=results['num_classes'])}")


# ============================================================
#  메인
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vision encoder / projector features for linear probing")
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/before_llm_features")
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    if args.num_gpus > 1:
        mp.set_start_method("spawn", force=True)

    results = extract_features_llava(args)
    if results is not None:  # single GPU — multi GPU는 내부에서 직접 저장
        save_results(args.output_dir, results)
