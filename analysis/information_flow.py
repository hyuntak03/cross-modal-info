import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# InformationFlow.py
import re
import copy
import pdb

from types import MethodType
from core.methods import (
    trace_with_attn_block_llava, set_block_attn_hooks_llava, remove_wrapper_llava,
    _precomputed_mask_cache, _precomputed_index_cache,
)

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()


from PIL import Image, ImageDraw


import argparse
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init, process_video_with_decord
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerateOutput
import requests

from core.utils import prepare_image_patch_bbx,create_mask_with_bbox,show_original_image,show_transferred_maskandimage, generate_plot

from core.data_pipeline import (
    CustomDataset, collate_fn, create_data_loader,
    find_token_range, generate_llava, blockdesc2range, blockdesc2range_patches
)

from core.model_loader import parse_model_args, load_model_from_args, load_model_legacy


#! Attention Knock out 없이 그냥 모델 돌리기
def _extract_mcq_letter(text: str) -> str:
    """MCQ 응답에서 옵션 letter 추출. '(a)', '(A)', 'A.', 'a' 등 다양한 포맷 대응."""
    text = text.strip()
    m = re.match(r'^\(?([a-eA-E])\)?', text)
    if m:
        return m.group(1).upper()
    return text[0].upper() if text else ""


def run_original(model, inps, tokenizer, model_name, answer, mask_tensor=None, args=None):
    with torch.inference_mode():
        model.old_generate= model.generate
        model.generate =  MethodType(generate_llava, model)
        if args.certain_part_image:
            patched_mask, objects_indices, pad_indices,original_patch_indices,hd_patch_indice, objects_indices_in_hd, inputs_embeds_shape, output_details = model.generate(mask=mask_tensor, args=args, **inps)
        else:
            inputs_embeds_shape, output_details = model.generate(args=args,**inps)
        model.generate = model.old_generate

    answer_token_id = output_details['sequences']
    generated_first_id = answer_token_id[:, 0]

    #! 정답의 첫 토큰 ID
    answer_cap = answer.strip().upper()
    gt_token_ids = tokenizer.encode(answer_cap, add_special_tokens=False)
    gt_first_token_id = gt_token_ids[0]
    gt_first_token_id_tensor = torch.tensor([gt_first_token_id], device=generated_first_id.device)
    logits_first_answer_token = output_details['scores'][0]
    probs = torch.softmax(logits_first_answer_token, dim=-1)[0]
    #! GT 토큰과 예측 토큰 각각의 base score
    gt_base_score = probs[gt_first_token_id_tensor].item()
    predicted_base_score = probs[generated_first_id].item()

    # 전체 디코딩 후 MCQ letter 추출
    raw_predicted = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip()
    predicted_answer = _extract_mcq_letter(raw_predicted)

    is_correct_bool = predicted_answer == answer_cap

    if args.certain_part_image:
        return gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id_tensor, generated_first_id, inputs_embeds_shape, is_correct_bool, objects_indices, pad_indices, original_patch_indices, hd_patch_indice, objects_indices_in_hd, patched_mask
    else:
        return gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id_tensor, generated_first_id, inputs_embeds_shape, is_correct_bool



# Information flow analysis
def InforFlowAna(args):

    cache_dir = os.environ.get("HF_HOME", None)

    # Model: model_args 스타일 또는 기존 --model-path 스타일
    if args.model_args:
        model_args_dict = parse_model_args(args.model_args)
        tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
        args.conv_mode = conv_template
    else:
        tokenizer, model, image_processor, context_len, model_name, _ = load_model_legacy(args.model_path, args.model_base, args.conv_mode)

    model.prepare_image_patch_bbx=MethodType(prepare_image_patch_bbx, model)
    model.eval()

    if args.noHD_noPad:
        model.config.image_aspect_ratio="pad"  #HD: anyres
        model.config.mm_patch_merge_type='spatial'  # pad: 'spatial_unpad

    #dataset
    #predict correct and filter
    task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
    df = pd.read_csv(args.refined_dataset, dtype={"question_id":str}).fillna('')
    dataset_dict = df.set_index('question_id').T.to_dict('dict')
    questions = [ {**detail, "q_id":qu_id} for qu_id, detail in dataset_dict.items()]

    # data_loader = create_data_loader(questions, args.image_folder,  args.batch_size, args.num_workers, tokenizer,  image_processor, model.config, task_name, args.conv_mode)
    data_loader = create_data_loader(questions, args.image_folder, args.batch_size, args.num_workers,
                                  tokenizer, image_processor, model.config, task_name, args.conv_mode,
                                  video_folder=args.video_folder, video_fps=args.video_fps,
                                  frames_upbound=args.frames_upbound, force_sample=args.force_sample)



    # Run attention knockouts
    results = []
    index=0

    #! 모두 정답 못 맞추더라도 코드 뻑나는거 방지 (for 돌기 전에 미리 초기화)
    block_descs = []
    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality), line in tqdm(zip(data_loader, questions), total=len(questions)):

        # 새 샘플마다 mask 캐시 클리어
        _precomputed_mask_cache.clear()
        _precomputed_index_cache.clear()

        question_id = line["q_id"]
        # img_id= str(line["img_id"]) + ".png"

        if "video" in line and line["video"] != "":
            img_id = str(line["video"])
        else:
            img_id_str = str(line["img_id"])
            if os.path.splitext(img_id_str)[1]:
                img_id = img_id_str
            else:
                img_id = img_id_str
        #! last token은 ":"임 (마지막이 Assitant: 이기 때문에)
        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]
        # mask_tensor = [ma.to(device='cuda') for ma in mask_tensor]
        #! mask_tensor Bounding Box 없이도 돌아가게 만들기
        if mask_tensor is not None:
            mask_tensor = [ma.to(device='cuda') for ma in mask_tensor]

        #! LLaVA v1.5, v1.6 일 경우 modality를 image로 고정
        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        inps={
            "inputs":input_ids,
            "images":image_tensor,
            "image_sizes":original_image_sizes,
            "do_sample":True if args.temperature > 0 else False,
            "modalities": [effective_modality], #! video 인지 image 인지 구분
            "temperature":args.temperature,
            "top_p":args.top_p,
            "num_beams":args.num_beams,
            "max_new_tokens" : args.max_new_tokens,
            "use_cache" : True,
            "return_dict_in_generate" : True,
            "output_scores" : True,
            "pad_token_id": tokenizer.eos_token_id

        }

        question = dataset_dict[question_id]["question"]
        answer = dataset_dict[question_id]["answer"]

        #! run_original: GT 토큰과 예측 토큰 각각의 base score 반환
        if args.certain_part_image:
            gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id, predicted_first_token_id, inputs_embeds_shape, is_correct_bool, central_object_patch_indices, pad_patch_indices, original_patch_indices, hd_patch_indice, objects_indices_in_hd, patched_mask = run_original(model, inps, tokenizer, model_name, answer, mask_tensor, args=args)
        else:
            gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id, predicted_first_token_id, inputs_embeds_shape, is_correct_bool = run_original(model, inps, tokenizer, model_name, answer, args=args)


        if is_correct_bool == False:
            is_correct=False
        else:
            is_correct=True
            index += 1
            print("Finish samples:", index)

        #get range
        #! 콤마로 구분된 여러 block_description 지원
        #! e.g. "Image->Question,Image->Last"
        block_desc_pairs = [bd.strip() for bd in args.block_description.split(",")]

        all_temp2 = []
        for bd_pair in block_desc_pairs:
            bd_split = bd_pair.split("->")
            if args.certain_part_image:
                r1 = blockdesc2range_patches(bd_split[0], input_ids, inputs_embeds_shape, central_object_patch_indices, pad_patch_indices, hd_patch_indice, objects_indices_in_hd, original_patch_indices)
            else:
                r1 = blockdesc2range(bd_split[0], dataset_dict, question_id, input_ids, inputs_embeds_shape, tokenizer, model_name, args=args)
            r2 = blockdesc2range(bd_split[1], dataset_dict, question_id, input_ids, inputs_embeds_shape, tokenizer, model_name, args=args)
            all_temp2.extend([(stok1, stok0) for stok0 in r1 for stok1 in r2])

        block_descs = [(all_temp2, args.block_description)]

        #! decode 단계에서 ->Last pair만 필터링하기 위해 last_token_idx 계산
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        ntoks = input_ids.shape[1] + image_dim - 1
        last_token_idx = ntoks - 1

        #! inference_only 모드: knockout 없이 정확도만 측정
        if args.inference_only:
            re_result = {
                "question_id": question_id,
                "image": img_id,
                "goden answer": answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "question": question,
                "gt_base_score": gt_base_score,
                "predicted_base_score": predicted_base_score,
            }
            results.append(re_result)
            continue

        for temp2, block_desc in block_descs:

            if args.block_all_layers:
                block_config = {
                    l: temp2
                    for l in range(model.config.num_hidden_layers)
                }
                inps["max_new_tokens"] = 1

                #! full probs 반환 → GT/predicted 둘 다 indexing
                new_probs, knocked_predicted_answer = trace_with_attn_block_llava(
                    model, inps, block_config, block_desc, model_name,
                    tokenizer=tokenizer, last_token_idx=last_token_idx,
                    use_cached_embeds=True,
                )

                new_score_gt = new_probs[gt_first_token_id].cpu().item()
                new_score_predicted = new_probs[predicted_first_token_id].cpu().item()

                #! GT answer tracing (항상 저장)
                re_gt = {
                    "question_id": question_id,
                    "image": img_id,
                    "goden answer": answer,
                    "origin_predicted_answer": predicted_answer,
                    "knocked_predicted_answer": knocked_predicted_answer,
                    "is_correct": is_correct,
                    "question": question,
                    "block_desc": block_desc,
                    "layer": "all",
                    "trace_target": "gt_answer",
                    "base_score_first": gt_base_score,
                    "new_score_first": new_score_gt,
                    "relative diff first": (new_score_gt - gt_base_score) * 100.0 / gt_base_score if gt_base_score != 0 else 0.0,
                }
                results.append(re_gt)

                #! predicted answer tracing (오답일 때만 별도 저장)
                if not is_correct:
                    re_pred = {
                        "question_id": question_id,
                        "image": img_id,
                        "goden answer": answer,
                        "origin_predicted_answer": predicted_answer,
                        "knocked_predicted_answer": knocked_predicted_answer,
                        "is_correct": is_correct,
                        "question": question,
                        "block_desc": block_desc,
                        "layer": "all",
                        "trace_target": "predicted_answer",
                        "base_score_first": predicted_base_score,
                        "new_score_first": new_score_predicted,
                        "relative diff first": (new_score_predicted - predicted_base_score) * 100.0 / predicted_base_score if predicted_base_score != 0 else 0.0,
                    }
                    results.append(re_pred)
            else:
                #! 기존: layer별 sliding window knockout
                for layer in range(model.config.num_hidden_layers):
                    layerlist = [
                        l for l in range(
                            max(0, layer - args.window // 2), min(model.config.num_hidden_layers, layer - (-args.window // 2))
                        )
                    ]
                    block_config = {
                        l: temp2
                        for l in layerlist
                    }

                    inps["max_new_tokens"] = 1
                    new_probs, knocked_predicted_answer = trace_with_attn_block_llava(
                        model, inps, block_config, block_desc, model_name,
                        tokenizer=tokenizer, last_token_idx=last_token_idx,
                        use_cached_embeds=True,
                    )

                    new_score_gt = new_probs[gt_first_token_id].cpu().item()
                    new_score_predicted = new_probs[predicted_first_token_id].cpu().item()

                    #! GT answer tracing
                    re_gt = {
                        "question_id": question_id,
                        "image": img_id,
                        "goden answer": answer,
                        "origin_predicted_answer": predicted_answer,
                        "knocked_predicted_answer": knocked_predicted_answer,
                        "is_correct": is_correct,
                        "question": question,
                        "block_desc": block_desc,
                        "layer": layer,
                        "trace_target": "gt_answer",
                        "base_score_first": gt_base_score,
                        "new_score_first": new_score_gt,
                        "relative diff first": (new_score_gt - gt_base_score) * 100.0 / gt_base_score if gt_base_score != 0 else 0.0,
                    }
                    results.append(re_gt)

                    #! predicted answer tracing (오답일 때만)
                    if not is_correct:
                        re_pred = {
                            "question_id": question_id,
                            "image": img_id,
                            "goden answer": answer,
                            "origin_predicted_answer": predicted_answer,
                            "knocked_predicted_answer": knocked_predicted_answer,
                            "is_correct": is_correct,
                            "question": question,
                            "block_desc": block_desc,
                            "layer": layer,
                            "trace_target": "predicted_answer",
                            "base_score_first": predicted_base_score,
                            "new_score_first": new_score_predicted,
                            "relative diff first": (new_score_predicted - predicted_base_score) * 100.0 / predicted_base_score if predicted_base_score != 0 else 0.0,
                        }
                        results.append(re_pred)


    if args.inference_only:
        tmp = pd.DataFrame.from_records(results)
        model_name_safe = model_name.replace('-', '_').replace('.', '_')
        dataset_name = args.refined_dataset.split("/")[-1].split(".csv")[0]
        os.makedirs(f"output/inference_only/{model_name_safe}", exist_ok=True)
        out_path = f"output/inference_only/{model_name_safe}/{dataset_name}_inference.csv"
        tmp.to_csv(out_path, index=False)

        acc = tmp["is_correct"].sum() / len(tmp) * 100
        print(f"\n{'='*50}")
        print(f"  Accuracy: {acc:.2f}% ({tmp['is_correct'].sum()}/{len(tmp)})")
        print(f"  Saved: {out_path}")
        print(f"{'='*50}")
        return

    save_name = "_".join([des[1].replace(" ", "_").replace("->", "___") for des in block_descs])

    if args.noHD_noPad:
        save_name=save_name+"_noHD_noPad"
    if args.block_all_layers:
        save_name=save_name+"_block_all_layers"

    tmp = pd.DataFrame.from_records(results)
    model_name = model_name.replace('-', '_').replace('.', '_')
    os.makedirs(f"output/information_flow/{model_name}/{task_name}/val/{save_name}", exist_ok=True)

    tmp.to_csv(f'output/information_flow/{model_name}/{task_name}/val/{save_name}/{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}.csv', index=False)

    base_path = f'output/information_flow/{model_name}/{task_name}/val/{save_name}/{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}'

    # 전체
    # generate_plot(tmp, f'{base_path}_first_all.pdf', x="layer", y="relative diff first", hue="block_desc", layers=model.config.num_hidden_layers)

    if args.block_all_layers:
        #! block_all_layers 모드: answer class별 bar plot 생성
        print(f"[INFO] block_all_layers mode: generating summary bar plots.", flush=True)
        print(f"[INFO] Correct samples: {len(tmp[tmp['is_correct']==True]['question_id'].unique())}, "
              f"Incorrect samples: {len(tmp[tmp['is_correct']==False]['question_id'].unique())}", flush=True)

        #! bar plot은 trace_target별로 분리해서 그려야 함 (gt_answer / predicted_answer 혼합 방지)
        tmp_gt = tmp[tmp["trace_target"] == "gt_answer"]
        tmp_pred = tmp[tmp["trace_target"] == "predicted_answer"]

        #! 전체 (gt answer 기준)
        generate_plot(tmp_gt, f'{base_path}_first_all.pdf',
                      y="relative diff first", block_all_layers=True, block_description=args.block_description)

        #! 정답만 (gt answer만 존재)
        tmp_correct = tmp_gt[tmp_gt["is_correct"] == True]
        if len(tmp_correct) > 0:
            generate_plot(tmp_correct, f'{base_path}_first_correct.pdf',
                          y="relative diff first", block_all_layers=True, block_description=args.block_description)

        #! 오답만 — gt answer tracing
        tmp_incorrect_gt = tmp_gt[tmp_gt["is_correct"] == False]
        if len(tmp_incorrect_gt) > 0:
            generate_plot(tmp_incorrect_gt, f'{base_path}_first_incorrect_gt.pdf',
                          y="relative diff first", block_all_layers=True, block_description=args.block_description)

        #! 오답만 — predicted answer tracing
        if len(tmp_pred) > 0:
            generate_plot(tmp_pred, f'{base_path}_first_incorrect_predicted.pdf',
                          y="relative diff first", block_all_layers=True, block_description=args.block_description)
    else:
        #! 기존: layer별 plot 생성
        #! 정답만 (knockout 전 기준)
        tmp_correct = tmp[tmp["is_correct"] == True]
        if len(tmp_correct) > 0:
            generate_plot(tmp_correct, f'{base_path}_first_correct.pdf', x="layer", y="relative diff first", hue="block_desc", layers=model.config.num_hidden_layers)

        #! 오답만 (knockout 전 기준)
        tmp_incorrect = tmp[tmp["is_correct"] == False]
        if len(tmp_incorrect) > 0:
            generate_plot(tmp_incorrect, f'{base_path}_first_incorrect.pdf', x="layer", y="relative diff first", hue="block_desc", layers=model.config.num_hidden_layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", type=str, default=None,
                        help='lmms_eval style model args. '
                             'e.g., "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto"')
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image-folder", type=str, default="")


    parser.add_argument("--window", type=int, default=9)
    parser.add_argument('--refined_dataset', default="", type=str, help="refined dataset")
    parser.add_argument('--block_description', default=None, type=str, help="block_description")
    parser.add_argument('--certain_part_image', default=False, action="store_true")
    parser.add_argument('--noHD_noPad', default=False, action="store_true", help="noHD_noPad")

    #! video 관련 인자 추가
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)

    #! 모든 layer에서 Attention Knock Out 적용
    parser.add_argument('--block_all_layers', default=False, action="store_true", help="Block attention across all layers at once")

    #! Instruction에 Assistant도 포함시킬지 argument로 받음
    parser.add_argument('--block_ASSIST', default=False, action="store_true", help="Also block ASSISTANT tokens in Instruction range")

    #! Inference Only
    parser.add_argument('--inference_only', default=False, action="store_true", help="Run inference only without knockout, just measure accuracy")

    args = parser.parse_args()

    #! 콤마로 구분된 여러 block_description에서 Image patch 관련 여부 감지
    for bd_pair in args.block_description.split(","):
        bd_split = bd_pair.strip().split("->")
        if "Image " in bd_split[0]:
            args.certain_part_image = True
            break

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    InforFlowAna(args)
