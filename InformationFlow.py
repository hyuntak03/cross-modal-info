# InformationFlow.py
import copy
import pdb

from methods import *

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()


from PIL import Image, ImageDraw


import argparse
import os
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, process_video_with_decord
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerateOutput
import requests
import copy

from utils import prepare_image_patch_bbx,create_mask_with_bbox,show_original_image,show_transferred_maskandimage, generate_plot


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode,
                 video_folder=None, video_fps=1, frames_upbound=32, force_sample=False):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.image_processor_mask = copy.deepcopy(image_processor)
        self.model_name = get_model_name_from_path(self.model_config._name_or_path)
        self.task_name = task_name
        self.conv_mode = conv_mode

        #! video 처리 logic 추가
        self.video_folder = video_folder

        self.video_data_args = argparse.Namespace(
            video_fps=video_fps,
            frames_upbound=frames_upbound,
            force_sample=force_sample,
        )

        if self.model_name == "llama3-llava-next-8b" or self.model_name == "llava-v1.6-vicuna-7b" or self.model_name == "llava-v1.5-7b" or self.model_name == "llava-v1.5-13b":
            self.image_processor_mask.do_normalize=False
            self.image_processor_mask.do_rescale=False
        elif self.model_name == "llava-next-qwen-32b" or "onevision" in self.model_name.lower() or "qwen" in self.model_name.lower():
            self.image_processor_mask.image_mean = (0, 0, 0)
            self.image_processor_mask.image_std = (1, 1, 1)
            self.image_processor_mask.rescale_factor = 1

    def __getitem__(self, index):

        line = self.questions[index]
        question = line["question"]
        question = question + " \nAnswer the question using a single word or phrase."

        is_video = "video" in line and line["video"] != ""

        if is_video:
            video_file = str(line["video"])
            video_path = os.path.join(self.video_folder, video_file)
        else:
            #! 기존엔 img는 .jpg로만 처리됨 -> 다양한 확장자 대응하도록 수정
            img_id_str = str(line["img_id"])
            if os.path.splitext(img_id_str)[1]:
                image_file = img_id_str
            else:
                image_file = None
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]:
                    candidate = img_id_str + ext
                    if os.path.exists(os.path.join(self.image_folder, candidate)):
                        image_file = candidate
                        break
                if image_file is None:
                    raise FileNotFoundError(f"No image found for {img_id_str} in {self.image_folder}")


        qs = DEFAULT_IMAGE_TOKEN + "\n" + question  #

        conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # prompt 뜯어보기
        #! "A chat between a curious user and an artificial intelligence assistant. 
        #! The assistant gives helpful, detailed, and polite answers to the user's questions.
        #! USER: <image>\nWhat direction does the circle move in the frame? \nAnswer the question using a single word or phrase. ASSISTANT:"
        prompt = conv.get_prompt()
        if self.model_name == "llama3-llava-next-8b":
            prompt+=" \n"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

        #! image & video 처리 분기

        #! video 처리 logic
        if is_video:
            video_frames, video_time, frame_time, num_frames = process_video_with_decord(
                video_path, self.video_data_args
            )
            image_tensor = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]

            image_tensor = [image_tensor.to(dtype=torch.float16)]

            if isinstance(video_frames, np.ndarray):
                h, w = video_frames.shape[1], video_frames.shape[2]
                image_sizes = [(w, h)]
            else:
                image_sizes = [video_frames[0].size]
            modality = "video"
            mask_tensor = None
        #! image 처리 logic
        else:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model_config)
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            image_sizes = [image.size]
            modality = "image"

            # bounding box mask (기존 로직 그대로)
            if self.task_name == "CompareAttr" or self.task_name == "ChooseRel" or self.task_name == "LogicalObj":
                bounding_boxes=[]
                bounding_boxes.append((int(line[f'object1 x']), int(line[f'object1 y']), int(line[f'object1 x'])+int(line[f'object1 w']), int(line[f'object1 y'])+int(line[f'object1 h'])))
                if line[f'object2 x'] !="-":
                    bounding_boxes.append((int(line[f'object2 x']), int(line[f'object2 y']), int(line[f'object2 x'])+int(line[f'object2 w']), int(line[f'object2 y'])+int(line[f'object2 h'])))
            elif self.task_name=="ChooseAttr" or self.task_name=="ChooseCat" or self.task_name=="QueryAttr":
                bounding_boxes = [(int(line['central object x']), int(line['central object y']), int(line['central object x'])+int(line['central object w']), int(line['central object y'])+int(line['central object h']))]
            else:
                bounding_boxes = None

            if bounding_boxes !=None:
                mask = create_mask_with_bbox(image, bounding_boxes)
                mask_tensor = process_images([mask], self.image_processor_mask, self.model_config)
                mask_tensor = [_image.to(dtype=torch.float16) for _image in mask_tensor]
            else:
                mask_tensor=None

        # show_original_image(image, bounding_boxes, self.model_name.replace('-', '_').replace('.', '_'), save_name=str(line["img_id"]), question = line["question"], answer=line["answer"])
        # if mask_tensor[0].ndim==3:
        #     for ind, (ma, img) in enumerate(zip(mask_tensor, image_tensor)):
        #         show_transferred_maskandimage(ma,img, ind, self.model_name.replace('-', '_').replace('.', '_'), save_name=str(line["img_id"]))
        # else:
        #     for ind, (ma, img) in enumerate(zip(mask_tensor[0], image_tensor[0])):
        #         show_transferred_maskandimage(ma,img, ind, self.model_name.replace('-', '_').replace('.', '_'), save_name=str(line["img_id"]))

        return input_ids, image_tensor, image_sizes, prompt, mask_tensor, modality



    def __len__(self):
        return len(self.questions)


# def collate_fn(batch):
#     input_ids, image_tensors, image_sizes, prompts, mask_tensors = zip(*batch)

#     input_ids = input_ids[0]
#     image_tensors = image_tensors[0]
#     image_sizes=image_sizes[0]
#     mask_tensors=mask_tensors[0]
#     return input_ids, image_tensors, image_sizes, prompts,mask_tensors

def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompts, mask_tensors, modalities = zip(*batch)

    input_ids = input_ids[0]
    image_tensors = image_tensors[0]
    image_sizes = image_sizes[0]
    mask_tensors = mask_tensors[0]
    modality = modalities[0]
    return input_ids, image_tensors, image_sizes, prompts, mask_tensors, modality


#! video 받을 수 있도록 수정함
# def create_data_loader(questions, image_folder, batch_size, num_workers, tokenizer, image_processor, model_config, task_name, conv_mode):
def create_data_loader(questions, image_folder, batch_size, num_workers, tokenizer, image_processor, model_config, task_name, conv_mode,
                       video_folder=None, video_fps=1, frames_upbound=32, force_sample=False):
    assert batch_size == 1, "batch_size must be 1"
    # dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode)
    # data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    # return data_loader

    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode,
                            video_folder=video_folder, video_fps=video_fps,
                            frames_upbound=frames_upbound, force_sample=force_sample)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def find_token_range(tokenizer, token_array, substring, model_name):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = tokenizer.convert_ids_to_tokens(token_array)
  
  if model_name in ("llava-v1.6-vicuna-7b", "llava-v1.5-7b", "llava-v1.5-13b", "LLaVA-NeXT-Video-7B"):
      whole_string = "".join(toks).replace("▁", " ")
  elif model_name in ("llama3-llava-next-8b", "llava-next-qwen-32b", "llava-onevision-qwen2-7b-si") or "onevision" in model_name.lower() or "qwen2" in model_name.lower():
    whole_string = "".join(toks).replace("Ġ"," ").replace("Ċ","\n")

  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


@torch.no_grad()
def generate_llava(
    self,
    mask=None, #[5, 3, 336, 336]
    args=None,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")


    if images is not None:
        if args.certain_part_image:
            (inputs_, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            patched_mask = self.prepare_image_patch_bbx(mask, image_sizes=image_sizes) #[2352, 14*14, 3]   patch_size:14
            patched_mask = np.array(patched_mask[0].cpu())
            target_object = np.array([255, 0, 0], dtype=np.uint8) #red
            match_object = np.all(patched_mask == target_object, axis=-1)
            objects_indices = np.where(np.any(match_object, axis=1))[0]
            target_pad = np.array([-1, -1, -1], dtype=np.int8) #pad
            match_pad = np.all(patched_mask == target_pad, axis=-1)
            pad_indices = np.where(np.any(match_pad, axis=1))[0]
            original_patch_number = (mask[0].size(-1)//self.get_vision_tower().config.patch_size)**2
            original_patch_indices = list(range(patched_mask.shape[0]))[0:original_patch_number]
            hd_patch_indice = list(range(patched_mask.shape[0]))[original_patch_number:]
            objects_indices_in_hd =objects_indices[objects_indices>=original_patch_number]
            return patched_mask, objects_indices, pad_indices,original_patch_indices,hd_patch_indice,objects_indices_in_hd, inputs_embeds.shape, super(self.__class__, self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        else:
            (inputs_, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            return inputs_embeds.shape, super(self.__class__, self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)



#! Attention Knock out 없이 그냥 모델 돌리기
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
    decoded_generated_first_id = tokenizer.decode(generated_first_id.item())
    #! 정답의 첫 토큰 ID
    #! csv 파일 기준 answer를 captalized해서 동일하게 맞추기
    answer = answer.capitalize()
    gt_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    gt_first_token_id = gt_token_ids[0]
    gt_first_token_id_tensor = torch.tensor([gt_first_token_id], device=generated_first_id.device)
    logits_first_answer_token = output_details['scores'][0]
    probs = torch.softmax(logits_first_answer_token, dim=-1)[0]
    #! GT 토큰과 예측 토큰 각각의 base score
    gt_base_score = probs[gt_first_token_id_tensor].item()
    predicted_base_score = probs[generated_first_id].item()
    if decoded_generated_first_id.strip().lower() == answer.strip().lower():
        is_correct_bool = True
    else:
        is_correct_bool = False
    predicted_answer = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip().lower()
    if args.certain_part_image:
        #! gt_first_token_id_tensor 기준으로 앞으로 정답이 모두 tracing 됨
        #! Left는 19941임
        return gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id_tensor, generated_first_id, inputs_embeds_shape, is_correct_bool, objects_indices, pad_indices, original_patch_indices, hd_patch_indice, objects_indices_in_hd, patched_mask
    else:
        return gt_base_score, predicted_base_score, predicted_answer, gt_first_token_id_tensor, generated_first_id, inputs_embeds_shape, is_correct_bool



def blockdesc2range(des, dataset_dict, question_id, input_ids, inputs_embeds_shape, tokenizer, model_name, args=None):
    if des=="Last":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        ntoks = input_ids.shape[1] + image_dim - 1
        source_ = ntoks - 1
        return [source_]
    if des=="Question":
        question = dataset_dict[question_id]["question"]
        #! LLaVA-1.5 7B, Vision Encoder : CLIP-ViT-L-336px / 14
        #! input img size : 336 -> patch token = (336/14)*(336/14) = 576
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        #! image_dim = 576 나옴
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [input_ids[0].shape[0]]
        #! ex.) image_token_indices = [-1, 35, 64]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        #! <img> 뒤 text들은 input_ids_noim[1]에 저장됨
        #! question_range는 find_token_range에서 input_ids_noim[1]에서 상대 위치 반환을 통해 얻음
        question_range = find_token_range(tokenizer, input_ids_noim[1], question, model_name)
        question_range = [x for x in range(question_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                           question_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        #! ex : question_range = [1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198]
        return question_range
    if des=="Image":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_range = [x for x in range(torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0],
                                        torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0] + image_dim)]
        return image_range
    if des=="True Option":
        true_option = dataset_dict[question_id]["true option"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [ input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        true_option_range = find_token_range(tokenizer, input_ids_noim[1], true_option, model_name)
        true_option_range = [x for x in range(true_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                              true_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return true_option_range
    if des=="False Option":
        false_option = dataset_dict[question_id]["false option"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
            input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        false_option_range = find_token_range(tokenizer, input_ids_noim[1], false_option, model_name)
        false_option_range = [x for x in range(false_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                               false_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return false_option_range
    if des=="Central Object":
        central_object = dataset_dict[question_id]["central object name"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
            input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        central_object_range = find_token_range(tokenizer, input_ids_noim[1], central_object, model_name)
        central_object_range = [x for x in range(central_object_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                                 central_object_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        return central_object_range

    if des=="Instruction":
        #! "Answer the question using a single word or phrase. ASSISTANT:" 구간
        #! = Question 끝 ~ Last 사이의 모든 토큰
        question = dataset_dict[question_id]["question"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        question_range = find_token_range(tokenizer, input_ids_noim[1], question, model_name)
        question_end = question_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1
        ntoks = input_ids.shape[1] + image_dim - 1
        last_token_idx = ntoks - 1

        if args.block_ASSIST:
            #! question 끝부터 Last 직전까지 (ASSISTANT 포함, \n 포함 — 간접 경로 완전 차단)
            instruction_range = list(range(question_end, last_token_idx))
        else:
            #! question 끝부터 Last 직전까지 (ASSISTANT 토큰 제외) (기본값)
            if "qwen" in model_name.lower() or "onevision" in model_name.lower():
                assistant_str = "assistant"  # ChatML format
            else:
                assistant_str = "ASSISTANT"  # Vicuna format
            
            assistant_range_rel = find_token_range(tokenizer, input_ids_noim[1], assistant_str, model_name)
            assistant_start = assistant_range_rel[0] + len(input_ids_noim[0]) + 1 + image_dim - 1
            assistant_end = assistant_range_rel[1] + len(input_ids_noim[0]) + 1 + image_dim - 1
            assistant_set = set(range(assistant_start, assistant_end))
            instruction_range = [x for x in range(question_end, last_token_idx) if x not in assistant_set]

        return instruction_range

    if des=="Question without Options":
        true_option = dataset_dict[question_id]["true option"]
        false_option = dataset_dict[question_id]["false option"]
        question = dataset_dict[question_id]["question"]
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_token_indices = [-1] + torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist() + [
            input_ids[0].shape[0]]
        input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            input_ids_noim.append(input_ids[0][image_token_indices[i] + 1:image_token_indices[i + 1]])
        true_option_range = find_token_range(tokenizer, input_ids_noim[1], true_option, model_name)
        false_option_range = find_token_range(tokenizer, input_ids_noim[1], false_option, model_name)
        question_range = find_token_range(tokenizer, input_ids_noim[1], question, model_name)
        true_option_range = [x for x in range(true_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                              true_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        false_option_range = [x for x in range(false_option_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                               false_option_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        question_range = [x for x in range(question_range[0] + len(input_ids_noim[0]) + 1 + image_dim - 1,
                                           question_range[1] + len(input_ids_noim[0]) + 1 + image_dim - 1)]
        question_withoutOptions_range = [item for item in question_range if
                                         item not in true_option_range + false_option_range]
        return question_withoutOptions_range


def blockdesc2range_patches(des, input_ids, inputs_embeds_shape, central_object_patch_indices, pad_patch_indices, hd_patch_indice, objects_indices_in_hd, original_patch_indices):
    if des=="Image Without Central Object":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        other_indices_without_central_object = list(set(range(image_dim)) - set(central_object_patch_indices) - set(pad_patch_indices))
        image_without_central_object_range = (np.array(other_indices_without_central_object) + image_index).tolist()
        return image_without_central_object_range
    if des=="Image Central Object":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        image_central_object_range = (np.array(central_object_patch_indices) + image_index).tolist()
        return image_central_object_range
    if des=="Image Pad":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        image_pad_range = (np.array(pad_patch_indices) + image_index).tolist()
        return image_pad_range
    if des=="Image Without Central Object with pad":
        image_dim = inputs_embeds_shape[1] - (input_ids.shape[-1] - 1)
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        other_indices_without_central_object = list(set(range(image_dim)) - set(central_object_patch_indices) - set(pad_patch_indices))
        image_pad_range = (np.array(pad_patch_indices) + image_index).tolist()
        image_without_central_object_range = (np.array(other_indices_without_central_object) + image_index).tolist()
        return image_without_central_object_range + image_pad_range
    if des=="Image Original Patch":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        original_patch_range = (np.array(original_patch_indices) + image_index).tolist()
        return original_patch_range
    if des=="Image HD Patch Indice":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        hd_patch_indice_range = (np.array(hd_patch_indice) + image_index).tolist()
        return hd_patch_indice_range
    if des=="Image Central Object in HD Patch Indice":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        objects_indices_in_hd_range = (np.array(objects_indices_in_hd) + image_index).tolist()
        return objects_indices_in_hd_range
    if des=="Image HD Patch Without Central Object Indice":
        image_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()[0]
        other_indices_without_central_object_in_hd = list(set(hd_patch_indice) - set(objects_indices_in_hd))
        other_indices_without_central_object_in_hd_range = ( np.array(other_indices_without_central_object_in_hd) + image_index).tolist()
        return other_indices_without_central_object_in_hd_range





# Information flow analysis
def InforFlowAna(args):


    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    cache_dir = os.environ.get("HF_HOME", None)

    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base,model_name,device_map="auto",attn_implementation=None, cache_dir=cache_dir)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device_map="auto", attn_implementation=None, cache_dir=cache_dir
    )

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
            re = {
                "question_id": question_id,
                "image": img_id,
                "goden answer": answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "question": question,
                "gt_base_score": gt_base_score,
                "predicted_base_score": predicted_base_score,
            }
            results.append(re)
            continue

        for temp2, block_desc in block_descs:

            if args.block_all_layers:
                block_config = {
                    l: copy.deepcopy(temp2)
                    for l in range(model.config.num_hidden_layers)
                }
                inps["max_new_tokens"] = 1

                #! full probs 반환 → GT/predicted 둘 다 indexing
                new_probs, knocked_predicted_answer = trace_with_attn_block_llava(
                    model, inps, block_config, block_desc, model_name, tokenizer=tokenizer, last_token_idx=last_token_idx
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
                        l: copy.deepcopy(temp2)
                        for l in layerlist
                    }

                    inps["max_new_tokens"] = 1
                    new_probs, knocked_predicted_answer = trace_with_attn_block_llava(
                        model, inps, block_config, block_desc, model_name, tokenizer=tokenizer, last_token_idx=last_token_idx
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
    parser.add_argument("--model-path", type=str, default="")
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


