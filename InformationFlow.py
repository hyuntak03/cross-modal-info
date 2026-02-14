# InformationFlow.py
import copy
import pdb

from methods import *

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torch.multiprocessing as mp

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
        elif self.model_name == "llava-next-qwen-32b":
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
  elif model_name in ("llama3-llava-next-8b", "llava-next-qwen-32b"):
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

    #! 정답의 첫 토큰 ID (생성 컨텍스트에 맞게 space prefix)
    gt_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    gt_first_token_id = gt_token_ids[0]


    if decoded_generated_first_id.strip().lower() == answer.strip().lower():
        is_correct_bool = True
        first_answer_token_id = generated_first_id
    else:
        is_correct_bool = False
        first_answer_token_id = torch.tensor([gt_first_token_id], 
                                            device=generated_first_id.device)

    # first_answer_token_id = answer_token_id[:, 0]
    # logits_first_answer_token = output_details['scores'][0]

    # [base_score_first] = torch.softmax(logits_first_answer_token, dim=-1)[0][first_answer_token_id]  # (1,1)
    # base_score_first = base_score_first.item()


    logits_first_answer_token = output_details['scores'][0]

    [base_score_first] = torch.softmax(logits_first_answer_token, dim=-1)[0][first_answer_token_id]
    base_score_first = base_score_first.item()

    predicted_answer = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip().lower()

    if args.certain_part_image:
        return base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape, is_correct_bool, objects_indices, pad_indices, original_patch_indices,hd_patch_indice, objects_indices_in_hd, patched_mask
    else:
        return base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape, is_correct_bool



def blockdesc2range(des, dataset_dict, question_id, input_ids, inputs_embeds_shape, tokenizer, model_name):
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
# def InforFlowAna(args):
#! 데이터 병렬
def InforFlowAna(rank, world_size, args):

    torch.cuda.set_device(rank)
    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    cache_dir = os.environ.get("HF_HOME", None)

    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base,model_name,device_map="auto",attn_implementation=None, cache_dir=cache_dir)

    #! 단일 gpu
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path, args.model_base, model_name,
    #     device_map="auto", attn_implementation=None, cache_dir=cache_dir
    # )

    #! multi-gpu (데이터 병렬)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device_map={"": rank}, attn_implementation=None, cache_dir=cache_dir
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

    all_questions = [ {**detail, "q_id":qu_id} for qu_id, detail in dataset_dict.items()]

    #! 데이터 병렬을 위해 데이터 슬라이싱
    questions = all_questions[rank::world_size]

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

        #! run_original 돌리면, 얘가 대답한 answer token index가 나옴 (vocab 기준)
        if args.certain_part_image:
            base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape, is_correct_bool, central_object_patch_indices, pad_patch_indices,original_patch_indices,hd_patch_indice,  objects_indices_in_hd, patched_mask = run_original(model, inps, tokenizer, model_name, answer, mask_tensor, args=args)
        else:
            # base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape = run_original(model, inps, tokenizer, model_name, args=args)
            base_score_first, predicted_answer, first_answer_token_id, inputs_embeds_shape, is_correct_bool = run_original(model, inps, tokenizer, model_name, answer, args=args)

        if is_correct_bool == False:
            is_correct=False
        else:
            is_correct=True
            index += 1
            print("Finish samples:", index)


        #get range
        block_descs_split = args.block_description.split("->")
        if args.certain_part_image:
            range1 = blockdesc2range_patches(block_descs_split[0], input_ids, inputs_embeds_shape,central_object_patch_indices, pad_patch_indices, hd_patch_indice,objects_indices_in_hd, original_patch_indices)
        else:
            range1 = blockdesc2range(block_descs_split[0], dataset_dict, question_id, input_ids, inputs_embeds_shape,tokenizer, model_name)
        range2 = blockdesc2range(block_descs_split[1], dataset_dict, question_id, input_ids, inputs_embeds_shape,tokenizer, model_name)
        block_descs = [([range1, range2], args.block_description)]

        for block_ids, block_desc in block_descs:

            temp2 = [(stok1, stok0) for stok0 in block_ids[0] for stok1 in block_ids[1]]

            for layer in range(model.config.num_hidden_layers):
                layerlist = [
                    l for l in range(
                        max(0, layer - args.window // 2), min(model.config.num_hidden_layers, layer - (-args.window // 2))
                    )
                ]
                block_config = {
                    l:copy.deepcopy(temp2)
                    for l in layerlist
                }

                inps["max_new_tokens"] = 1
                #! 그래서 여기서 token_id 인자 넘겨주게 됨 (만약 틀린 샘플에 대해선 그냥 로직 바꾸면 될 듯?)
                new_score_first = trace_with_attn_block_llava(
                    model, inps, block_config, first_answer_token_id, block_desc, model_name
                )
                new_score_first = new_score_first.cpu().item()

                re={
                    "question_id": question_id,
                    "image": img_id,
                    "goden answer": answer,
                    "predicted answer": predicted_answer,
                    "is_correct": is_correct,
                    "question": question,
                    "block_desc": block_desc,
                    "layer": layer,
                    "base_score_first": base_score_first,
                    "new_score_first": new_score_first,
                    "relative diff first": (new_score_first - base_score_first) * 100.0 / base_score_first,
                }
                results.append(re)


    save_name = "_".join([des[1].replace(" ", "_").replace("->", "___") for des in block_descs])
    
    if args.noHD_noPad:
        save_name=save_name+"_noHD_noPad"
    tmp = pd.DataFrame.from_records(results)

    #! multi-gpu: 각 rank별 임시 저장만 하고 return (plot은 merge 후에)
    if world_size > 1:
        os.makedirs("/tmp/infoflow_parts", exist_ok=True)
        tmp.to_csv(f'/tmp/infoflow_parts/rank{rank}.csv', index=False)
        print(f"[GPU {rank}] Saved {len(tmp)} results to /tmp/infoflow_parts/rank{rank}.csv")
        return

    model_name = model_name.replace('-', '_').replace('.', '_')
    os.makedirs(f"output/information_flow/{model_name}/{task_name}/val/{save_name}", exist_ok=True)
    
    tmp.to_csv(f'output/information_flow/{model_name}/{task_name}/val/{save_name}/{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}.csv', index=False)

    base_path = f'output/information_flow/{model_name}/{task_name}/val/{save_name}/{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}'

    # 전체
    # generate_plot(tmp, f'{base_path}_first_all.pdf', x="layer", y="relative diff first", hue="block_desc", layers=model.config.num_hidden_layers)

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

    #! gpu 수 인자 추가
    parser.add_argument("--num-gpus", type=int, default=1)

    args = parser.parse_args()

    block_descs_split = args.block_description.split("->")
    if "Image " in block_descs_split[0]:
        args.certain_part_image=True

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    # InforFlowAna(args)

    if args.num_gpus > 1:
        mp.spawn(InforFlowAna, args=(args.num_gpus, args), 
                 nprocs=args.num_gpus, join=True)
        
        #! merge all rank results
        dfs = [pd.read_csv(f'/tmp/infoflow_parts/rank{i}.csv') for i in range(args.num_gpus)]
        merged = pd.concat(dfs, ignore_index=True)
        
        #! 저장 경로 구성 (InforFlowAna 내부와 동일하게)
        model_name_clean = get_model_name_from_path(os.path.expanduser(args.model_path))
        task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
        
        #! block_descs에서 save_name 구성
        save_name = args.block_description.replace(" ", "_").replace("->", "___")
        if args.noHD_noPad:
            save_name = save_name + "_noHD_noPad"
        
        model_name_clean = model_name_clean.replace('-', '_').replace('.', '_')
        out_dir = f"output/information_flow/{model_name_clean}/{task_name}/val/{save_name}"
        os.makedirs(out_dir, exist_ok=True)
        
        csv_name = f'{args.refined_dataset.split("/")[-1].split(".csv")[0]}_window{args.window}_{save_name}'
        merged.to_csv(f'{out_dir}/{csv_name}.csv', index=False)
        print(f"Merged {len(merged)} results -> {out_dir}/{csv_name}.csv")
        
        #! plot 생성
        base_path = f'{out_dir}/{csv_name}'
        
        # 전체 layer 수는 merged 데이터에서 추론
        num_layers = merged["layer"].max() + 1
        
        tmp_correct = merged[merged["is_correct"] == True]
        if len(tmp_correct) > 0:
            generate_plot(tmp_correct, f'{base_path}_first_correct.pdf', 
                         x="layer", y="relative diff first", hue="block_desc", layers=num_layers)
        
        tmp_incorrect = merged[merged["is_correct"] == False]
        if len(tmp_incorrect) > 0:
            generate_plot(tmp_incorrect, f'{base_path}_first_incorrect.pdf', 
                         x="layer", y="relative diff first", hue="block_desc", layers=num_layers)
        
        #! 임시 파일 정리
        import shutil
        shutil.rmtree('/tmp/infoflow_parts', ignore_errors=True)
        
    else:
        InforFlowAna(0, 1, args)  #! single-gpu: rank=0, world_size=1


