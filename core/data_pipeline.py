import copy
import math
import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from types import MethodType
from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerateOutput
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import process_video_with_decord
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from core.utils import prepare_image_patch_bbx, create_mask_with_bbox


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
        try:
            return self._load_item(index)
        except Exception as e:
            print(f"[WARN] Sample {index} 로드 실패 (스킵): {e}")
            return None

    def _load_item(self, index):

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


def collate_fn(batch):
    # None 필터링 (로드 실패 샘플 스킵)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    input_ids, image_tensors, image_sizes, prompts, mask_tensors, modalities = zip(*batch)

    input_ids = input_ids[0]
    image_tensors = image_tensors[0]
    image_sizes = image_sizes[0]
    mask_tensors = mask_tensors[0]
    modality = modalities[0]
    return input_ids, image_tensors, image_sizes, prompts, mask_tensors, modality


#! video 받을 수 있도록 수정함
def create_data_loader(questions, image_folder, batch_size, num_workers, tokenizer, image_processor, model_config, task_name, conv_mode,
                       video_folder=None, video_fps=1, frames_upbound=32, force_sample=False):
    assert batch_size == 1, "batch_size must be 1"

    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, task_name, conv_mode,
                            video_folder=video_folder, video_fps=video_fps,
                            frames_upbound=frames_upbound, force_sample=force_sample)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=collate_fn,
        pin_memory=True, prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return data_loader


def find_token_range(tokenizer, token_array, substring, model_name):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = tokenizer.convert_ids_to_tokens(token_array)

  # 토큰별로 normalize하여 위치 추적을 정확하게 수행
  if model_name in ("llava-v1.6-vicuna-7b", "llava-v1.5-7b", "llava-v1.5-13b", "LLaVA-NeXT-Video-7B"):
      norm = lambda t: t.replace("▁", " ").replace("<0x0A>", "\n")
  elif model_name in ("llama3-llava-next-8b", "llava-next-qwen-32b", "llava-onevision-qwen2-7b-si") or "onevision" in model_name.lower() or "qwen2" in model_name.lower():
      norm = lambda t: t.replace("Ġ", " ").replace("Ċ", "\n")
  else:
      norm = lambda t: t

  normed = [norm(t) for t in toks]
  whole_string = "".join(normed)

  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, nt in enumerate(normed):
    loc += len(nt)
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
            # knockout 재사용을 위해 캐싱
            self._cached_inputs_embeds = inputs_embeds
            self._cached_position_ids = position_ids
            self._cached_attention_mask = attention_mask
            return inputs_embeds.shape, super(self.__class__, self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


def generate_llava_cached(self, **kwargs):
    """vision encoder를 건너뛰고 캐시된 inputs_embeds로 generate 수행."""
    inputs_embeds = self._cached_inputs_embeds
    position_ids = self._cached_position_ids
    attention_mask = self._cached_attention_mask
    return inputs_embeds.shape, super(self.__class__, self).generate(
        position_ids=position_ids, attention_mask=attention_mask,
        inputs_embeds=inputs_embeds, **kwargs)


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
