# model_loader.py
# 모델 로딩 유틸리티 (lmms_eval 호환)
# - parse_model_args: "key=val,key=val" 문자열 파싱
# - load_model_from_args: HF repo / local path / LoRA 모델 로딩
#
# 사용:
#   from model_loader import parse_model_args, load_model_from_args

import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def parse_model_args(args_string):
    """
    lmms_eval 스타일 model_args 파싱.
    "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto"
    → {"pretrained": "...", "conv_template": "qwen_1_5", "device_map": "auto"}
    """
    if not args_string:
        return {}
    result = {}
    for item in args_string.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() == "none":
            val = None
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        result[key.strip()] = val
    return result


def load_model_from_args(model_args_dict):
    """
    model_args dict로부터 모델을 로드.

    지원 패턴:
      1. HF repo:  pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si
      2. Local:    pretrained=/path/to/checkpoint
      3. LoRA:     lora_pretrained=/path/to/lora,pretrained=lmms-lab/llava-onevision-qwen2-7b-ov

    Returns:
        tokenizer, model, image_processor, context_len, model_name, conv_template
    """
    pretrained = model_args_dict.get("pretrained", "")
    lora_pretrained = model_args_dict.get("lora_pretrained", None)
    device_map = model_args_dict.get("device_map", "auto")
    attn_implementation = model_args_dict.get("attn_implementation", None)
    conv_template = model_args_dict.get("conv_template", "qwen_1_5")
    cache_dir = os.environ.get("HF_HOME", None)

    if lora_pretrained:
        model_path = lora_pretrained
        model_base = pretrained
        model_name = get_model_name_from_path(lora_pretrained)
        print(f"[MODEL] LoRA loading: base={pretrained}, lora={lora_pretrained}")
    else:
        model_path = pretrained
        model_base = None
        model_name = get_model_name_from_path(pretrained)
        print(f"[MODEL] Loading: {pretrained}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        device_map=device_map,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )
    model.eval()

    # video config 오버라이드
    if "mm_spatial_pool_stride" in model_args_dict:
        model.config.mm_spatial_pool_stride = model_args_dict["mm_spatial_pool_stride"]
    if "mm_spatial_pool_mode" in model_args_dict:
        model.config.mm_spatial_pool_mode = model_args_dict["mm_spatial_pool_mode"]

    return tokenizer, model, image_processor, context_len, model_name, conv_template


def load_model_legacy(model_path, model_base=None, conv_mode="qwen_1_5"):
    """
    기존 --model-path / --model-base 방식 로딩.
    HF repo name도 지원 (load_pretrained_model이 자동 처리).
    """
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    cache_dir = os.environ.get("HF_HOME", None)

    print(f"[MODEL] Loading (legacy): {model_path}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        device_map="auto", attn_implementation=None, cache_dir=cache_dir,
    )
    model.eval()

    return tokenizer, model, image_processor, context_len, model_name, conv_mode
