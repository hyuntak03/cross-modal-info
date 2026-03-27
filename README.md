# Cross-modal Information Flow in Multimodal Large Language Models

VLM의 cross-modal information flow를 분석하기 위한 도구 모음. Attention knockout, logit lens 등을 통해 vision-language 모델 내부의 정보 흐름을 추적합니다.

Based on: [Cross-modal Information Flow in Multimodal Large Language Models](https://arxiv.org/abs/2411.18620) (Zhang et al., CVPR 2024)

## Project Structure

```
cross-modal-info/
├── analysis/                   # Analysis scripts
│   ├── information_flow.py         # Image-level attention knockout
│   ├── video_information_flow.py   # Video cross-frame attention knockout
│   └── logit_lens.py               # Logit lens probing
├── core/                       # Core modules
│   ├── data_pipeline.py            # Data processing & custom generate
│   ├── dataset_loader.py           # HF/CSV dataset loading
│   ├── methods.py                  # Attention knockout implementation
│   ├── model_loader.py             # Model loading utilities
│   └── utils.py                    # Misc helpers
├── tasks/                      # Task configs (lmms-eval style YAML)
│   ├── mvbench/
│   ├── tvbench/
│   ├── videomme/
│   ├── activitynetqa/
│   └── extrapolation_bench/
├── scripts/                    # SLURM / launch scripts
│   ├── run_information_flow.sh
│   ├── run_crossframe_mvbench.sh
│   ├── run_crossframe_videomme.sh
│   └── run_logit_lens.sh
└── datasets/                   # GQA-based image datasets (original)
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM.git
   cd cross-modal-information-flow-in-MLLM
   ```

2. Install [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) following their instructions.
   After installation, set `PYTHONPATH` to point to the LLaVA-NeXT directory.

## Supported Models

| Model | `pretrained` | `conv_template` |
|-------|-------------|-----------------|
| LLaVA-1.6-Vicuna-7B | `liuhaotian/llava-v1.6-vicuna-7b` | `vicuna_v1` |
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` | `vicuna_v1` |
| LLaVA-1.5-13B | `liuhaotian/llava-v1.5-13b` | `vicuna_v1` |
| LLaMA3-LLaVA-NeXT-8B | `lmms-lab/llama3-llava-next-8b` | `llava_llama_3` |
| LLaVA-OneVision-0.5B | `lmms-lab/llava-onevision-qwen2-0.5b-si` | `qwen_1_5` |
| LLaVA-OneVision-7B | `lmms-lab/llava-onevision-qwen2-7b-si` | `qwen_1_5` |

## Usage

### 1. Information Flow (Image, Attention Knockout)

```bash
export PYTHONPATH=/path/to/LLaVA-NeXT
export HF_HOME=/path/to/hf_cache

python analysis/information_flow.py \
    --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto" \
    --refined_dataset "datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv" \
    --image-folder "/path/to/gqa/images" \
    --block_description "Image->Last" \
    --window 9 \
    --temperature 0 \
    --max_new_tokens 128
```

**`block_description`** options:
- `Question->Last` / `Image->Last` / `Last->Last`
- `Image->Question`
- `Image Central Object->Question` / `Image Without Central Object->Question`

### 2. Video Cross-Frame Flow (Attention Knockout)

```bash
python analysis/video_information_flow.py \
    --model_args "pretrained=liuhaotian/llava-v1.6-vicuna-7b,conv_template=vicuna_v1,max_frames_num=8,device_map=auto,force_sample=True" \
    --task tvbench_moving_direction \
    --cross_frame_targets cross-frame \
    --option MCQ \
    --window 9 \
    --temperature 0 \
    --max_new_tokens 1 \
    --num_workers 6
```

**`cross_frame_targets`** options:
- `cross-frame` -- Frame_i -/-> Frame_j (j < i)
- `intra-frame` -- Within-frame spatial knockout
- `video-to-question` / `video-to-last` / `question-to-last`
- `perframe-to-last` -- Per-frame contribution to last token

### 3. Logit Lens

```bash
python analysis/logit_lens.py \
    --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto" \
    --task mvbench_moving_direction \
    --option MCQ \
    --temperature 0 \
    --max_new_tokens 1
```

### Task Datasets

Tasks are defined as YAML configs under `tasks/` (lmms-eval style). Available tasks:

| Directory | Examples |
|-----------|----------|
| `mvbench/` | `mvbench_moving_direction`, `mvbench_action_sequence`, ... |
| `tvbench/` | `tvbench_moving_direction` |
| `videomme/` | `videomme` |
| `activitynetqa/` | `activitynetqa` |

For image tasks, use the original GQA CSV datasets in `datasets/`.

## Citation

```bibtex
@article{zhang2024cross,
  title={Cross-modal Information Flow in Multimodal Large Language Models},
  author={Zhang, Zhi and Yadav, Srishti and Han, Fengze and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2411.18620},
  year={2024}
}
```

## Acknowledgement

Built upon [Dissecting Factual Predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions) and [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT). Image datasets collected from [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html).
