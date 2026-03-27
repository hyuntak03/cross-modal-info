import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import pdb

from core.methods import *

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns

import argparse
from tqdm import tqdm

from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from core.data_pipeline import create_data_loader
from core.model_loader import parse_model_args, load_model_from_args, load_model_legacy
from core.dataset_loader import load_dataset_as_questions, list_tasks




def generate_plot_attrscore(data, save_file, x="layer", ys="", layer_num=0):

    #! Noncapitalized/Capitalized 쌍이면 같은 색 + 실선/점선으로 구분
    #! "[옵션명]" 패턴이 있으면 옵션별 색 매핑
    import re

    # 옵션 텍스트 추출하여 고유 옵션 수 파악
    option_names = []
    for y in ys:
        m = re.search(r'\[(.+?)\]', y)
        if m:
            option_names.append(m.group(1).lower())
        else:
            option_names.append(y)
    unique_options = list(dict.fromkeys(option_names))  # 순서 유지 중복 제거

    #! 옵션 수 기준으로 색상 할당 (구분 잘 되는 tab10 사용)
    if len(unique_options) <= 4:
        base_colors = ["#f20089", "#5c95ff", "#2db84b", "#ff8c1a"]
    else:
        base_colors = sns.color_palette("tab10", len(unique_options)).as_hex()

    option_color_map = {opt: base_colors[i] for i, opt in enumerate(unique_options)}

    sns.set(context="notebook")
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(5, 4))

    for ind, y in enumerate(ys):
        opt_key = option_names[ind]
        color = option_color_map[opt_key]
        #! Capitalized → 점선, Noncapitalized → 실선
        is_cap = y.startswith("Capitalized")
        linestyle = "--" if is_cap else "-"

        ax = sns.lineplot(data, x=x, y=y,
                          label=y, color=color,
                          linestyle=linestyle,
                          linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, layer_num + 0.5)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.legend(fontsize=5, handlelength=2)

    plt.savefig(save_file)
    plt.close()




def run_original(model, inps, tokenizer, model_name):
    with torch.inference_mode():
        output_details = model.generate(**inps)

    answer_token_id = output_details['sequences']

    first_answer_hidden_id=0

    predicted_answer = tokenizer.batch_decode(answer_token_id, skip_special_tokens=True)[0].strip().lower()



    hs_alllayer_first_answer_gen=[]
    for layer_id in range(model.config.num_hidden_layers+1):
        hs_first_answer_gen = output_details['hidden_states'][first_answer_hidden_id][layer_id][:,-1,:].squeeze().cpu() #torch.Size([4096])
        hs_alllayer_first_answer_gen.append(hs_first_answer_gen)
    return hs_alllayer_first_answer_gen, predicted_answer




def cache_hiddenstate(data_loader, questions, model, tokenizer, dataset_dict, model_name, args):
    # Run attention knockouts
    layers_to_cache = list(range(model.config.num_hidden_layers + 1))
    hs_cache_first_answer_gen_all = {}
    # for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor), line in tqdm(zip(data_loader, questions),total=len(questions)):
    #! video modality 받을 수 있도록
    for (input_ids, image_tensor, original_image_sizes, prompts, mask_tensor, modality), line in tqdm(zip(data_loader, questions),total=len(questions)):

        question_id = line["q_id"]

        #! 기존 image만 받음
        # img_id=line["img_id"]

        #! video도 받을 수 있도록 수정
        if "video" in line and line["video"] != "":
            img_id = str(line["video"])
        else:
            img_id = str(line["img_id"])


        input_ids = input_ids.to(device='cuda')
        image_tensor = [img_t.to(device='cuda') for img_t in image_tensor]

        # LLaVA v1.5/v1.6은 항상 "image"로 처리 (InformationFlow.py 참고)
        if "v1.6" in model_name.lower() or "v1.5" in model_name.lower():
            effective_modality = "image"
        else:
            effective_modality = modality

        inps = {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": original_image_sizes,
            "modalities": [effective_modality],
            "do_sample": True if args.temperature > 0 else False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
            "output_hidden_states":True

        }

        answer = dataset_dict[question_id]["answer"]


        hs_cache_first_answer_gen, predicted_answer = run_original(model, inps,tokenizer,model_name)

        #! 정답인지, 오답인지 확인
        is_correct = (answer.lower() == predicted_answer)

        hs_cache_first_answer_gen_all[question_id]={}

        #! is_correct & predicted answer 넣어서 파일 저장하기.
        hs_cache_first_answer_gen_all[question_id]["is_correct"] = is_correct
        hs_cache_first_answer_gen_all[question_id]["predicted_answer"] = predicted_answer

        #! 그냥 question_id, img_id 별로 layer별 last token hiddenstate 저장하는 거임
        for layer in layers_to_cache:
            hs_cache_first_answer_gen_all[question_id][(question_id, img_id, layer)]=hs_cache_first_answer_gen[layer]

    return hs_cache_first_answer_gen_all


# Information flow analysis
def main(args):

    cache_dir = os.environ.get("HF_HOME", None)

    # Model: model_args 스타일 또는 기존 --model-path 스타일
    if args.model_args:
        model_args_dict = parse_model_args(args.model_args)
        tokenizer, model, image_processor, context_len, model_name, conv_template = load_model_from_args(model_args_dict)
        args.conv_mode = conv_template
    else:
        tokenizer, model, image_processor, context_len, model_name, _ = load_model_legacy(args.model_path, args.model_base, args.conv_mode)

    model.eval()
    model.tie_weights()


    #dataset
    #predict correct and filter

    # Dataset: HuggingFace task 또는 CSV에서 로딩
    if args.task:
        task_name = args.task
        questions, dataset_dict = load_dataset_as_questions(
            task_name=args.task,
            video_folder=args.video_folder,
            image_folder=args.image_folder,
            hf_cache_dir=cache_dir,
            max_samples=args.max_samples,
        )
    elif args.refined_dataset:
        #! option argument로 MCQ 또는 일반론적으로 분기 처리
        if args.option == "MCQ":
            task_name = "MCQ"
        else:
            task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]

        # task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
        df = pd.read_csv(args.refined_dataset, dtype={"question_id":str}).fillna('')
        dataset_dict = df.set_index('question_id').T.to_dict('dict')
        questions = [ {**detail, "q_id":qu_id} for qu_id, detail in dataset_dict.items()]
    else:
        raise ValueError("--task (HuggingFace) 또는 --refined_dataset (CSV) 중 하나는 필수")

    # data_loader = create_data_loader(questions, args.image_folder,  args.batch_size, args.num_workers, tokenizer,  image_processor, model.config, task_name, args.conv_mode)
    #! video load 가능하도록 수정
    data_loader = create_data_loader(questions, args.image_folder, args.batch_size, args.num_workers,
                                  tokenizer, image_processor, model.config, task_name, args.conv_mode,
                                  video_folder=args.video_folder, video_fps=args.video_fps,
                                  frames_upbound=args.frames_upbound, force_sample=args.force_sample)



    if args.only_read_cache:
        file_name = f"cache_hiddenFeature"
        cache_path=f"output/temp/last_position_answer_probs/{model_name}/{task_name}/val/{file_name}.npy"
        print(f"read files form here: {cache_path}", flush=True)
        hs_cache_first_answer_gen_all = np.load(cache_path, allow_pickle=True).tolist()
    else :
        #cashe hidden state
        hs_cache_first_answer_gen_all = cache_hiddenstate(data_loader, questions, model, tokenizer, dataset_dict, model_name, args)
        if args.only_cache:
            file_name = "cache_hiddenFeature"
            os.makedirs(f"output/temp/last_position_answer_probs/{model_name}/{task_name}/val", exist_ok=True)
            np.save(f"output/temp/last_position_answer_probs/{model_name}/{task_name}/val/{file_name}.npy",hs_cache_first_answer_gen_all)
            exit(0)





    records = []

    #! lm_head weight 불러오기
    E = model.get_output_embeddings().weight.to(torch.float32).cpu().detach()
    for line in tqdm(questions,total=len(questions)):


        question_id = line["q_id"]
        #! 기존 image만 받음
        # img_id=line["img_id"]

        #! video도 받을 수 있도록 수정
        if "video" in line and line["video"] != "":
            img_id = str(line["video"])
        else:
            img_id = str(line["img_id"])

        if question_id not in hs_cache_first_answer_gen_all: continue

        #! 정답/오답 여부와 모델 예측 답변 가져오기
        is_correct = hs_cache_first_answer_gen_all[question_id]["is_correct"]
        predicted_answer = hs_cache_first_answer_gen_all[question_id]["predicted_answer"]

        question = dataset_dict[question_id]["question"]
        answer = dataset_dict[question_id]["answer"].lower()

        #! MCQ 추가
        if task_name in ("ChooseRel", "ChooseAttr", "ChooseCat"):
            true_option = dataset_dict[question_id]["true option"]
            false_option = dataset_dict[question_id]["false option"]
        elif task_name in ("MCQ"):
            true_option = answer
            false_option = dataset_dict[question_id]["false option"].lower()


        hs_cache_first_answer_gen_question=hs_cache_first_answer_gen_all[question_id]
        for layer in range(model.config.num_hidden_layers+1):
            hs_first_generated_token = hs_cache_first_answer_gen_question[(question_id, img_id, layer)].cpu().to(torch.float32)
            logits_first_generated_token = hs_first_generated_token.matmul(E.T)
            scores_first_generated_token = torch.softmax(logits_first_generated_token, dim=-1).numpy()

            top_k = [(tokenizer.decode([i]), i, scores_first_generated_token[i]) for i in np.argsort(-scores_first_generated_token)[:50]]
            top_k_word, top_k_token, top_k_score = zip(*top_k)

            #! is_correct & predicted_answer 추가
            temp_re={
                "question_id": question_id,
                "image": img_id,
                "goden answer": answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "question": question,
                "layer": layer,
                "top_k_word":top_k_word,
                "top_k_score":top_k_score,
            }

            if task_name == "ChooseRel" or task_name == "ChooseAttr" or task_name == "ChooseCat":
                true_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(true_option, add_special_tokens=False)[0]]
                false_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(false_option, add_special_tokens=False)[0]]

                true_option_InitialsUpperCase = true_option.capitalize()
                false_option_InitialsUpperCase = false_option.capitalize()

                true_InitialsUpperCase_score_first = scores_first_generated_token[
                    tokenizer.encode(true_option_InitialsUpperCase, add_special_tokens=False)[0]]
                false_InitialsUpperCase_score_first = scores_first_generated_token[
                    tokenizer.encode(false_option_InitialsUpperCase, add_special_tokens=False)[0]]
                temp_re.update({
                    "Noncapitalized Answer": true_LowerCase_score_first*100.0,
                    "Noncapitalized False Option": false_LowerCase_score_first*100.0,
                    "Capitalized Answer": true_InitialsUpperCase_score_first*100.0,
                    "Capitalized False Option": false_InitialsUpperCase_score_first*100.0,
                })
            elif task_name == "MCQ":
                #! 실제 옵션 텍스트를 컬럼명에 포함 (어떤 옵션인지 식별 가능)
                true_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(true_option, add_special_tokens=False)[0]]
                true_option_upper = true_option.capitalize()
                true_UpperCase_score_first = scores_first_generated_token[
                    tokenizer.encode(true_option_upper, add_special_tokens=False)[0]]
                temp_re.update({
                    f"Noncapitalized [{true_option}]": true_LowerCase_score_first * 100.0,
                    f"Capitalized [{true_option_upper}]": true_UpperCase_score_first * 100.0,
                })

                false_options = [fo.strip() for fo in false_option.split("|")]
                for fo in false_options:
                    fo_lower_score = scores_first_generated_token[tokenizer.encode(fo, add_special_tokens=False)[0]]
                    fo_upper = fo.capitalize()
                    fo_upper_score = scores_first_generated_token[tokenizer.encode(fo_upper, add_special_tokens=False)[0]]
                    temp_re.update({
                        f"Noncapitalized [{fo}]": fo_lower_score * 100.0,
                        f"Capitalized [{fo_upper}]": fo_upper_score * 100.0,
                    })
            else:
                #! answer 소문자 tracing
                answer_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(answer, add_special_tokens=False)[0]]
                answer_InitialsUpperCase = answer.capitalize()
                #! answer 대문자 tracing
                answer_InitialsUpperCase_score_first = scores_first_generated_token[tokenizer.encode(answer_InitialsUpperCase, add_special_tokens=False)[0]]
                temp_re.update({
                    "Noncapitalized Answer": answer_LowerCase_score_first*100.0,
                    "Capitalized Answer": answer_InitialsUpperCase_score_first*100.0,
                })


            records.append(temp_re)

    tmp = pd.DataFrame.from_records(records)

    tmp_correct = tmp[tmp["is_correct"] == True]
    tmp_incorrect = tmp[tmp["is_correct"] == False]

    save_name = ""
    model_name = model_name.replace('-', '_').replace('.', '_')
    os.makedirs(f"output/last_position_answer_probs/{model_name}/{task_name}/val/", exist_ok=True)

    #! 전체/정답/오답 CSV 각각 저장
    if args.refined_dataset:
        base_name = args.refined_dataset.split("/")[-1].split(".csv")[0]
    else:
        base_name = args.task if args.task else "unknown"
    tmp.to_csv(f'output/last_position_answer_probs/{model_name}/{task_name}/val/{base_name}{save_name}_all.csv', index=False)
    tmp_correct.to_csv(f'output/last_position_answer_probs/{model_name}/{task_name}/val/{base_name}{save_name}_correct.csv', index=False)
    tmp_incorrect.to_csv(f'output/last_position_answer_probs/{model_name}/{task_name}/val/{base_name}{save_name}_incorrect.csv', index=False)

    # Plot the results
    if task_name == "ChooseRel" or task_name == "ChooseAttr" or task_name == "ChooseCat":
        measures = [
            "Noncapitalized Answer",
            "Capitalized Answer",
            "Noncapitalized False Option",
            "Capitalized False Option"
        ]
    elif task_name == "MCQ":
        #! 전체 옵션 텍스트를 수집하여 measures 구성
        all_options = set()
        for line in questions:
            qid = line["q_id"]
            if qid not in dataset_dict:
                continue
            ans = dataset_dict[qid]["answer"].lower()
            all_options.add(ans)
            if dataset_dict[qid].get("false option", ""):
                for fo in dataset_dict[qid]["false option"].split("|"):
                    fo = fo.strip()
                    if fo:
                        all_options.add(fo)
        measures = []
        for opt in sorted(all_options):
            measures.append(f"Noncapitalized [{opt}]")
            measures.append(f"Capitalized [{opt.capitalize()}]")
    else:
        measures = [
            "Noncapitalized Answer",
            "Capitalized Answer",
        ]

    save_name += "_" + model_name

    #! 정답/오답 plot 각각 생성
    for label, df_sub in [("correct", tmp_correct), ("incorrect", tmp_incorrect), ("all", tmp)]:
        if len(df_sub) == 0:
            print(f"[WARN] No {label} samples, skipping plot.", flush=True)
            continue

        if task_name == "MCQ":
            #! MCQ: answer 클래스별 폴더 생성 후 correct/incorrect 저장
            for answer_class, df_class in df_sub.groupby("goden answer"):
                safe_class_name = answer_class.replace("/", "_").replace(" ", "_")
                class_dir = f'output/last_position_answer_probs/{model_name}/{task_name}/val/{safe_class_name}'
                os.makedirs(class_dir, exist_ok=True)
                save_path = f'{class_dir}/{base_name}{save_name}_{label}_first.pdf'
                generate_plot_attrscore(df_class, save_path, x="layer", ys=measures, layer_num=model.config.num_hidden_layers)
        else:
            save_path = f'output/last_position_answer_probs/{model_name}/{task_name}/val/{base_name}{save_name}_{label}_first.pdf'
            generate_plot_attrscore(df_sub, save_path, x="layer", ys=measures, layer_num=model.config.num_hidden_layers)










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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument('--refined_dataset', default=None, type=str, help="refined dataset")

    parser.add_argument("--only_read_cache",action='store_true', default=False)
    parser.add_argument("--only_cache",action='store_true', default=False)

    #! video 관련 인자 추가
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)

    #! MCQ option 인자 추가
    parser.add_argument("--option", type=str, default="standard")

    #! HuggingFace task support
    parser.add_argument('--task', type=str, default=None,
                        help=f"HuggingFace task name. Available: {list_tasks()}")
    parser.add_argument('--max_samples', type=int, default=-1,
                        help="Max samples limit (-1 for all). For debugging.")

    args = parser.parse_args()

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    main(args)
