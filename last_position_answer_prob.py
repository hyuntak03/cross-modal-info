from InformationFlow import create_data_loader

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

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns

import argparse
import os
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path




def generate_plot_attrscore(data, save_file, x="layer", ys="", layer_num=0):
    
    #! measures 개수에 맞게 palette 동적 생성
    if len(ys) <= 4:
        hex_colors = ["#f20089", "#5c95ff", "#ffa9a3", "#b9e6ff"]
    else:
        hex_colors = sns.color_palette("husl", len(ys)).as_hex()

    palette = sns.color_palette(hex_colors)

    sns.set(context="notebook")
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(4, 4))

    ax = sns.lineplot(data, x=x, y=ys[0],
                      label=ys[0],color=palette[0],
                      dashes=False,
                      linewidth=3)

    for ind, y in enumerate(ys[1:]):
        sns.lineplot(data, x=x, y=y,
                     label=y,color=palette[ind+1],
                     dashes=False,
                     linewidth=3)


    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, layer_num + 0.5)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.legend(fontsize=6,handlelength=1)

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




def cache_hiddenstate(data_loader, questions, model, tokenizer, dataset_dict, model_name):
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
        is_correct = (answer == predicted_answer)

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

    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
    #                                                                       args.model_base,
    #                                                                       model_name,
    #                                                                       device_map="auto",
    #                                                                       attn_implementation=None)
    
    #! cache dir 주기 위해 수정
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device_map="auto", attn_implementation=None, cache_dir=cache_dir
    )

    model.eval()
    model.tie_weights()


    #dataset
    #predict correct and filter

    #! option argument로 MCQ 또는 일반론적으로 분기 처리
    if args.option == "MCQ":
        task_name = "MCQ"
    else:
        task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]

    # task_name = args.refined_dataset.split("/")[-1].split(".csv")[0].split("_")[-1]
    df = pd.read_csv(args.refined_dataset, dtype={"question_id":str}).fillna('')
    dataset_dict = df.set_index('question_id').T.to_dict('dict')
    questions = [ {**detail, "q_id":qu_id} for qu_id, detail in dataset_dict.items()]
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
        hs_cache_first_answer_gen_all = cache_hiddenstate(data_loader, questions, model, tokenizer, dataset_dict, model_name)
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
        if task_name in ("ChooseRel", "ChooseAttr", "ChooseCat", "MCQ"):
            true_option = dataset_dict[question_id]["true option"]
            false_option = dataset_dict[question_id]["false option"]


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
                true_LowerCase_score_first = scores_first_generated_token[tokenizer.encode(true_option, add_special_tokens=False)[0]]
                true_option_InitialsUpperCase = true_option.capitalize()
                true_InitialsUpperCase_score_first = scores_first_generated_token[
                    tokenizer.encode(true_option_InitialsUpperCase, add_special_tokens=False)[0]]
                temp_re.update({
                    "Noncapitalized Answer": true_LowerCase_score_first*100.0,
                    "Capitalized Answer": true_InitialsUpperCase_score_first*100.0,
                })

                #! false option 여러 개를 | 구분자로 split하여 각각 처리
                false_options = [fo.strip() for fo in false_option.split("|")]
                for fi, fo in enumerate(false_options):
                    fo_lower_score = scores_first_generated_token[tokenizer.encode(fo, add_special_tokens=False)[0]]
                    fo_upper = fo.capitalize()
                    fo_upper_score = scores_first_generated_token[tokenizer.encode(fo_upper, add_special_tokens=False)[0]]
                    temp_re.update({
                        f"Noncapitalized False Option {fi}": fo_lower_score * 100.0,
                        f"Capitalized False Option {fi}": fo_upper_score * 100.0,
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
    base_name = args.refined_dataset.split("/")[-1].split(".csv")[0]
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
        measures = [
            "Noncapitalized Answer",
            "Capitalized Answer",
        ]
        #! false option 컬럼들을 동적으로 추가
        false_cols = [c for c in tmp.columns if c.startswith("Noncapitalized False Option") or c.startswith("Capitalized False Option")]
        measures.extend(sorted(false_cols))
    
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
        save_path = f'output/last_position_answer_probs/{model_name}/{task_name}/val/{base_name}{save_name}_{label}_first.pdf'
        generate_plot_attrscore(df_sub, save_path, x="layer", ys=measures, layer_num=model.config.num_hidden_layers)










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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument('--refined_dataset', default="", type=str, help="refined dataset")

    parser.add_argument("--only_read_cache",action='store_true', default=False)
    parser.add_argument("--only_cache",action='store_true', default=False)

    #! video 관련 인자 추가
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", action="store_true", default=False)

    #! MCQ option 인자 추가
    parser.add_argument("--option", type=str, default="standard")

    args = parser.parse_args()

    print("-------------------args-------------------")
    print(args)
    print("------------------------------------------")

    main(args)




