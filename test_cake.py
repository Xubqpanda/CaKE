from EasyEdit.easyeditor import MEMITHyperParams,LoRAHyperParams,WISEHyperParams, ROMEHyperParams, CAKEWISEHyperParams, AlphaEditHyperParams
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from edit_utils import edit, edit_ifmet, edit_mello, edit_rome, cake, edit_wise, get_sent_embeddings
from eval_utils import calculate_averages
from Edit_mode.sequential_edit import cake_sequential_edit, sequential_edit_rome, sequential_edit_wise,sequential_edit
from Edit_mode.multi_edit import multi_edit_rome, multi_edit, cake_multi_edit, multi_edit_wise
from Edit_mode.continual_edit import continual_edit_rome, continual_edit, cake_continual_edit, continual_edit_wise
from CaKE_method.cake_wise import cake_wise_sequential_edit, cake_wise_multi_edit, cake_wise_continual_edit
from CaKE_method.cake_kl import cake_kl_sequential_edit, cake_kl_multi_edit, cake_kl_continual_edit, cake_kl_single_edit
from CaKE_method.cake_batch import cake_batch_sequential_edit, cake_batch_multi_edit, cake_batch_continual_edit
from CaKE_method.cake_batch_kl import cake_batch_kl_sequential_edit, cake_batch_kl_multi_edit, cake_batch_kl_continual_edit
from CaKE_method.cake_Overtone import cake_overtone
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import os
import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None,type=str)
    parser.add_argument('--model_type', default=None,type=str)
    parser.add_argument('--edit_mode', default='single_edit', choices=['single_edit', 'multi_edit','sequential_edit', 'continual_edit'], type=str)
    parser.add_argument('--edit_freq', default=10, type=int)
    args = parser.parse_args()

    if args.editing_method == 'MEMIT' or args.editing_method == 'IFMET':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'CAKE_WISE':
        editing_hparams = CAKEWISEHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    else:
        editing_hparams = LoRAHyperParams
    if args.editing_method == 'CAKE' or args.editing_method == 'Mello' or args.editing_method == 'CAKE_KL' or args.editing_method == 'CAKE_Batch' or args.editing_method == 'CAKE_OverTone' or args.editing_method == 'CAKE_Batch_KL':
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/LoRA/{args.model_type}.yaml')
    elif args.editing_method == 'IFMET':
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-shallow.yaml')
        hparams_s=hparams
        hparams_d=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}-deeper.yaml')
    else:
        hparams=editing_hparams.from_hparams(f'./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml')
    
    if args.editing_method != 'Mello' and args.editing_method != 'CAKE_WISE':
        alg_name = hparams.alg_name
        apply_algo = ALG_DICT[alg_name]

    MODEL_PATH = hparams.model_name
    if args.editing_method == 'CAKE' or args.editing_method == 'CAKE_WISE' \
        or args.editing_method == 'CAKE_KL' or args.editing_method == 'CAKE_Batch' \
        or args.editing_method == 'CAKE_OverTone' or args.editing_method == 'CAKE_Batch_KL':
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",torch_dtype=torch.float32)
    # 添加FFN层的LoRA配置
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    import json
    data = json.load(open(f'./datasets/{args.datatype}-cake.json','r'))
    datatype = args.datatype
    if args.editing_method == 'WISE':
        loc_data = json.load(open('./datasets/ZsRE/zsre_mend_train.json','r'))
        loc_data = loc_data[:7000]
        loc_index = 0
    elif args.editing_method == 'Mello':
        contriever = AutoModel.from_pretrained("/data1/xubuqiang/contriever-msmarco", device_map=f"cuda:{hparams.device}")
        contriever_tokenizer = AutoTokenizer.from_pretrained("/data1/xubuqiang/contriever-msmarco")
        with open('./prompts/MeLLo-prompt.txt', 'r') as f:
            task_prompt = f.read()
        stop = ["Retrieved fact:"]
        new_facts = set()
        for d in data:
            for r in d["requested_rewrite"]:
                new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
        new_facts = list(new_facts)
        embs = get_sent_embeddings(new_facts, contriever, contriever_tokenizer, hparams.device)
    
    if args.edit_mode == 'single_edit':
        print("Evaluating single-edit retention...") 
        all_metrics = []
        for item in tqdm(data[:100]):
            if args.editing_method == 'CAKE':
                model, metrics = cake(model, tokenizer, item, hparams, datatype, test_generation=False)
            elif args.editing_method == 'CAKE_KL':
                model, metrics = cake_kl_single_edit(model, tokenizer, item, hparams, MODEL_PATH, datatype, test_generation=False)
            elif args.editing_method == 'CAKE_OverTone':
                model, metrics = cake_overtone(model, tokenizer, item, hparams, datatype, test_generation=False)
            elif args.editing_method == 'WISE':
                metrics, loc_index = edit_wise(model, tokenizer, item, hparams, loc_data, loc_index, apply_algo, datatype, test_generation=False)
            elif args.editing_method == 'Mello':
                metrics = edit_mello(model, task_prompt, stop, tokenizer, item, hparams, contriever, contriever_tokenizer, embs, new_facts, datatype, test_generation=False)
            elif args.editing_method == 'IFMET':
                model, metrics = edit_ifmet(model, tokenizer, item, hparams_s, hparams_d, apply_algo, datatype, test_generation=False)
            elif args.editing_method == 'ROME':
                model, metrics = edit_rome(model, tokenizer, item, hparams, apply_algo,datatype, test_generation=False)
            else:
                model, metrics = edit(model, tokenizer, item, hparams, alg_name, apply_algo, datatype, test_generation=False)
            print(metrics)
            all_metrics.append(metrics)
        res = calculate_averages(all_metrics)
        os.makedirs(args.metrics_save_dir,exist_ok=True)
        json.dump(all_metrics, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_metrics_2025_9_3_kl_0_0_5.json','w'),indent=4)
        json.dump(res, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_res_2025_9_3_kl_0_0_5.json','w'),indent=4)

    
    elif args.edit_mode == 'sequential_edit':
        print("Evaluating sequential-edit retention...")    
        edit_freq = args.edit_freq
        sequential_edit_items = data[:100]
        sequential_edit_metrics = []
        if args.editing_method == 'CAKE':
            model, sequential_edit_metrics = cake_sequential_edit(model, tokenizer, sequential_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'ROME':
            model, sequential_edit_metrics = sequential_edit_rome(model, tokenizer, sequential_edit_items, hparams, apply_algo, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_WISE':
            model, sequential_edit_metrics = cake_wise_sequential_edit(model, tokenizer, sequential_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_KL':
            model, sequential_edit_metrics = cake_kl_sequential_edit(model, tokenizer, sequential_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_Batch':
            model, sequential_edit_metrics = cake_batch_sequential_edit(model, tokenizer, sequential_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_Batch_KL':
            model, sequential_edit_metrics = cake_batch_kl_sequential_edit(model, tokenizer, sequential_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'WISE':
            metrics, sequential_edit_metrics = sequential_edit_wise(model, tokenizer, sequential_edit_items, hparams, loc_data, loc_index, apply_algo, edit_freq, datatype, test_generation=False) 
        else:
            model, sequential_edit_metrics = sequential_edit(model, tokenizer, sequential_edit_items, hparams, alg_name, apply_algo, edit_freq, datatype, test_generation=False)
        # TODO
        # elif args.editing_method == 'Mello':
        #     metrics = sequential_edit_mello(model, task_prompt, stop, tokenizer, sequential_edit_items, hparams, contriever, contriever_tokenizer, embs, datatype, test_generation=False)
        # elif args.editing_method == 'IFMET':
        #     model, metrics = sequential_edit_ifmet(model, tokenizer, sequential_edit_items, hparams_s, hparams_d, apply_algo, datatype, test_generation=False)
        res_retention = calculate_averages(sequential_edit_metrics)
        os.makedirs(args.metrics_save_dir,exist_ok=True)
        json.dump(sequential_edit_metrics, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}_metrics_2025_9_3.json','w'),indent=4)
        json.dump(res_retention, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}_res_2025_9_3.json','w'),indent=4)

    elif args.edit_mode == 'continual_edit':
        print("Evaluating continual-edit retention...")    
        edit_freq = args.edit_freq
        continual_edit_items = data[:100]
        continual_edit_metrics = []
        if args.editing_method == 'CAKE':
            model, continual_edit_metrics = cake_continual_edit(model, tokenizer, continual_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'ROME':
            model, continual_edit_metrics = continual_edit_rome(model, tokenizer, continual_edit_items, hparams, apply_algo, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_WISE':
            model, continual_edit_metrics = cake_wise_continual_edit(model, tokenizer, continual_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_KL':
            model, continual_edit_metrics = cake_kl_continual_edit(model, tokenizer, continual_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_Batch':
            model, continual_edit_metrics = cake_batch_continual_edit(model, tokenizer, continual_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_Batch_KL':
            model, continual_edit_metrics = cake_batch_kl_continual_edit(model, tokenizer, continual_edit_items, hparams, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'WISE':
            metrics, continual_edit_metrics = continual_edit_wise(model, tokenizer, continual_edit_items, hparams, loc_data, loc_index, apply_algo, edit_freq, datatype, test_generation=False) 
        else:
            model, continual_edit_metrics = continual_edit(model, tokenizer, continual_edit_items, hparams, alg_name, apply_algo, edit_freq, datatype, test_generation=False)
        # TODO
        # elif args.editing_method == 'Mello':
        #     metrics = continual_edit_mello(model, task_prompt, stop, tokenizer, continual_edit_items, hparams, contriever, contriever_tokenizer, embs, datatype, test_generation=False)
        # elif args.editing_method == 'IFMET':
        #     model, metrics = continual_edit_ifmet(model, tokenizer, continual_edit_items, hparams_s, hparams_d, apply_algo, datatype, test_generation=False)
        res_retention = calculate_averages(continual_edit_metrics)
        os.makedirs(args.metrics_save_dir,exist_ok=True)
        json.dump(continual_edit_metrics, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}_metrics_2025_9_3_kl_0_0_5.json','w'),indent=4)
        json.dump(res_retention, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}_res_2025_9_3_kl_0_0_5.json','w'),indent=4)

    
    elif args.edit_mode == 'multi_edit':
        print("Evaluating multi-edit retention...")    
        edit_freq = args.edit_freq
        multi_edit_items = data[:100]
        multi_edit_metrics = []
        
        if args.editing_method == 'ROME':
            model, multi_edit_metrics = multi_edit_rome(model, tokenizer, multi_edit_items, hparams, apply_algo, edit_freq, datatype, test_generation=False)
        elif args.editing_method == 'CAKE':
            model, multi_edit_metrics = cake_multi_edit(model, tokenizer, multi_edit_items, hparams, edit_freq, MODEL_PATH, datatype, test_generation=False) 
        elif args.editing_method == 'WISE':
            metrics, multi_edit_metrics = multi_edit_wise(model, tokenizer, multi_edit_items, hparams, loc_data, loc_index, apply_algo, edit_freq, datatype, test_generation=False) 
        elif args.editing_method == 'CAKE_WISE':
            model, multi_edit_metrics = cake_wise_multi_edit(model, tokenizer, multi_edit_items, hparams, edit_freq, MODEL_PATH, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_KL':
            model, multi_edit_metrics = cake_kl_multi_edit(model, tokenizer, multi_edit_items, hparams, edit_freq, MODEL_PATH, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_Batch_KL':
            model, multi_edit_metrics = cake_batch_kl_multi_edit(model, tokenizer, multi_edit_items, hparams, edit_freq, MODEL_PATH, datatype, test_generation=False)
        elif args.editing_method == 'CAKE_Batch':
            model, multi_edit_metrics = cake_batch_multi_edit(model, tokenizer, multi_edit_items, hparams, edit_freq, MODEL_PATH, datatype, test_generation=False)
        else:
            model, multi_edit_metrics = multi_edit(model, tokenizer, multi_edit_items, hparams, alg_name, apply_algo, edit_freq, datatype, test_generation=False)
        # TODO
        # elif args.editing_method == 'Mello':
        #     metrics = multi_edit_mello(model, task_prompt, stop, tokenizer, multi_edit_items, hparams, contriever, contriever_tokenizer, embs, datatype, test_generation=False)
        # elif args.editing_method == 'IFMET':
        #     model, metrics = multi_edit_ifmet(model, tokenizer, multi_edit_items, hparams_s, hparams_d, apply_algo, datatype, test_generation=False)
        res_retention = calculate_averages(multi_edit_metrics)
        os.makedirs(args.metrics_save_dir,exist_ok=True)
        json.dump(multi_edit_metrics, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}_metrics_2025_9_3.json','w'),indent=4)
        json.dump(res_retention, open(f'{args.metrics_save_dir}/{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}_res_2025_9_3.json','w'),indent=4)
