import torch
from typing import List, Dict, Union
from tqdm import tqdm
from time import time
from peft import LoraConfig, TaskType
from EasyEdit.easyeditor.util import nethook
import random
from datasets import Dataset
from transformers import TrainingArguments, Trainer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model_state_dict, get_peft_model, set_peft_model_state_dict, LoraConfig, TaskType
from edit_utils import build_lora_training_args, preprocess_function_chat, create_lora_model, resolve_lora_training_config
from eval_utils import test_current_edited_knowledge

def cake_wise_return_lora_weights(original_model, tokenizer, item, target_modules, hparams, test_generation=False):
    config = resolve_lora_training_config(
        hparams,
        method_name="cake_wise",
        overrides={"target_modules": target_modules},
    )
    model = create_lora_model(
        original_model,
        r=config["rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
    )
    # original_model = original_model.to(device)
    model.enable_input_require_grads()
    train_examples = []
    item_case_examples = []
    learning_examples = []
    for rewrite in item['requested_rewrite']:
        prompt = rewrite['prompt'].format(rewrite['subject'])
        target = rewrite['target_new']['str']
        item_case_examples.append({
            "text": prompt,
            "target": target
        })
        if 'rephrase_prompt' in rewrite:
            for rewrite_item in rewrite['rephrase_prompt']:
                item_case_examples.append({
                    "text": rewrite_item['question'],
                    "target": rewrite_item['answer']
                })
        if 'learning_prompt' in rewrite:
            for learning_item in rewrite['learning_prompt']:
                learning_examples.append({
                    "text": learning_item['question'],
                    "target": learning_item['answer']
                })
    train_examples.append({'item_case_examples':item_case_examples,'learning_examples':learning_examples})
    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        preprocess_function_chat,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer,"model": model}
    )

    training_args = build_lora_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    start = time()
    trainer.train()
    exec_time = time() - start
    lora_weights = get_peft_model_state_dict(model)
    model = model.unload()
    return lora_weights, exec_time

def apply_lora_weights_to_model(base_model, lora_weights, target_modules, hparams=None):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=getattr(hparams, 'rank', 8),
        lora_alpha=getattr(hparams, 'lora_alpha', 32),
        lora_dropout=getattr(hparams, 'lora_dropout', 0.05),
        target_modules=target_modules
    )
    
    peft_model = get_peft_model(base_model, peft_config)
    set_peft_model_state_dict(peft_model, lora_weights)
    
    return peft_model

def apply_lora_wise_merge(base_model, lora_weights_list, target_modules, hparams):
    from EasyEdit.easyeditor.models.wise.WISE import merge_dict
    merge_alg = hparams.merge_alg
    merger = merge_dict[merge_alg]
    all_param_names = set()
    for lora_weights in lora_weights_list:
        all_param_names.update(lora_weights.keys())
    merged_lora_weights = {}
    for param_name in all_param_names:
        param_variants = [lw[param_name] for lw in lora_weights_list if param_name in lw]
        if len(param_variants) > 1:
            weights = [1.0/len(param_variants)] * len(param_variants)
            base_param = torch.zeros_like(param_variants[0]) 
            merged_param = merger.execute(weights, base_param, param_variants, hparams.densities)
            merged_lora_weights[param_name] = merged_param
        elif len(param_variants) == 1:
            merged_lora_weights[param_name] = param_variants[0]
    return apply_lora_weights_to_model(base_model, merged_lora_weights, target_modules, hparams)

class CakeWiseState:
    def __init__(self):
        self.editing_count = 0
        self.lora_memory = []
        self.base_model = None
        
    def reset(self):
        self.editing_count = 0
        self.lora_memory = []
        self.base_model = None

cake_wise_state = CakeWiseState()
 
def cake_wise_sequential_edit(model, tokenizer, items_list, hparams, edit_freq, datatype,test_generation=False):
    current_model = model
    target_modules = resolve_lora_training_config(hparams, method_name="cake_wise")["target_modules"]
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_lora_weights = [] 
    print("Starting CAKE_WISE sequential-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        lora_weights, exec_time = cake_wise_return_lora_weights(current_model, tokenizer, item, target_modules, hparams, test_generation)
        accumulated_lora_weights.append(lora_weights)
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            print(f"Merging {len(accumulated_lora_weights)} LoRA weights...")
            current_model = apply_lora_wise_merge(current_model, accumulated_lora_weights, target_modules, hparams)
            current_model = current_model.merge_and_unload()
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, datatype,test_generation)
            all_metrics.extend(test_metrics)
            accumulated_lora_weights = []
            current_training_times = []
            edited_items = []
    print("CAKE_WISE sequential-edit completed.")
    return current_model, all_metrics

def cake_wise_continual_edit(model, tokenizer, items_list, hparams, edit_freq, datatype,test_generation=False):
    current_model = model
    target_modules = resolve_lora_training_config(hparams, method_name="cake_wise")["target_modules"]
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_lora_weights = [] 
    print("Starting CAKE_WISE continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        lora_weights, exec_time = cake_wise_return_lora_weights(current_model, tokenizer, item, target_modules, hparams, test_generation)
        accumulated_lora_weights.append(lora_weights)
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0:
            print(f"Merging {len(accumulated_lora_weights)} LoRA weights...")
            current_model = apply_lora_wise_merge(current_model, accumulated_lora_weights, target_modules, hparams)
            current_model = current_model.merge_and_unload()
            accumulated_lora_weights = []
        if (i + 1) == len(items_list):
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times,datatype, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print("CAKE_WISE continual-edit completed.")
    return current_model, all_metrics

def cake_wise_multi_edit(model, tokenizer, items_list, hparams, edit_freq, MODEL_PATH, datatype,test_generation=False):
    current_model = model
    target_modules = resolve_lora_training_config(hparams, method_name="cake_wise")["target_modules"]
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_lora_weights = [] 
    print("Starting CAKE_WISE multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        lora_weights, exec_time = cake_wise_return_lora_weights(current_model, tokenizer, item, target_modules, hparams, test_generation)
        accumulated_lora_weights.append(lora_weights)
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            print(f"Merging {len(accumulated_lora_weights)} LoRA weights...")
            current_model = apply_lora_wise_merge(current_model, accumulated_lora_weights, target_modules, hparams)
            current_model = current_model.merge_and_unload()
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times,datatype, test_generation)
            all_metrics.extend(test_metrics)
            accumulated_lora_weights = []
            current_training_times = []
            edited_items = []
            current_model = None
            del current_model
            current_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={"": "cuda:0"},torch_dtype=torch.bfloat16)
    print("CAKE_WISE multi-edit completed.")
    return current_model, all_metrics
