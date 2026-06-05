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
from eval_utils import test_current_edited_knowledge, compute_edit_quality

def cake_batch_return_lora_weights(original_model, tokenizer, items_list, target_modules, hparams, test_generation=False):
    config = resolve_lora_training_config(
        hparams,
        method_name="cake_batch",
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
    for item in items_list:
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
 
def cake_batch_sequential_edit(model, tokenizer, items_list, hparams, edit_freq,datatype, test_generation=False):
    current_model = model
    target_modules = resolve_lora_training_config(hparams, method_name="cake_batch")["target_modules"]
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_items = [] 
    print("Starting CAKE_Batch sequential-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        accumulated_items.append(item)
        edited_items.append(item) 
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
            lora_weights, exec_time = cake_batch_return_lora_weights(current_model, tokenizer, accumulated_items, target_modules, hparams, test_generation)
            current_training_times.append(exec_time)
            current_model = apply_lora_weights_to_model(current_model, lora_weights, target_modules, hparams)
            current_model = current_model.merge_and_unload()
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, datatype,test_generation)
            all_metrics.extend(test_metrics)
            accumulated_items = []
            current_training_times = []
            edited_items = []
    print("CAKE_Batch sequential-edit completed.")
    return current_model, all_metrics

def cake_batch_continual_edit(model, tokenizer, items_list, hparams, edit_freq,datatype, test_generation=False):
    current_model = model
    target_modules = resolve_lora_training_config(hparams, method_name="cake_batch")["target_modules"]
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_items = [] 
    print("Starting CAKE_Batch continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        accumulated_items.append(item)
        edited_items.append(item) 
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        if (i+1) % edit_freq == 0:
            print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
            lora_weights, exec_time = cake_batch_return_lora_weights(current_model, tokenizer, accumulated_items, target_modules, hparams, test_generation)
            current_training_times.append(exec_time)
            current_model = apply_lora_weights_to_model(current_model, lora_weights, target_modules, hparams)
            current_model = current_model.merge_and_unload()
            accumulated_items = []
        if (i + 1) == len(items_list):
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, datatype,test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print("CAKE_Batch continual-edit completed.")
    return current_model, all_metrics

def cake_batch_multi_edit(model, tokenizer, items_list, hparams, edit_freq, MODEL_PATH, datatype,test_generation=False):
    current_model = model
    target_modules = resolve_lora_training_config(hparams, method_name="cake_batch")["target_modules"]
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_items = [] 
    print("Starting CAKE_Batch multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        accumulated_items.append(item)
        edited_items.append(item) 
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
            lora_weights, exec_time = cake_batch_return_lora_weights(current_model, tokenizer, accumulated_items, target_modules, hparams, test_generation)
            current_training_times.append(exec_time)
            current_model = apply_lora_weights_to_model(current_model, lora_weights, target_modules, hparams)
            current_model = current_model.merge_and_unload()
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, datatype,test_generation)
            all_metrics.extend(test_metrics)
            accumulated_items = []
            current_training_times = []
            edited_items = []
            current_model = None
            del current_model
            current_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={"": "cuda:0"},torch_dtype=torch.bfloat16)
    print("CAKE_Batch multi-edit completed.")
    return current_model, all_metrics
