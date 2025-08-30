import torch
import torch.nn.functional as F
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
from edit_utils import  preprocess_function_chat, create_lora_model
from eval_utils import test_current_edited_knowledge, compute_edit_quality

class CakeKLTrainer(Trainer):
    def __init__(self, original_model_ref, kl_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model_ref = original_model_ref
        self.kl_lambda = kl_lambda
        self._step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        edit_loss = outputs.loss
        
        with torch.no_grad():
            original_outputs = self.original_model_ref(**inputs)
            original_logits = original_outputs.logits
        
        # KL(P_edited || P_original)
        kl_loss = F.kl_div(
            F.log_softmax(outputs.logits, dim=-1),
            F.softmax(original_logits.detach(), dim=-1),
            reduction='batchmean'
        )
        total_loss = edit_loss + self.kl_lambda * kl_loss
        self._step_count += 1
        if self._step_count % 10 == 0:
            print(f"Step {self._step_count}: Edit={edit_loss:.4f}, KL={kl_loss:.4f}, Total={total_loss:.4f}")
        
        return (total_loss, outputs) if return_outputs else total_loss

def cake_batch_kl_return_lora_weights(original_model, tokenizer, items_list, hparams, test_generation=False):
    # target_modules = ["q_proj", "v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"] 
    target_modules = ["up_proj","down_proj"]
    model = create_lora_model(original_model,target_modules=target_modules)
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
        fn_kwargs={"tokenizer": tokenizer}
    )

    training_args = TrainingArguments(
            output_dir=f'./output/',
            overwrite_output_dir=True,
            num_train_epochs=30,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            save_strategy="no",
            bf16=True,
            logging_steps=10,
            report_to="none",
        )
    # kl_lambda = getattr(hparams, 'kl_lambda', 0.05)
    # kl_lambda = getattr(hparams, 'kl_lambda', 0.1)
    kl_lambda = getattr(hparams, 'kl_lambda', 0.2)
    trainer = CakeKLTrainer(
        original_model_ref=original_model,
        kl_lambda=kl_lambda,
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

def apply_lora_weights_to_model(base_model, lora_weights, hparams=None):
    # target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    target_modules = ["up_proj","down_proj"]
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
 
def cake_batch_kl_sequential_edit(model, tokenizer, items_list, hparams, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_items = [] 
    print("Starting CAKE_Batch_KL sequential-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        accumulated_items.append(item)
        edited_items.append(item) 
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
            lora_weights, exec_time = cake_batch_kl_return_lora_weights(current_model, tokenizer, accumulated_items, hparams, test_generation)
            current_training_times.append(exec_time)
            current_model = apply_lora_weights_to_model(current_model, lora_weights, hparams)
            current_model = current_model.merge_and_unload()
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            accumulated_items = []
            current_training_times = []
            edited_items = []
    print("CAKE_Batch_KL sequential-edit completed.")
    return current_model, all_metrics

def cake_batch_kl_continual_edit(model, tokenizer, items_list, hparams, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_items = [] 
    print("Starting CAKE_Batch_KL continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        accumulated_items.append(item)
        edited_items.append(item) 
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        if (i+1) % edit_freq == 0:
            print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
            lora_weights, exec_time = cake_batch_kl_return_lora_weights(current_model, tokenizer, accumulated_items, hparams, test_generation)
            current_training_times.append(exec_time)
            current_model = apply_lora_weights_to_model(current_model, lora_weights, hparams)
            current_model = current_model.merge_and_unload()
            accumulated_items = []
        if (i + 1) == len(items_list):
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print("CAKE_Batch_KL continual-edit completed.")
    return current_model, all_metrics

def cake_batch_kl_multi_edit(model, tokenizer, items_list, hparams, edit_freq, MODEL_PATH, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    accumulated_items = [] 
    print("Starting CAKE_Batch_KL multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        accumulated_items.append(item)
        edited_items.append(item) 
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
            lora_weights, exec_time = cake_batch_kl_return_lora_weights(current_model, tokenizer, accumulated_items, hparams, test_generation)
            current_training_times.append(exec_time)
            current_model = apply_lora_weights_to_model(current_model, lora_weights, hparams)
            current_model = current_model.merge_and_unload()
            print(f"Testing knowledge retention after {i+1} edits...")
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            accumulated_items = []
            current_training_times = []
            edited_items = []
            current_model = None
            del current_model
            current_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={"": "cuda:0"},torch_dtype=torch.bfloat16)
    print("CAKE_Batch_KL multi-edit completed.")
    return current_model, all_metrics
