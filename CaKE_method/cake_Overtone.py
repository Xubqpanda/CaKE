import torch
import torch.nn.functional as F
from typing import List, Dict, Union
from tqdm import tqdm
from time import time
from edit_utils import create_lora_model, preprocess_function_chat, compute_edit_quality
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from collections import Counter
import numpy as np

class OvertoneTrainer(Trainer):
    def __init__(self, 
                 smooth_factor: float = 0.1,
                 n_sigma: float = 1.0, 
                 epsilon: float = 5e-2,
                 debug_mode: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.smooth_factor = smooth_factor
        self.n_sigma = n_sigma
        self.epsilon = epsilon
        self.do_label_smoothing = smooth_factor < 1.0
        self.debug_mode = debug_mode
        self.step_count = 0
        
        print(f"OVERTONE parameters:")
        print(f"   - smooth_factor (λ): {self.smooth_factor}")
        print(f"   - n_sigma: {self.n_sigma}")
        print(f"   - epsilon: {self.epsilon}")
        print(f"   - do_label_smoothing: {self.do_label_smoothing}")
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.step_count += 1
        
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if labels is None:
            if return_outputs:
                return (None, outputs)
            return None
        # Shift for next token prediction  
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = (shift_labels != -100)
        
        if self.debug_mode and self.step_count <= 3: 
            print(f"\n🔍 Step {self.step_count} - OVERTONE Debug:")
            print(f"   - Batch size: {shift_labels.shape[0]}")
            print(f"   - Sequence length: {shift_labels.shape[1]}")
            print(f"   - Total tokens: {shift_labels.numel()}")
            print(f"   - Valid tokens (target): {shift_mask.sum().item()}")
            print(f"   - Masked tokens (-100): {(~shift_mask).sum().item()}")
            
            if shift_labels.shape[0] > 0:
                first_sample_labels = shift_labels[0]
                first_sample_mask = shift_mask[0]
                valid_indices = torch.where(first_sample_mask)[0]
                
                if len(valid_indices) > 0:
                    print(f"   - first sample valid tokens: {valid_indices[:10].tolist()}...")
                    print(f"   - first sample valid tokens values: {first_sample_labels[valid_indices][:10].tolist()}...")
                    valid_tokens = first_sample_labels[first_sample_mask]
                    decoded = self.tokenizer.decode(valid_tokens.cpu().numpy(), skip_special_tokens=True)
                    print(f"   - decoded target: '{decoded}'")
                 
        if not shift_mask.any():
            print("no valid tokens to calculate loss")
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            if return_outputs:
                return (loss, outputs)
            return loss
        
        vocab_size = shift_logits.size(-1)
        smooth_labels = torch.zeros_like(shift_logits).to(shift_logits.device)
        
        valid_positions = shift_mask.unsqueeze(-1)
        shift_labels_masked = shift_labels.clone()
        shift_labels_masked[~shift_mask] = 0
        smooth_labels.scatter_(2, shift_labels_masked.unsqueeze(-1), 1.0)
        smooth_labels = smooth_labels * valid_positions.float()
        
        if self.do_label_smoothing:
            with torch.no_grad():
                pred_probs = torch.softmax(shift_logits, dim=-1)
                
                if self.n_sigma > 0:
                    threshold = shift_logits.max(dim=-1, keepdim=True).values - \
                               self.n_sigma * shift_logits.std(dim=-1, keepdim=True)
                    noise_mask = (shift_logits >= threshold)
                    filtered_logits = shift_logits.masked_fill(~noise_mask, float('-inf'))
                    pred_probs = torch.softmax(filtered_logits, dim=-1)
                
                candidate_mix = self.smooth_factor * smooth_labels + \
                               (1 - self.smooth_factor) * pred_probs
                predicted_tokens = candidate_mix.argmax(dim=-1)
                
                correct_prediction = ((predicted_tokens == shift_labels) | ~shift_mask).unsqueeze(-1)
                
                adaptive_factors = self.smooth_factor * correct_prediction.float() + \
                                  (1 - correct_prediction.float())
                smooth_labels = (adaptive_factors * smooth_labels + 
                                (1 - adaptive_factors) * pred_probs).detach()
                
                pred_labels = pred_probs.argmax(dim=-1)
                entropy_correct_pred = ((pred_labels == shift_labels) | ~shift_mask).unsqueeze(-1)
                entropy = -(smooth_labels * torch.log(smooth_labels + 1e-8)).sum(dim=-1)
                entropy[~entropy_correct_pred.squeeze(-1)] = float("-inf")
                
                log_probs_for_ce = torch.log_softmax(shift_logits, dim=-1)
                log_probs_for_ce = log_probs_for_ce.masked_fill(log_probs_for_ce == float('-inf'), 0.0)
                ce_loss = -(smooth_labels * log_probs_for_ce).sum(dim=-1)
                loss_mask = shift_mask.float() * (ce_loss >= entropy + self.epsilon).float()
                
                if self.debug_mode and self.step_count <= 3:
                    total_valid = shift_mask.sum().item()
                    early_stopped = (shift_mask.float() - loss_mask).sum().item()
                    still_training = loss_mask.sum().item()
                    
                    print(f"   OVERTONE stastics:")
                    print(f"      - total valid tokens: {total_valid}")
                    print(f"      - early stopped tokens: {early_stopped}")
                    print(f"      - stll training tokens: {still_training}")
                    print(f"      - early topped/total valid: {early_stopped/total_valid*100:.1f}%")
                    
                    if total_valid > 0:
                        valid_entropy = entropy[shift_mask]
                        valid_ce = ce_loss[shift_mask]
                        print(f"      - valid tokens' entropy: {valid_entropy.tolist()}")
                        print(f"      - valid tokens' CE: {valid_ce.tolist()}")
        else:
            loss_mask = shift_mask.float()
        
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        log_probs = log_probs.masked_fill(log_probs == float('-inf'), 0.0)
        sample_loss = -(smooth_labels * log_probs).sum(dim=-1)
        masked_loss = sample_loss * loss_mask
        
        if loss_mask.sum() > 0:
            loss = masked_loss[loss_mask.bool()].mean()
        else:
            loss = masked_loss[shift_mask].mean()
            if self.debug_mode:
                print("all tokens all early stoped, using standard mask to calculate loss")

        if self.debug_mode and self.step_count <= 3:
            print(f"   - final loss: {loss.item():.4f}")
        
        if return_outputs:
            return (loss, outputs)
        return loss
    
    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        formatted_logs = {}
        for key, value in logs.items():
            if key == "train_loss":
                formatted_logs[f"OVERTONE_{key}"] = value
            else:
                formatted_logs[key] = value
        
        super().log(formatted_logs)

def cake_overtone(original_model, tokenizer, item, hparams, datatype,test_generation=False):
    target_modules = ["q_proj", "v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"] 
    model = create_lora_model(original_model,target_modules=target_modules)
    # original_model = original_model.to(device)
    model.enable_input_require_grads()
    
    train_examples = []
    item_case_examples = []
    learning_examples = []
    
    print(f"Setting up OVERTONE edit for case: {item.get('case_id', 'unknown')}")
    for i, rewrite in enumerate(item['requested_rewrite']):
        prompt = rewrite['prompt'].format(rewrite['subject'])
        target = rewrite['target_new']['str']
        print(f"   Edit {i+1}: '{prompt}' → '{target}'")
        
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
        # if 'learning_prompt' in rewrite:
        #     for learning_item in rewrite['learning_prompt']:
        #         learning_examples.append({
        #             "text": learning_item['question'],
        #             "target": learning_item['answer']
        #         })
    train_examples.append({'item_case_examples':item_case_examples,'learning_examples':learning_examples})
    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        preprocess_function_chat, 
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer,"model": model}
    )

    training_args = TrainingArguments(
        output_dir=f'./output/',
        overwrite_output_dir=True,
        num_train_epochs=80,
        per_device_train_batch_size=4,
        learning_rate=5e-4,
        save_strategy="no",
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = OvertoneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        smooth_factor=0.1,   
        n_sigma=1.0,         
        epsilon=5e-2,        
        debug_mode=True,     
    )
    
    print(f"Starting OVERTONE training with {training_args.num_train_epochs} epochs...")
    start = time()
    trainer.train()
    exec_time = time() - start
    
    print(f"Training completed in {exec_time:.2f} seconds")
    
    metrics = {
        'case_id': item['case_id'],
        "requested_rewrite": item['requested_rewrite'],
        "time": exec_time,
        "post": compute_edit_quality(model, tokenizer, item, hparams,datatype, test_generation=test_generation),
    }
    
    model = model.unload()
    del model.peft_config

    return model, metrics