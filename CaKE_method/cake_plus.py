"""CaKE+ continual editing with general-task KL and batch checkpoints."""
import hashlib
import json
import os
import random
import shutil
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import (LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from .eval_utils import compute_edit_quality

CAKE_PLUS_TEST3_TARGET_MODULES = [
    "up_proj", "down_proj", "gate_proj",
    *[f"model.layers.{layer}.self_attn.{projection}_proj"
      for layer in (0, 1, 27, 2, 3, 26, 24) for projection in ("q", "v")],
]


def preprocess_function_chat(examples,tokenizer,model):
    # 将每个item_case_examples中的text和target组合
    all_texts = []
    all_targets = []

    for item in examples['item_case_examples']:
        for example in item:
            all_texts.append(example['text'])
            all_targets.append(example['target'])

    learning_texts = []
    learning_targets = []
    for item in examples['learning_examples']:
        for example in item:
            learning_texts.append(example['text'])
            learning_targets.append(example['target'])

    final_inputs = []
    for text, target in zip(all_texts + learning_texts, all_targets + learning_targets):
        # 使用chat template格式：user问问题，assistant回答
        chat_format = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": target}
        ]
        formatted_text = tokenizer.apply_chat_template(chat_format, tokenize=False)
        final_inputs.append(formatted_text)
    random.shuffle(final_inputs)
    # 对整个序列进行tokenize
    model_inputs = tokenizer(final_inputs, padding=True, truncation=True, max_length=256, return_tensors="np")

    # 创建labels，只计算assistant回答部分的loss
    labels = model_inputs["input_ids"].copy()
    model_name = model.config.model_type
    if "llama" in model_name.lower():
        ass_tag = "<|start_header_id|>assistant<|end_header_id|"
    elif "qwen" in model_name.lower():
        ass_tag = "<|im_start|>assistant\n"
    for i, formatted_text in enumerate(final_inputs):
        # 找到assistant回答开始的位置
        assistant_start = formatted_text.find(ass_tag)
        if assistant_start != -1:
            tokens = tokenizer.tokenize(formatted_text[:assistant_start])
            start_pos = len(tokens)
            labels[i, :start_pos] = -100
    model_inputs['labels'] = labels
    return model_inputs


class CakeKLTrainer(Trainer):
    def __init__(self, original_model_ref, kl_lambda, kl_lambda_general=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model_ref = original_model_ref
        self.kl_lambda = kl_lambda
        self.kl_lambda_general = kl_lambda_general if kl_lambda_general is not None else kl_lambda
        self._step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 取出 is_general 标记，避免传入模型
        is_general = inputs.pop("is_general", None)

        outputs = model(**inputs)

        if is_general is not None:
            device = outputs.logits.device
            is_general = torch.as_tensor(is_general, device=device)

        # 手动计算 edit_loss：只对非 general 样本
        if is_general is not None:
            edit_mask = (is_general == 0)
            logits = outputs.logits
            labels = inputs.get("labels")
            if edit_mask.any() and labels is not None:
                shift_logits = logits[edit_mask, :-1, :].contiguous()
                shift_labels = labels[edit_mask, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                edit_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                edit_loss = torch.tensor(0.0, device=logits.device)
        else:
            edit_loss = outputs.loss

        with torch.no_grad():
            # Reference forward only needs logits for KL; passing labels triggers an
            # unnecessary causal-LM loss allocation on the base model and can OOM.
            ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            original_outputs = self.original_model_ref(**ref_inputs)
            original_logits = original_outputs.logits

        # KL(P_reference || P_edited), matching the recorded experiments.
        if is_general is not None:
            # 分别计算编辑样本和 general task 样本上的 KL
            edit_mask = (is_general == 0)
            general_mask = (is_general == 1)

            kl_edit = torch.tensor(0.0, device=outputs.logits.device)
            kl_general = torch.tensor(0.0, device=outputs.logits.device)

            if edit_mask.any():
                kl_edit = F.kl_div(
                    F.log_softmax(outputs.logits[edit_mask], dim=-1),
                    F.softmax(original_logits[edit_mask].detach(), dim=-1),
                    reduction='batchmean'
                )
            if general_mask.any():
                kl_general = F.kl_div(
                    F.log_softmax(outputs.logits[general_mask], dim=-1),
                    F.softmax(original_logits[general_mask].detach(), dim=-1),
                    reduction='batchmean'
                )

            kl_loss = self.kl_lambda * kl_edit + self.kl_lambda_general * kl_general
        else:
            kl_loss = self.kl_lambda * F.kl_div(
                F.log_softmax(outputs.logits, dim=-1),
                F.softmax(original_logits.detach(), dim=-1),
                reduction='batchmean'
            )

        total_loss = edit_loss + kl_loss
        self._step_count += 1
        if self._step_count % 10 == 0:
            if is_general is not None:
                print(f"Step {self._step_count}: Edit={edit_loss:.4f}, KL_edit={kl_edit:.4f}, KL_general={kl_general:.4f}, Total={total_loss:.4f}")
            else:
                print(f"Step {self._step_count}: Edit={edit_loss:.4f}, KL={kl_loss:.4f}, Total={total_loss:.4f}")

        return (total_loss, outputs) if return_outputs else total_loss


def load_general_task_data(hparams):
    """加载 general task 样本（文本格式），支持 general_task_num 限制数量"""
    general_examples = []
    general_task_file = getattr(hparams, 'general_task_file', None)
    general_task_num = getattr(hparams, 'general_task_num', None)

    if general_task_file is not None and os.path.exists(general_task_file):
        with open(general_task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 支持两种格式：{"text": "...", "target": "..."} 或 {"instruction": "...", "input": "...", "output": "..."}
        for item in data:
            if 'text' in item and 'target' in item:
                general_examples.append({"text": item['text'], "target": item['target']})
            elif 'instruction' in item:
                text = item['instruction']
                if item.get('input'):
                    text += "\n" + item['input']
                general_examples.append({"text": text, "target": item.get('output', '')})
        # 限制 general task 样本数量
        if general_task_num is not None and general_task_num > 0:
            general_examples = general_examples[:general_task_num]
        print(f"Loaded {len(general_examples)} general task samples from {general_task_file}")
    else:
        if general_task_file:
            print(f"Warning: general_task_file {general_task_file} not found, skipping general task KL.")
    return general_examples


def tokenize_general_task_pool(tokenizer, general_examples, model, max_length=256):
    """对 general task 样本进行预 tokenize（使用 chat template），返回 pool。"""
    if not general_examples:
        return []

    pool = []
    for ex in general_examples:
        chat_format = [
            {"role": "user", "content": ex['text']},
            {"role": "assistant", "content": ex['target']}
        ]
        formatted_text = tokenizer.apply_chat_template(chat_format, tokenize=False)
        encoded = tokenizer(formatted_text, truncation=True, max_length=max_length)
        pool.append(encoded)

    print(f"Pre-tokenized {len(pool)} general task samples for dynamic batching.")
    return pool


class DynamicGeneralTaskCollator:
    """
    Data collator that dynamically mixes general task samples into each batch.

    For each batch of edit samples, randomly samples `general_per_batch` general task
    examples, tokenizes them on-the-fly (if needed) or uses pre-tokenized pool,
    and concatenates them with edit samples.

    General task samples have all labels set to -100 (only KL constraint, no edit loss).
    """
    def __init__(self, tokenizer, general_task_pool, general_per_batch=2):
        self.tokenizer = tokenizer
        # general_task_pool: list of dicts with 'input_ids', 'attention_mask' (pre-tokenized)
        self.general_task_pool = general_task_pool
        self.general_per_batch = general_per_batch

    def __call__(self, features):
        # features: list of dicts for edit samples (may contain numpy arrays)
        # Pad edit samples manually
        batch = self._pad_batch(features)

        if not self.general_task_pool or self.general_per_batch <= 0:
            return batch

        # Randomly sample general task examples
        sampled = random.sample(
            self.general_task_pool,
            min(self.general_per_batch, len(self.general_task_pool))
        )

        # Pad general task samples
        general_batch = self._pad_batch(sampled)

        # General samples contribute KL only; exclude them from edit cross-entropy.
        general_batch['labels'] = torch.full_like(general_batch['input_ids'], -100)
        general_batch['is_general'] = torch.ones(general_batch['input_ids'].size(0), dtype=torch.long)

        # Mark edit samples
        batch['is_general'] = torch.zeros(batch['input_ids'].size(0), dtype=torch.long)

        # Unify sequence length between edit and general batches before concatenation
        edit_seq_len = batch['input_ids'].size(1)
        general_seq_len = general_batch['input_ids'].size(1)
        target_len = max(edit_seq_len, general_seq_len)

        batch = self._pad_to_length(batch, target_len)
        general_batch = self._pad_to_length(general_batch, target_len)

        # Concatenate edit and general task batches
        merged = {}
        for key in set(list(batch.keys()) + list(general_batch.keys())):
            if key in batch and key in general_batch:
                merged[key] = torch.cat([batch[key], general_batch[key]], dim=0)
            elif key in batch:
                merged[key] = batch[key]
            else:
                merged[key] = general_batch[key]

        return merged

    def _pad_batch(self, features):
        """Manual padding for a list of feature dicts."""
        max_len = max(len(f['input_ids']) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            # Handle both list and numpy array
            ids = list(f['input_ids'])
            mask = list(f['attention_mask'])
            # Labels may not exist (e.g., in general task pool pre-tokenized without labels)
            lbl = list(f.get('labels', [-100] * len(ids)))
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lbl + [-100] * pad_len)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def _pad_to_length(self, batch, target_len):
        """Pad a batch dict to the target sequence length."""
        current_len = batch['input_ids'].size(1)
        if current_len >= target_len:
            return batch
        pad_len = target_len - current_len
        bs = batch['input_ids'].size(0)
        result = {}
        for key, tensor in batch.items():
            if key == 'input_ids':
                result[key] = torch.cat([tensor, torch.full((bs, pad_len), self.tokenizer.pad_token_id, dtype=tensor.dtype)], dim=1)
            elif key == 'attention_mask':
                result[key] = torch.cat([tensor, torch.zeros(bs, pad_len, dtype=tensor.dtype)], dim=1)
            elif key == 'labels':
                result[key] = torch.cat([tensor, torch.full((bs, pad_len), -100, dtype=tensor.dtype)], dim=1)
            else:
                # is_general and other 1D tensors - no padding needed
                result[key] = tensor
        return result


def cake_plus_return_lora_weights(original_model, tokenizer, items_list, hparams, batch_num,test_generation=False):
    kl_reference = getattr(hparams, 'kl_reference_model', None)
    target_modules = CAKE_PLUS_TEST3_TARGET_MODULES
    model = get_peft_model(original_model, LoraConfig(
        r=hparams.rank, lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.train_lora_dropout, bias="none",
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM,
    ))
    model.enable_input_require_grads()

    general_per_batch = getattr(hparams, 'general_per_batch', 2)
    general_task_pool = []
    if general_per_batch > 0:
        # Only load/mix general-task samples when they are actually used in a batch.
        general_examples = load_general_task_data(hparams)
        general_task_pool = tokenize_general_task_pool(tokenizer, general_examples, model, max_length=256)
        if general_task_pool:
            print(f"Dynamic general task mixing: {general_per_batch} samples per batch from {len(general_task_pool)} pool")
    else:
        print("General-task KL disabled for this run (general_per_batch <= 0).")

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

        train_examples.append({
            'item_case_examples': item_case_examples,
            'learning_examples': learning_examples,
        })
    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        preprocess_function_chat,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer,"model": model}
    )

    # 构建 data collator（动态混合 general task 样本）
    data_collator = DynamicGeneralTaskCollator(
        tokenizer=tokenizer,
        general_task_pool=general_task_pool,
        general_per_batch=general_per_batch,
    ) if general_task_pool else None

    training_args = TrainingArguments(
            output_dir=f'./output/',
            overwrite_output_dir=True,
            num_train_epochs=hparams.num_train_epochs,
            per_device_train_batch_size=hparams.per_device_train_batch_size,
            learning_rate=hparams.learning_rate,
            save_strategy="no",
            bf16=True,
            logging_steps=10,
            report_to="none",
        )
    kl_lambda = getattr(hparams, 'kl_lambda', 0.05)
    kl_lambda_general = getattr(hparams, 'kl_lambda_general', kl_lambda)
    print(f"Using kl_lambda = {kl_lambda}, kl_lambda_general = {kl_lambda_general} for batch KL training")
    trainer = CakeKLTrainer(
        original_model_ref=kl_reference if kl_reference is not None else original_model,
        kl_lambda=kl_lambda,
        kl_lambda_general=kl_lambda_general,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    start = time()
    trainer.train()
    exec_time = time() - start
    lora_weights = get_peft_model_state_dict(model)
    model = model.unload()
    return lora_weights, exec_time


def apply_lora_weights_to_model(base_model, lora_weights, hparams=None):
    target_modules = CAKE_PLUS_TEST3_TARGET_MODULES
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


def load_kl_reference(hparams):
    """Load a frozen reference, refreshed before each edit batch."""
    model_name = hparams.model_name
    print(f"Loading frozen KL reference model from {model_name} ...")
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=getattr(hparams, 'max_memory', None),
        torch_dtype=torch.bfloat16,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print("Frozen KL reference model ready (eval mode, requires_grad=False).")
    return ref


def sync_model_weights(dst, src):
    """Copy src weights into dst in place (refresh a per-batch KL reference)."""
    state = src.state_dict()
    with torch.no_grad():
        for name, param in dst.named_parameters():
            if name in state:
                param.copy_(state[name])


def cake_plus_continual_edit(model, tokenizer, items_list, hparams, edit_freq, datatype,test_generation=False):
    current_model = model
    num_edits = len(items_list)
    all_metrics = []
    current_training_times = []
    kl_ref_mode = getattr(hparams, 'kl_ref', 'current')
    hparams.kl_reference_model = None
    if kl_ref_mode == 'batch':
        hparams.kl_reference_model = load_kl_reference(hparams)
    print(f"KL reference mode = {kl_ref_mode}")
    # Chunk the items_list into batches of size edit_freq
    def chunk_list(data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    chunks = list(chunk_list(items_list, edit_freq))
    checkpoint_dir = getattr(hparams, 'checkpoint_dir', None)
    start_batch = 0
    signature = hashlib.sha256(json.dumps({
        'items': items_list, 'edit_freq': edit_freq, 'kl_ref': kl_ref_mode,
        'config': {k: v for k, v in vars(hparams).items()
                   if isinstance(v, (str, int, float, bool, type(None)))
                   and k != 'checkpoint_dir'},
    }, sort_keys=True, default=str).encode()).hexdigest()
    if checkpoint_dir:
        # Keep a previous committed snapshot until its replacement is complete.
        previous = checkpoint_dir + '.previous'
        if not os.path.exists(checkpoint_dir) and os.path.exists(previous):
            os.rename(previous, checkpoint_dir)
        state_path = os.path.join(checkpoint_dir, 'resume.pt')
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            if state['signature'] != signature:
                raise ValueError('Checkpoint configuration/data mismatch; use a new checkpoint_dir.')
            saved = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16,
                                                        device_map='cpu')
            current_model.load_state_dict(saved.state_dict())
            del saved
            start_batch = state['completed_batches']
            current_training_times = state['training_times']
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            print(f'Resumed checkpoint: {start_batch}/{len(chunks)} batches complete', flush=True)
    print("Starting CaKE+ continual-edit...")
    for i, accumulated_items in enumerate(chunks):
        if i < start_batch:
            continue
        print(f"Processing {i+1}th batch of {len(chunks)}")
        # the last item of the batch shows the total training time
        current_training_times.append(0)
        print(f"Training LoRA for {len(accumulated_items)} knowledge points...")
        if kl_ref_mode == 'batch' and hparams.kl_reference_model is not None:
            sync_model_weights(hparams.kl_reference_model, current_model)
            print("  -> KL reference refreshed to the model before this batch")
        lora_weights, exec_time = cake_plus_return_lora_weights(current_model, tokenizer, accumulated_items, hparams, i, test_generation)
        current_training_times.append(exec_time)
        current_model = apply_lora_weights_to_model(current_model, lora_weights, hparams)
        current_model = current_model.merge_and_unload()
        if checkpoint_dir:
            pending = checkpoint_dir + '.pending'
            shutil.rmtree(pending, ignore_errors=True)
            os.makedirs(pending, exist_ok=True)
            current_model.save_pretrained(pending, safe_serialization=True)
            tokenizer.save_pretrained(pending)
            torch.save({
                'signature': signature, 'completed_batches': i + 1,
                'training_times': current_training_times,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            }, os.path.join(pending, 'resume.pt'))
            previous = checkpoint_dir + '.previous'
            shutil.rmtree(previous, ignore_errors=True)
            if os.path.exists(checkpoint_dir):
                os.rename(checkpoint_dir, previous)
            os.rename(pending, checkpoint_dir)
            shutil.rmtree(previous, ignore_errors=True)
            print(f'Checkpoint saved: batch {i + 1}/{len(chunks)} -> {checkpoint_dir}', flush=True)
    print(f"Testing knowledge retention after {num_edits} edits...")
    test_metrics = [
        {"case_id": item["case_id"], "requested_rewrite": item["requested_rewrite"],
         "post": compute_edit_quality(current_model, tokenizer, item, hparams,
                                      datatype, test_generation)}
        for item in items_list
    ]
    all_metrics.extend(test_metrics)
    current_training_times = []
    print("CaKE+ continual-edit completed.")
    return current_model, all_metrics
