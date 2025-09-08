import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
import os
from time import time
from transformers.pytorch_utils import Conv1D
from transformers import AutoTokenizer
import math
from torch.nn.utils.rnn import pad_sequence
    
# ===== tool func =====

def get_module(module: nn.Module, module_name: str) -> nn.Module:
    for name in module_name.split("."):
        module = getattr(module, name)
    return module

def get_shape(module: Union[nn.Linear, Conv1D]) -> Tuple[int]:
    shape = tuple(module.weight.shape)
    return shape[::-1] if isinstance(module, nn.Linear) else shape

def cross_entropy(logits: torch.FloatTensor, labels: torch.LongTensor):
    if len(logits.shape) == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)

    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        return F.cross_entropy(logits, labels)

def pad_tensor(tensor, target_length, dim=0, padding_value=0):
    tensor_length = tensor.size(dim)
    if tensor_length >= target_length:
        return tensor.narrow(dim, 0, target_length)
    else:
        padding = target_length - tensor_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding
        pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_tensor], dim=dim)
# ===== Data preprocess =====

def prepare_ultraedit_data(edit_item, tokenizer, model, hparams):
    edit_data = []
    for rewrite in edit_item['requested_rewrite']:
        prompt = rewrite['prompt'].format(rewrite['subject'])
        target = rewrite['target_new']['str']
        main_sample = create_ultraedit_sample(prompt, target, tokenizer, model)
        edit_data.append(main_sample)
        if 'rephrase_prompt' in rewrite:
            for rephrase_item in rewrite['rephrase_prompt']:
                rephrase_sample = create_ultraedit_sample(
                    rephrase_item['question'], 
                    rephrase_item['answer'], 
                    tokenizer, 
                    model
                )
                edit_data.append(rephrase_sample)
        if 'learning_prompt' in rewrite:
            for learning_item in rewrite['learning_prompt']:
                learning_sample = create_ultraedit_sample(
                    learning_item['question'],
                    learning_item['answer'],
                    tokenizer,
                    model
                )
                edit_data.append(learning_sample)
    
    return edit_data

def create_ultraedit_sample(prompt, target, tokenizer, model):
    answer = " " + target
    tok_prompt = tokenizer(
        prompt,
        return_tensors="pt",
    )
    tok_answer = tokenizer(
        answer,
        return_tensors="pt",
        add_special_tokens=False
    )
    tok_tuples = {
        key: torch.cat((value, tok_answer[key][:, :-1]), -1)
        for key, value in tok_prompt.items()
    }
    tok_tuples["labels"] = torch.cat((
        torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
        tok_answer["input_ids"]
    ), -1)

    device = next(model.parameters()).device
    tok_tuples = {k: v.to(device) for k, v in tok_tuples.items()}
    
    return tok_tuples

def pad_tok_tuples(tok_tuples_list: List[Dict[str, torch.LongTensor]], device) -> Dict[str, torch.LongTensor]:
    return {
        k: pad_sequence(
            [t[k].squeeze(0) for t in tok_tuples_list],
            batch_first=True,
            padding_value=-100 if k == "labels" else 0
        ).to(device)
        for k in tok_tuples_list[0].keys()
    }
# ===== TracerDict Class =====

class Tracer:
    def __init__(self, module: nn.Module, cache_mask: torch.LongTensor):
        cache_indices = torch.where(cache_mask)

        def forward_hook(module: nn.Module, inputs: Tuple[torch.FloatTensor], outputs: Tuple[torch.FloatTensor]):
            self.keys = inputs[0][cache_indices].detach()
            
        def backward_hook(module: nn.Module, inputs_grad: Tuple[torch.FloatTensor], outputs_grad: Tuple[torch.FloatTensor]):
            self.values_grad = outputs_grad[0][cache_indices].detach()

        self.handles = [
            module.register_forward_hook(forward_hook),
            module.register_full_backward_hook(backward_hook)
        ]

class TracerDict(dict):
    def __init__(self, model: nn.Module, edit_modules: List[str], tuples: Dict[str, torch.LongTensor]):
        cache_mask = tuples["labels"] != -100
        for module_name in edit_modules:
            module = get_module(model, module_name)
            self[module_name] = Tracer(module, cache_mask)
            
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        for v in self.values():
            for h in v.handles:
                h.remove()

# ===== RunningMeanStd class =====

class RunningMeanStd(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

# ===== UltraEdit core component =====

def create_shape_counter(model, hparams):
    from collections import Counter
    
    shape_counter = Counter()
    name2idx = {}
    edit_modules = ["model.layers.{}.mlp.up_proj".format(i) for i in range(model.config.num_hidden_layers)] + \
                  ["model.layers.{}.mlp.down_proj".format(i) for i in range(model.config.num_hidden_layers)]
    for module_name in edit_modules:
        shape = get_shape(get_module(model, module_name))
        name2idx[module_name] = shape_counter[shape]
        shape_counter[shape] += 1        
    return shape_counter, name2idx, edit_modules

def create_ultraedit_components(model, hparams):
    shape_counter, name2idx, edit_modules = create_shape_counter(model, hparams)
    lifelong_normalizer = nn.ModuleDict({
        str(k): RunningMeanStd(k[0] + k[1])
        for k, v in shape_counter.items()
    }).to(f"cuda:{hparams.device}")
    
    return {
        'normalizer': lifelong_normalizer,
        'shape_counter': shape_counter,
        'name2idx': name2idx,
        'edit_modules': edit_modules
    }

def cache_activations_and_gradients(model, edit_data, components, hparams, batch_idx):
    normalizer = components['normalizer']
    edit_modules = components['edit_modules']
    cache_dir = f"./ultraedit_cache/"
    os.makedirs(cache_dir, exist_ok=True)
    device = f"cuda:{hparams.device}"
    batched_data = pad_tok_tuples(edit_data, device)
    with TracerDict(model, edit_modules, batched_data) as tr:
        logits = model(**batched_data)["logits"]
        cross_entropy(logits, batched_data["labels"]).backward()
    for module_idx, module_name in enumerate(edit_modules):
        if module_name not in tr:
            continue
        shape = get_shape(get_module(model, module_name))
        keys = tr[module_name].keys.to(torch.float32).to(f"cuda:{hparams.device}")
        values_grad = tr[module_name].values_grad.to(torch.float32).to(f"cuda:{hparams.device}")
        normalizer[str(shape)].update(torch.cat((keys, values_grad), -1))
        torch.save(keys, f"{cache_dir}/{module_idx}_{batch_idx}_keys.pth")
        torch.save(values_grad, f"{cache_dir}/{module_idx}_{batch_idx}_values_grad.pth")
            
def predict_parameter_shifts(model, components, hparams, batch_idx, edit_freq):
    lr = 1e-3
    normalizer = components['normalizer']
    edit_modules = components['edit_modules']
    cache_dir = f"./ultraedit_cache/"
    param_shifts = {}
    for module_idx, module_name in enumerate(edit_modules):
        shape = get_shape(get_module(model, module_name))
        hidden_states = torch.cat([torch.load(f"{cache_dir}/{module_idx}_{idx}_keys.pth")for idx in range(batch_idx + 1)])
        values_grad = torch.cat([torch.load(f"{cache_dir}/{module_idx}_{idx}_values_grad.pth")for idx in range(batch_idx + 1)])
        v_feature = torch.empty((0, shape[1]), device=f"cuda:{hparams.device}")
        batch_size = edit_freq
        for start_idx in range(0, hidden_states.shape[0], batch_size):
            end_idx = start_idx + batch_size
            hidden_states_once = pad_tensor(hidden_states[start_idx:end_idx], batch_size, 0)
            values_grad_once = pad_tensor(values_grad[start_idx:end_idx], batch_size, 0)
            with torch.no_grad():
                z_feature = torch.cat((hidden_states_once, values_grad_once), -1)
                z_feature = normalizer[str(shape)](z_feature)
                (hidden_states_hat, pseudo_values_hat) = z_feature.split([shape[0], shape[1]], -1)
                coeffs = -lr * (hidden_states_hat * hidden_states_hat).sum(-1).unsqueeze(-1)
            v_feature = torch.cat((v_feature, coeffs * pseudo_values_hat))
        with torch.no_grad():
            mat = hidden_states.T @ hidden_states + torch.eye(shape[0], device=f"cuda:{hparams.device}")
        v_feature = v_feature[:hidden_states.shape[0], :]
        param_shift = torch.linalg.solve(mat, hidden_states.T @ v_feature)
        param_shifts[module_name] = param_shift.to(next(model.parameters()).device)
    return param_shifts

def apply_parameter_shifts(model, param_shifts, is_reverse=False):
    for module_name, param_shift in param_shifts.items():
        module = get_module(model, module_name)
        if isinstance(module, nn.Linear):
            param_shift = param_shift.T
        if is_reverse:
            param_shift = -param_shift
        module.weight.data += param_shift.to(module.weight.data.dtype)

def cleanup_ultraedit_cache():
    cache_dir = f"./ultraedit_cache/"
    if os.path.exists(cache_dir):
        for file_name in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
# === execute func ===

def execute_ultraedit_batch(model, tokenizer, batch_items, components, hparams, batch_idx, edit_freq):
    from time import time
    total_start_time = time()
    all_edit_data = []
    for item in batch_items:
        edit_data = prepare_ultraedit_data(item, tokenizer, model, hparams)
        all_edit_data.extend(edit_data) 
    cache_activations_and_gradients(model, all_edit_data, components, hparams, batch_idx)
    param_shifts = predict_parameter_shifts(model, components, hparams, batch_idx, edit_freq)
    apply_parameter_shifts(model, param_shifts, is_reverse=False)
    total_exec_time = time() - total_start_time
    return total_exec_time