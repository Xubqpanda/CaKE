from edit_utils  import cake_no_unload, rome_no_unload, wise_no_unload, edit_no_unload
from eval_utils import test_current_edited_knowledge
import torch
from EasyEdit.easyeditor.util import nethook
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

def multi_edit(model, tokenizer, items_list, hparams, alg_name, apply_algo, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    original_weights = None
    print(f"Starting {alg_name} multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time, weights_copy = edit_no_unload(current_model, tokenizer, item, hparams, alg_name, apply_algo, test_generation)
        if original_weights is None:
            original_weights = weights_copy
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
            if alg_name == 'KN' or alg_name == 'GRACE' or alg_name == 'WISE':
                with torch.no_grad():
                    original_weights()
            # LoRA maybe something wrong
            elif alg_name == 'LoRA' or alg_name == 'QLoRA' or alg_name == 'DPO':
                current_model=current_model.unload()
                del current_model.peft_config
            elif alg_name == 'MELO':
                model = current_model
            else:
                with torch.no_grad():
                    for k, v in original_weights.items():
                        nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")
    print(f"{alg_name} multi-edit completed!")
    return current_model, all_metrics

def cake_multi_edit(base_model, tokenizer, items_list, hparams, edit_freq, MODEL_PATH, test_generation=False):    
    current_model = base_model
    all_metrics = []
    current_training_times = []
    edited_items = [] 
    print("Starting CAKE multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time = cake_no_unload(current_model, tokenizer, item, hparams, test_generation)
        current_model = current_model.merge_and_unload()
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
            current_model = None
            del current_model
            current_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={"": "cuda:0"},torch_dtype=torch.bfloat16)
    print("CAKE multi-edit completed!")
    return current_model, all_metrics

def multi_edit_rome(model, tokenizer, items_list, hparams, apply_algo, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    original_weights = None
    print("Starting ROME multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time, weights_copy = rome_no_unload(current_model, tokenizer, item, hparams, apply_algo, test_generation)
        if original_weights is None:
            original_weights = weights_copy
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
            with torch.no_grad():
                for k, v in original_weights.items():
                    nethook.get_parameter(current_model, k)[...] = v.to(f"cuda:{hparams.device}")
    print("ROME multi-edit completed!")
    return current_model, all_metrics




def multi_edit_wise(model, tokenizer, items_list, hparams, loc_data, initial_loc_index, apply_algo, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    current_loc_index = initial_loc_index
    original_weights = None
    print("Starting WISE multi-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time, updated_loc_index, weights_copy = wise_no_unload(current_model, tokenizer, item, hparams, loc_data, current_loc_index, apply_algo, test_generation)
        if original_weights is None:
            original_weights = weights_copy
        current_loc_index = updated_loc_index
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i+1) % edit_freq == 0 or (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
            with torch.no_grad():
                original_weights()
    print("WISE multi-edit completed!")
    return current_model, all_metrics

 

