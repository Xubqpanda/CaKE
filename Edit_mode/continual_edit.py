from ..edit_utils import cake_no_unload, rome_no_unload, wise_no_unload, edit_no_unload
from ..eval_utils import test_current_edited_knowledge

def cake_continual_edit(base_model, tokenizer, items_list, hparams, edit_freq, test_generation=False):    
    current_model = base_model
    all_metrics = []
    current_training_times = []
    edited_items = [] 
    print("Starting CAKE continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time = cake_no_unload(current_model, tokenizer, item, hparams, test_generation)
        current_model = current_model.merge_and_unload()  
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print("CAKE continual-edit completed!")
    return current_model, all_metrics

def continual_edit_rome(model, tokenizer, items_list, hparams, apply_algo, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    print("Starting ROME continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time, weights_copy = rome_no_unload(current_model, tokenizer, item, hparams, apply_algo, test_generation)
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print("ROME continual-edit completed!")
    return current_model, all_metrics

def continual_edit_wise(model, tokenizer, items_list, hparams, loc_data, initial_loc_index, apply_algo, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    current_loc_index = initial_loc_index
    print("Starting WISE continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time, updated_loc_index, weights_copy = wise_no_unload(current_model, tokenizer, item, hparams, loc_data, current_loc_index, apply_algo, test_generation)
        current_loc_index = updated_loc_index
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print("WISE continual-edit completed!")
    return current_model, all_metrics

def continual_edit(model, tokenizer, items_list, hparams, alg_name, apply_algo, edit_freq, test_generation=False):
    current_model = model
    all_metrics = []
    current_training_times = []
    edited_items = []
    original_weights = None
    print(f"Starting {alg_name} continual-edit...")
    for i, item in enumerate(items_list):
        print(f"Processing item {i+1}/{len(items_list)}: {item.get('case_id', 'unknown')}")
        current_model, exec_time, weights_copy = edit_no_unload(current_model, tokenizer, item, hparams, alg_name, apply_algo, test_generation)
        edited_items.append(item)
        current_training_times.append(exec_time)
        if (i + 1) == len(items_list):
            test_metrics = test_current_edited_knowledge(current_model, tokenizer, edited_items, hparams, current_training_times, test_generation)
            all_metrics.extend(test_metrics)
            current_training_times = []
            edited_items = []
    print(f"{alg_name} continual-edit completed!")
    return current_model, all_metrics
