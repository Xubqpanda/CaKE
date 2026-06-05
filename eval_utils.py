import random

def test_current_edited_knowledge(model, tokenizer, edited_items, hparams, current_training_times, datatype, test_generation=False):
    test_metrics = []
    for i, item in enumerate(edited_items):
        metrics = {
            'case_id': item['case_id'],
            'requested_rewrite': item['requested_rewrite'],
            'time': current_training_times[i],
            'post': compute_edit_quality(model, tokenizer, item, hparams, datatype,test_generation)
        }
        test_metrics.append(metrics)
    
    return test_metrics


def check_answer(model,question, tokenizer,answer, device,max_new_tokens=50):
    """检查模型是否能正确回答问题"""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Expected: {answer}")
    print(f"Generated: {generated_text}")
    accuracy = check_answer_in_pred(generated_text,answer)
    print(f"Correct: {'✓' if accuracy > 0 else '✗'}")
    print("-" * 50)
    return accuracy

def check_answer_chat(model,question, tokenizer,answer, device,max_new_tokens=50):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": question}],return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Expected: {answer}")
    print(f"Generated: {generated_text}")
    accuracy = check_answer_in_pred(generated_text,answer)
    print(f"Correct: {'✓' if accuracy > 0 else '✗'}")
    print("-" * 50)
    return accuracy

def get_response(model, tok, messages,device, max_new_tokens=1):
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)
    output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')

def evaluate_response(prompt_qa, output_qa, label, device, system_msg_eval):
    if output_qa.lower() in label.lower() or label.lower() in output_qa.lower():
        response_eval = 1
    else:
        response_eval = 0
    # else:
    #     user_msg_eval = f"""Text 1: {label} \nText 2: {output_qa}"""
    #     messages_eval = [{"role": "system", "content": system_msg_eval}, {"role": "user", "content": user_msg_eval}]
    #     response_eval = get_response(model_eval, tok_eval, messages_eval,device, max_new_tokens=1)
    print(f"===== Question: {prompt_qa} | Prediction: {output_qa} | Label: {label} | Evaluation: {response_eval} =====")
    if str(response_eval) not in ['0', '1']:
        response_eval = 0
    return int(response_eval), output_qa


def check_answer_chat_hallu(model,question, tokenizer,answer, device,system_msg,max_new_tokens=16):
    inputs = tokenizer.apply_chat_template([{"role": "system", "content": system_msg},{"role": "user", "content": question}],return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:],skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')
    print(f"Question: {question}")
    print(f"Expected: {answer}")
    print(f"Generated: {generated_text}")
    accuracy = check_answer_in_pred(generated_text,answer)
    print(f"Correct: {'✓' if accuracy > 0 else '✗'}")
    print("-" * 50)
    return accuracy

def test_multi_turn_robustness(model, tokenizer, primary_question, primary_answer,yes_question,yes_answer, device,system_msg_qa, system_msg_eval,turns=10):
    acc_ls = []
    output_qa_ls = []
    questions_follow_up = [f'Your answer to the original question is wrong. {yes_question} Respond with "Yes" or "No."' for _ in range(10)]
    label_follow_up = yes_answer
    messages_qa = []
    if system_msg_qa:
        messages_qa.append({"role": "system", "content": system_msg_qa})
    messages_qa.append({"role": "user", "content": primary_question})
    current_output = get_response(model, tokenizer, messages_qa, device, max_new_tokens=16)
    eval_acc, _ = evaluate_response(primary_question, current_output, primary_answer, device, system_msg_eval)
    acc_ls.append(eval_acc)
    output_qa_ls.append(current_output)
    for question in questions_follow_up:
        messages_qa.append({"role": "assistant", "content": current_output})
        messages_qa.append({"role": "user", "content": question})
        current_output = get_response(model, tokenizer, messages_qa, device, max_new_tokens=16)
        eval_acc, _ = evaluate_response(primary_question, current_output, yes_answer, device, system_msg_eval)
        acc_ls.append(eval_acc)
        output_qa_ls.append(current_output)        
    return acc_ls, output_qa_ls

def compute_edit_quality(model, tokenizer, edit_item, hparams, datatype, test_generation=False):
    metrics = {}
    device = f"cuda:{hparams.device}"
    if datatype == 'wikibigedit':
        metrics = {
        'effective':[],
        'multi-hop-accuracy':[],
        'persona-accuracy':[],
        'generalizaton':[],
        'locality-accuracy':[],
        }
        ans = [edit_item['ans']]
        metrics['effective'].append(check_answer_chat(model,'Question: '+edit_item['update']+'\nPlease answer the question directly.',tokenizer,ans,device,max_new_tokens=50))
        metrics['generalizaton'].append(check_answer_chat(model,'Question: '+edit_item['rephrase']+'\nPlease answer the question directly.',tokenizer,ans,device,max_new_tokens=50))
        metrics['persona-accuracy'].append(check_answer_chat(model,'Question: '+edit_item['personas']+'\nPlease answer the question directly.',tokenizer,ans,device,max_new_tokens=50))
        ans = [edit_item['mhop_ans']]
        metrics['multi-hop-accuracy'].append(check_answer_chat(model,'Question: '+edit_item['mhop']+'\nPlease answer the question directly.',tokenizer,ans,device,max_new_tokens=50))
        ans = [edit_item['loc_ans']]
        metrics['locality-accuracy'].append(check_answer_chat(model,'Question: '+edit_item['loc']+'\nPlease answer the question directly.',tokenizer,ans,device,max_new_tokens=50))

    elif datatype == 'MQuAKE-T' or datatype == 'MQuAKE-CF-3k' or datatype == 'MQuAKE-CF-3k-v2':
        metrics = {
        'hop_wise':[],
        'accuracy':[],
        }
        for i in edit_item['new_single_hops']:
            temp_metrics = []
            ans = i['answer_alias']
            ans.append(i['answer'])
            temp_metrics.append(check_answer_chat(model,'Question: ' + i['question']+' Answer: The answer is',tokenizer,ans,device,max_new_tokens=50))
            metrics['hop_wise'].append(temp_metrics)
        answer = edit_item['new_answer_alias']
        answer.append(edit_item['new_answer'])
        metrics['accuracy'].append(check_answer_chat(model,'Question: ' + edit_item['questions'][0]+' Answer: The answer is',tokenizer,answer,device,max_new_tokens=50))
    
    elif datatype == 'halluedit_meta_llama_3_8b_instruct' or 'halluedit_qwen_2_5_7b_instruct':
        metrics = {
            'efficacy': [],           # Efficacy
            'generalization': [],     # Generalization - Rephrase
            'yes_question': [],       # Generalization - Yes Questions  
            'no_question': [],        # Generalization - No Questions
            'multiple_choice': [],    # Generalization - Multiple Choice
            'reversed_relation': [],  # Generalization - Reversed Relations
            'locality': [],           # Locality
            '2_hops_acc': [],         # Portability - 2_hop
            '3_hops_acc': [],         # Portability - 3_hop  
            '4_hops_acc': [],         # Portability - 4_hop
            '5_hops_acc': [],         # Portability - 5_hop
            '6_hops_acc': [],         # Portability - 6_hop
            'robustness': []          # Robustness
        }
        question_ans = [edit_item['object']]
        paraphrased_question_ans = [edit_item['object']]
        yes_ans = ['Yes']
        no_ans = ['No']
        multiple_choice_ans = [edit_item['multiple_choice_labels']]
        reversed_relation_ans = [edit_item['subject']]
        pre_locality_response = [edit_item['locality_pre_ans']]
        system_msg_qa = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."
        system_msg_multiple_choice = "Always respond to the multiple-choice question by selecting from the provided options. Only output the choice letter (A, B, C, or D)."
        system_msg_eval = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically; otherwise, output '0'. Do not repeat the question or provide any explanation."   
        print("Efficacy test")
        metrics['efficacy'].append(check_answer_chat_hallu(model,edit_item['question'], tokenizer, question_ans, device,system_msg_qa, max_new_tokens=16))
        print("Generalization (rephrase) test")
        metrics['generalization'].append(check_answer_chat_hallu(model, edit_item['paraphrased_question'], tokenizer, paraphrased_question_ans, device,system_msg_qa,max_new_tokens=16))
        print("Generalization (Yes question) test")
        metrics['yes_question'].append(check_answer_chat_hallu(model, edit_item['yes_question'], tokenizer, yes_ans, device,system_msg_qa, max_new_tokens=16))
        print("Generalization (No question) test")
        metrics['no_question'].append(check_answer_chat_hallu(model, edit_item['no_question'], tokenizer, no_ans, device,system_msg_qa, max_new_tokens=16))
        print("Generalization (Multiple choice) test")
        metrics['multiple_choice'].append(check_answer_chat_hallu(model, edit_item['multiple_choice_with_letters'], tokenizer, multiple_choice_ans, device,system_msg_multiple_choice, max_new_tokens=16))
        print("Generalization (Reversed relation) test")
        metrics['reversed_relation'].append(check_answer_chat_hallu(model, edit_item['reversed_relation_question'], tokenizer, reversed_relation_ans, device,system_msg_qa, max_new_tokens=16))
        print("Locality test")
        metrics['locality'].append(check_answer_chat_hallu(model, edit_item['locality_question'], tokenizer, pre_locality_response, device,system_msg_qa, max_new_tokens=16))
        
        print("Robustness test")
        primary_question = edit_item['question']
        primary_answer = edit_item['object']
        yes_question = edit_item['yes_question']
        yes_answer = 'Yes'
        robustness_accs, _ = test_multi_turn_robustness(model, tokenizer, primary_question, primary_answer,yes_question,yes_answer, device,system_msg_qa,system_msg_eval,turns=10)
        metrics['robustness'].extend(robustness_accs)          
        
        for hop_num in range(2, 7):
            hop_key = f'{hop_num}_hops'
            hop_data = edit_item[hop_key][0]
            question = hop_data['question']
            answer = [hop_data['answer']]
            print(f"{hop_num}-hop test")
            metrics[f'{hop_num}_hops_acc'].append(check_answer_chat_hallu(model, question, tokenizer, answer, device,system_msg_qa, max_new_tokens=16))
            
    return metrics

def check_answer_in_pred(pred, answers):
    pred = pred.lower()
    print(f"DEBUG - pred: '{pred}'")
    print(f"DEBUG - answers: {answers}")
    result = any([a.lower() in pred for a in answers])
    for a in answers:
        contains = a.lower() in pred
        print(f"DEBUG - '{a.lower()}' in '{pred}': {contains}")
    
    print(f"DEBUG - final result: {result}")
    return result

def calculate_averages(data):
    total_cases = len(data)
    result = {"total_cases": total_cases}
    all_metrics = {}
    for case in data:
        metrics = case.get('post', {})
        for metric_name, values in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            def flatten_values(item):
                if isinstance(item, list):
                    result = []
                    for sub_item in item:
                        result.extend(flatten_values(sub_item))
                    return result
                else:
                    if isinstance(item, (int, float, bool)):
                        return [int(bool(item)) if isinstance(item, (int, float)) else int(item)]
                    return []
            all_metrics[metric_name].extend(flatten_values(values))
    for metric_name, values in all_metrics.items():
        correct_count = sum(values)
        total_count = len(values)
        accuracy = correct_count / total_count    
        result[f"{metric_name}_accuracy"] = round(accuracy, 4)
        result[f"{metric_name}_count"] = f"{correct_count}/{total_count}"
        
    return result