import random

def test_current_edited_knowledge(model, tokenizer, edited_items, hparams, current_training_times, test_generation=False):
    test_metrics = []
    for i, item in enumerate(edited_items):
        metrics = {
            'case_id': item['case_id'],
            'requested_rewrite': item['requested_rewrite'],
            'time': current_training_times[i],
            'post': compute_edit_quality(model, tokenizer, item, hparams, test_generation)
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

def compute_edit_quality(model, tokenizer, edit_item, hparams, test_generation=False):
    metrics = {}
    device = f"cuda:{hparams.device}"
    if 'new_single_hops' in edit_item and edit_item['new_single_hops']:
        Efficacy_metrics = []
        for i in edit_item['new_single_hops']:
            ans = i['answer_alias']
            ans.append(i['answer'])
            # if i['cloze'] is not None:
            #     temp_metrics.append(check_answer_chat(model,'Answer: ' + i['cloze'],tokenizer,ans,device,max_new_tokens=50))
            print("Efficacy test")
            Efficacy_metrics.append(check_answer_chat(model,'Question: ' + i['question']+'\nPlease answer the question directly.',tokenizer,ans,device,max_new_tokens=50))
        metrics['Efficacy'] = Efficacy_metrics
        
    if 'questions' in edit_item and edit_item['questions'] and edit_item['questions'][0]:
        Multi_hop_metrics = []
        mhop_answer = edit_item['new_answer_alias']
        if edit_item.get('new_answer'):
            mhop_answer.append(edit_item['new_answer'])
        print("Multi_hop test")
        Multi_hop_metrics.append(check_answer_chat(model, 'Question: ' + edit_item['questions'][0]+'\nPlease answer the question directly.',tokenizer,mhop_answer,device,max_new_tokens=50))
        metrics['Multi_hop'] = Multi_hop_metrics
    
    if 'rephrase' in edit_item and edit_item['rephrase']:
        Generalization_metrics = []
        answer = edit_item['ans']
        answer = [answer]
        print("Generalization test")
        Generalization_metrics.append(check_answer_chat(model, 'Question: ' + edit_item['rephrase']+'\nPlease answer the question directly.',tokenizer,answer,device,max_new_tokens=50))
        metrics['Generalization'] = Generalization_metrics
        
    if 'personas' in edit_item and edit_item['personas']:
        Personas_metrics = []
        answer = edit_item['ans']
        answer = [answer]
        print("Personas test")
        Personas_metrics.append(check_answer_chat(model, 'Question: ' + edit_item['personas']+'\nPlease answer the question directly.',tokenizer,answer,device,max_new_tokens=50))
        metrics['Personas'] = Personas_metrics
        
    if 'loc' in edit_item and edit_item['loc']:
        Specificity_metrics = []
        loc_answer = edit_item['loc_ans']
        loc_answer = [loc_answer]
        print("Specificity test")
        Specificity_metrics.append(check_answer_chat(model, 'Question: ' + edit_item['loc']+'\nPlease answer the question directly.',tokenizer,loc_answer,device,max_new_tokens=50))
        metrics['Specificity'] = Specificity_metrics
                
    for hop_num in range(2, 7):
        hop_key = f'{hop_num}_hops'
        if hop_key in edit_item and edit_item[hop_key]:
            hop_data = edit_item[hop_key][0]
            if hop_data.get('question') and hop_data.get('answer'):
                question = hop_data['question']
                answer = [hop_data['answer']]
                if hop_data.get('answer_alias'):
                    answer.extend(hop_data['answer_alias'])
                hop_accuracy = check_answer_chat(model, 'Question: ' + question+'\nPlease answer the question directly.',tokenizer,answer,device,max_new_tokens=50)
                metrics[f'{hop_num}_hops_acc'] = hop_accuracy

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
