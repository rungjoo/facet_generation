from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import json
from tqdm import tqdm
import argparse, logging

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if args.document:
        document = "document"
    else:
        document = ""
    if args.related:
        related = "related"
    else:
        related = ""
    if args.LLM:
        LLM = "LLM"
    else:
        LLM = ""
    task_name = f"{document}_{related}_{LLM}".strip("_")    
    
    model_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/save_models/facet/multi_task/{task_name}"
    save_path = f"../../result/multitask_{task_name}.json"
        
    model = BartForConditionalGeneration.from_pretrained(model_path)        
    model = model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open("../../data/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_result = {}
    for k, data in tqdm(test_data.items()):
        query = data['query']
        options_overall_label = data['options_overall_label']
        gt_facet_list = data['facet']
        
        f_string = f"[facet] {query}"

        inputs = tokenizer(f_string, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        inputs.to(device)    

        token_ids = model.generate(inputs["input_ids"], max_length=tokenizer.model_max_length)
        pred_facet_string = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        pred_facet_list = [x.strip() for x in pred_facet_string.split("|")]

        test_result[k] = {}
        test_result[k]['query'] = query
        test_result[k]['pred'] = [x for x in pred_facet_list if x.strip() != ""]
        test_result[k]['label'] = gt_facet_list
        test_result[k]['options_overall_label'] = options_overall_label        

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(test_result, f)
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    
    parser.add_argument('--document', action='store_true', help='train document')
    parser.add_argument('--related', action='store_true', help='train related')
    parser.add_argument('--LLM', action='store_true', help='train LLM')
        
    args = parser.parse_args()
    
    main()    