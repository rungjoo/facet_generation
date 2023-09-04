from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import json
from tqdm import tqdm
import argparse, logging

def main():
    model_type = args.model_type
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    related_type = args.related_type
    if model_type == "query_related_bart":
        model_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/save_models/facet/query_related_bart"
        save_path = f"../../result/query_related_{related_type}.json"
        
    model = BartForConditionalGeneration.from_pretrained(model_path)        
    model = model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open("../../data/merge_test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_result = {}
    for k, data in tqdm(test_data.items()):
        query = data['query']
        if related_type == "label":
            related_string = "|".join(data['label_related'])
        elif related_type == "pred":
            related_string = "|".join(data['pred_related'])
        elif related_type == "none":
            related_string = ""            
            
        input_string = f"{query}:{related_string}"
        options_overall_label = data['options_overall_label']
        gt_facet_list = data['facet']

        inputs = tokenizer(input_string, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        inputs.to(device)    

        token_ids = model.generate(inputs["input_ids"], max_length=tokenizer.model_max_length)
        pred_facet_string = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        pred_facet_list = [x.strip() for x in pred_facet_string.split(",")]

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
    parser.add_argument( "--model_type", type=str, help = "model", default = 'query_related_bart')
    parser.add_argument( "--related_type", type=str, help = "related", default = 'label')
    args = parser.parse_args()
    
    main()    