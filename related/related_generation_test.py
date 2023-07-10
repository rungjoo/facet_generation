import os
import pdb
import argparse, logging
from tqdm import tqdm

import json
import torch
from transformers import get_linear_schedule_with_warmup

from dataloader import data_loader
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BartForConditionalGeneration
    
## finetune gpt2
def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    """logging and path"""    
    save_path = "result/test_related.json"
    log_path = "test.log"
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)
    
    """Model Loading"""
    model_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/save_models/facet/related/bart"
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model = model.cuda()
    model.eval()
    
    tokenizer_path = model_path # "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    """Test Setting"""
    with open("../data/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_result = {}
    for k, data in tqdm(test_data.items()):
        query = data['query']
        options_overall_label = data['options_overall_label']
        try:
            gt_related = data['related']
        except:
            gt_related = []

        inputs = tokenizer(query, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        inputs.to(device)
        
#         https://huggingface.co/transformers/v4.12.5/_modules/transformers/generation_utils.html
        token_ids = model.generate(
            input_ids=inputs["input_ids"], 
            max_length=tokenizer.model_max_length
        )
        pred_related_string = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        pred_related_list = [x.strip() for x in pred_related_string.split("|")]

        test_result[k] = {}
        test_result[k]['query'] = query
        test_result[k]['pred_related'] = [x for x in pred_related_list if x.strip() != ""]
        test_result[k]['label_related'] = gt_related
        test_result[k]['options_overall_label'] = options_overall_label
        
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(test_result, f)        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "snippet" )
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()