from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import json
from tqdm import tqdm
import argparse, logging

def main():
    model_type = args.model_type    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    data_type = "train"
    save_path = f"result/{data_type}_pick_documet.json"
    
#     model_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/save_models/facet/query_document_bart"
#     model = BartForConditionalGeneration.from_pretrained(model_path)        
#     model = model.cuda()
#     model.eval()
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(f"../data/{data_type}.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    """ 심플 방법 """
    result = {}
    for k, data in tqdm(dataset.items()):
        query = data['query']
        query_tokens = query.split()
        try:
            all_document = data['document']
        except:
            all_document = []
        facet_list = data['facet']
        
        facet_tokens = set()
        for facet in facet_list:
            for token in facet.split():
                facet_tokens.add(token.lower())
        facet_tokens = list(facet_tokens)
        
        for query_token in query_tokens:
            try:
                facet_tokens.remove(query_token)
            except:
                pass
        
        pick_documents = {}
        for document_string in all_document:
            document_string = document_string.lower()
            num = 0
            for facet_token in facet_tokens:
                if facet_token in document_string:
                    num += 2
            for query_token in query_tokens:
                if query_token in document_string:
                    num += 1
                    
            pick_documents[document_string] = num
        pick_documents = sorted(pick_documents.items(), key=lambda x:x[1], reverse=True)
        
        result[k] = {}
        result[k]['query'] = query
        result[k]['facet'] = facet_list
        result[k]['original_document'] = all_document
        result[k]['pick_document'] = pick_documents

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(result, f)
        
#         for document_string in all_document:
#             input_string = f"{query}:{document_string}"
#             options_overall_label = data['options_overall_label']
#             gt_facet_list = data['facet']

#             inputs = tokenizer(input_string, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
#             inputs.to(device)    

#             token_ids = model.generate(inputs["input_ids"], max_length=tokenizer.model_max_length)
#             pred_facet_string = tokenizer.decode(token_ids[0], skip_special_tokens=True)
#             pred_facet_list = [x.strip() for x in pred_facet_string.split(",")]        
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--model_type", type=str, help = "model", default = 'query_document_bart')
    args = parser.parse_args()
    
    main()    