import torch
import json
from tqdm import tqdm
import argparse, logging

def main():
    model_type = args.model_type    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    data_type = "train"
    save_path = f"../../data/{data_type}_pick.json"

    with open(f"../../data/{data_type}.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    """ 
        심플 방법 
        띄어쓰기로 토큰 구분
        소문자로 변환해서 가장 많이 겹치는 document 순으로 sorting
    """
    result = {}
    ind = 0 
    for k, data in tqdm(dataset.items()):
        query = data['query']
        query_tokens = query.split()
        try:
            all_document = data['document']
        except:
            all_document = []
        facet_list = data['facet']
        try:
            related = data['related']
        except:
            related = []
        
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
        
        pick_relateds = {}
        for related_string in related:
            related_string = related_string.lower()
            num = 0
            for facet_token in facet_tokens:
                if facet_token in related_string:
                    num += 2
            for query_token in query_tokens:
                if query_token in related_string:
                    num += 1
                    
            pick_relateds[document_string] = num
        pick_relateds = sorted(pick_relateds.items(), key=lambda x:x[1], reverse=True)
        
        # if len(all_document) > 0 and len(related) > 0:
        result[ind] = {}
        result[ind]['query'] = query
        result[ind]['facet'] = facet_list
        result[ind]['original_document'] = all_document
        result[ind]['pick_document'] = pick_documents
        result[ind]['original_related'] = related
        result[ind]['pick_related'] = pick_relateds
        result[ind]['related'] = related        
        ind += 1

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(result, f)               
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--model_type", type=str, help = "model", default = 'query_document_bart')
    args = parser.parse_args()
    
    main()    