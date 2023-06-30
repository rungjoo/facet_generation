from faspect import Faspect
import json
from tqdm import tqdm
import argparse

def main():
    model_type = args.model_type
    facet_extractor = Faspect()

    with open("../data/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    round_robin_result = {}    
    for ind, data in tqdm(test_data.items()):
        query = data['query']
        label = data['facet']
        options_overall_label = data['options_overall_label']
        try:
            documents = data['document']
            pred_facets = facet_extractor.extract_facets(query, 
                                                    documents,
                                                    aggregation=model_type, # mmr, round_robin, rank, abstractive, abstractive_query, extractive, classifier
                                                    mmr_lambda=0.5,
                                                    classification_threshold=0.05,
                                                    classification_topk=0)
        except:
            documents = None
            pred_facets = []

        round_robin_result[ind] = {}
        round_robin_result[ind]['query'] = query
        round_robin_result[ind]['pred'] = pred_facets
        round_robin_result[ind]['label'] = label
        round_robin_result[ind]['options_overall_label'] = options_overall_label        


    with open(f"../result/ictir_{model_type}.json", "w") as f:
        json.dump(round_robin_result, f)
    
if __name__ == '__main__':
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--model_type", type=str, help = "model", default = 'round_robin')
    args = parser.parse_args()
    
    main()            