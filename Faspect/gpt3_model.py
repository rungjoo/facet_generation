import jsonlines

## gpt3
gpt3_facets = {}
with jsonlines.open("gpt3_facets.jsonl") as f:
    for line in f.iter():
        query = line['query']
        pred = line['facets']
        gpt3_facets[query] = pred
        
import json
with open("../result/baseline.json", 'r', encoding='utf-8') as f:
    baseline = json.load(f)
    
gpt3_facets_result = {}
for ind, data in baseline.items():
    query = data['query']
    label = data['label']
    pred = gpt3_facets[query]
    
    gpt3_facets_result[ind] = {}
    gpt3_facets_result[ind]['query'] = query
    gpt3_facets_result[ind]['pred'] = pred
    gpt3_facets_result[ind]['label'] = label
    
with open("../result/gpt3_facets.json", "w", encoding='utf-8') as f:
    json.dump(gpt3_facets_result, f)    