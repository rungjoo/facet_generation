import json
from tqdm import tqdm

size = '7B'
with open(f"LLM_train_{size}.jsonl") as f:
    facet_dataset = f.readlines()
    
facet_dict = {}
for line in facet_dataset:
    data = json.loads(line)
    query = data['query']
    facet_dict[query] = data['facet_info']
    
with open("../../data/train.json", "r") as f:
    original_dataset = json.load(f)    
    
original_dict = {}
for ind, data in original_dataset.items():
    query = data['query']
    original_dict[query] = data
    
new_train_dataset = {}
ind = 0
for query, facet_info in tqdm(facet_dict.items()):
    original_data = original_dict[query]
    
    new_train_dataset[ind] = original_data
    new_train_dataset[ind]['rationale'] = facet_info
    
    ind += 1
    
with open(f"../../data/train_rationale_{size}.json", "w") as f:
    json.dump(new_train_dataset, f)    