import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
from tqdm import tqdm

data_type = "train"
data_path = f"../../data/{data_type}.json"
with open(data_path, "r") as f:
    dataset = json.load(f)

model_path = "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/HyperbeeAI/Tulpar-7b-v0"
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)


for ind, data in tqdm(dataset.items()):
    query = data['query']
    facet_list = data['facet']

    llm_data = {}
    llm_data['query'] = query
    llm_data['facet_info'] = []

    for facet in facet_list:
        prompt = f"### User:\nThe facet for '{query}' is '{facet}'. Tell me the rationale for it in one sentence.\n\n### Assistant:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # output = model.generate(**inputs, use_cache=True, max_new_tokens=float('inf'), temperature=0)
        output = model.generate(**inputs, use_cache=True, max_new_tokens=100, temperature=0.01)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        info = output[len(prompt):]

        facet_info = [facet, info]
        llm_data['facet_info'].append(facet_info)

    with open(f"LLM_{data_type}_7B.jsonl", "a", encoding='utf-8') as f:
        json.dump(llm_data, f)
        f.write("\n")