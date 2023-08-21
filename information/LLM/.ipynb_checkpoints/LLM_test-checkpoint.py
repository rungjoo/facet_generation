import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import pdb
import json
from tqdm import tqdm

data_path = "../../data/train.json"
with open(data_path, "r") as f:
    dataset = json.load(f)

model_path = "/home1/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/upstage/llama-30b-instruct-2048" # "upstage/llama-30b-instruct-2048"
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    # rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

llm_train = {}    
for ind, data in tqdm(dataset.items()):
    query = data['query']
    facet_list = data['facet']
    llm_train[ind] = {}
    llm_train[ind]['query'] = query
    llm_train[ind]['facet_info'] = []
    
    for facet in facet_list:
        prompt = f"### User:\nThe facet for '{query}' is '{facet}'. To create this facet, please create the necessary information in one sentence.\n\n### Assistant:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, use_cache=True, max_new_tokens=float('inf'), temperature=0)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        info = output[len(prompt):]
        
        facet_info = [facet, info]
        llm_train[ind]['facet_info'].append(facet_info)

with open("LLM_train.json", "w", encoding='utf-8') as f:
    json.dump(llm_train, f, indent="\t")