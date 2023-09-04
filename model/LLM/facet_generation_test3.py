import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import argparse, logging
import re

def make_prompt(query, pred_facets, method):
    if method == "post":
        one_shot = """### User:\nThe predicted facets for 'caesars atlantic city' are 'parking, hotels'. But the correct facets are 'caesars atlantic city events, caesars atlantic city jobs, caesars atlantic city parking'\n"""
        two_shot = """The predicted facets for 'vista, ca' are 'parking, hotels'. But the correct facets are 'weather, zip code, population, homes for sale'\n\n"""
        prompt = one_shot + two_shot + f"""As in the example above, modify the predicted facets.\nThe predicted facets for '{query}' are '{pred_facets}'. What are the correct facets?\n\n### Assistant:\nThe correct facets for '{query}' are"""    
    else: # unseen
        one_shot = """### User:\nThe facets for 'caesars atlantic city' are 'caesars atlantic city events, caesars atlantic city jobs, caesars atlantic city parking'\n"""
        two_shot = """The facets for 'vista, ca' are 'weather, zip code, population, homes for sale'\n\n"""
        prompt = one_shot + two_shot + f"""### Assistant:\nThe correct facets for '{query}' are"""    
    
    return prompt

# def make_prompt(query, pred_facets, method):
#     if method == "post":
#         one_shot = """### Instruction:\nThe predicted facets for 'caesars atlantic city' are 'parking, hotels'. But the correct facets are 'caesars atlantic city events, caesars atlantic city jobs, caesars atlantic city parking'\n"""
#         two_shot = """The predicted facets for 'vista, ca' are 'parking, hotels'. But the correct facets are 'weather, zip code, population, homes for sale'\n\n"""
#         prompt = one_shot + two_shot + f"""As in the example above, modify the predicted facets.\nThe predicted facets for '{query}' are '{pred_facets}'. What are the correct facets?\n\n### Response:\nThe correct facets for '{query}' are"""    
#     else: # unseen
#         one_shot = """### Instruction:\nThe facets for 'caesars atlantic city' are 'caesars atlantic city events, caesars atlantic city jobs, caesars atlantic city parking'\n"""
#         two_shot = """The facets for 'vista, ca' are 'weather, zip code, population, homes for sale'\n\n"""
#         prompt = one_shot + two_shot + f"""### Response:\nThe correct facets for '{query}' are"""    
    
#     return prompt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    method = args.method
    model_name = args.model_name
    model_str = model_name.replace("/", "_")
    data_path = f"../../result/{args.trained_model}.json"
    
    if method == "post": ## see small model  
        save_path = f"../../result/LLM_{model_str}_{args.trained_model}.json"
        error_path = f"../../result/LLM_{model_str}_{args.trained_model}_error.json"
    else: ## zero (unseen)
        save_path = f"../../result/LLM_{model_str}_{method}.json"
        error_path = f"../../result/LLM_{model_str}_{method}_error.json"
        
    with open(data_path, "r") as f:
        dataset = json.load(f)

    model_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/{model_name}" # "upstage/llama-30b-instruct-2048"
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    
    eng_rule = re.compile('\'.+\'')    
    test_result = {}
    error_result = {}
    for k, data in tqdm(dataset.items()):
        query = data['query']
        pred_facet_list = data['pred']
        pred_facets = ", ".join(pred_facet_list)
        label = data['label']
        options_overall_label = data['options_overall_label']
        
        prompt = make_prompt(query, pred_facets, method)
        # one_shot = """### User:\nThe predicted facets for 'caesars atlantic city' are 'parking, hotels'. But the correct facets are 'caesars atlantic city events, caesars atlantic city jobs, caesars atlantic city parking'\n"""
        # two_shot = """The predicted facets for 'vista, ca' are 'parking, hotels'. But the correct facets are 'weather, zip code, population, homes for sale'\n\n"""
        # prompt = one_shot + two_shot + f"""As in the example above, modify the predicted facets.\nThe predicted facets for '{query}' are '{pred_facets}'. What are the correct facets?\n\n### Assistant:\nThe correct facets for '{query}' are"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        output = model.generate(**inputs, use_cache=True, max_new_tokens=100, temperature=0.001, top_p=1)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        correct_facets = output[len(prompt):]
        
        try:
            # correct_facet_list = [x.strip() for x in correct_facets.strip().split("\n")[0].strip("'").strip(".").strip("'").split(",") if x.strip() != ""]
            correct_facet_list = [x.strip() for x in matches[0].strip("'").split(",") if x.strip() != ""]
            test_result[k] = {}
            test_result[k]['query'] = query
            test_result[k]['pred'] = correct_facet_list
            test_result[k]['label'] = label
            test_result[k]['options_overall_label'] = options_overall_label
        except:
            error_result[k] = {}
            error_result[k]['query'] = query
            error_result[k]['pred'] = correct_facets
            error_result[k]['label'] = label
            error_result[k]['options_overall_label'] = options_overall_label            

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(test_result, f)
    with open(error_path, "w", encoding='utf-8') as f:
        json.dump(error_result, f)
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--trained_model", type=str, help = "trained_model", default = "multitask_document_related")
    parser.add_argument( "--method", type=str, help = "post or zero", default = "post") # 1e-5
    parser.add_argument( "--model_name", type=str, help = "LLM name", default = "upstage/llama-30b-instruct-2048") # 1e-5
                    
    args = parser.parse_args()
    
    main()    