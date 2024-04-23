import json
import argparse

from collections import defaultdict
from tqdm import tqdm
import pdb

from openai import OpenAI
import google.generativeai as genai
GOOGLE_API_KEY="***"
genai.configure(api_key=GOOGLE_API_KEY)

def make_prompt(query, facet_list1, facet_list2, LLM_type):
    facet_str1 = ", ".join(facet_list1)
    facet_str2 = ", ".join(facet_list2)
    if LLM_type == "gemini":
        llm_input1 = f"""Facets refer to the sub-intents desired by the user who searched the query.
The following are facets about "{query}".
Which facets set is better? (without explanation)
A: {facet_str1}
B: {facet_str2}"""

        llm_input2 = f"""Facets refer to the sub-intents desired by the user who searched the query.
The following are facets about "{query}".
Which facets set is better? (without explanation)
A: {facet_str2}
B: {facet_str1}"""
    elif LLM_type == "gpt4":    
        llm_input1 = f"""Facets refer to the sub-intents desired by the user who searched the query.
The following are facets about "{query}".
Which facets set is better? (without explanation)
A: {facet_str1}
B: {facet_str2}
A or B?"""

        llm_input2 = f"""Facets refer to the sub-intents desired by the user who searched the query.
The following are facets about "{query}".
Which facets set is better? (without explanation)
A: {facet_str2}
B: {facet_str1}
A or B?"""    
    
    return llm_input1, llm_input2

def generator(llm_input, LLM_type, gemini_model, client):
    if LLM_type == "gemini":
        com = gemini_model.generate_content(llm_input)
        response = com.text 
    elif LLM_type == "gpt4":
        messages = [
            {"role": "user", "content": llm_input}
        ]

        com = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1
        )
        response = com.choices[0].message.content
    return response

def main():
    model1 = args.model1
    model2 = args.model2
    test_type = args.test_type
    LLM_type = args.LLM
    
    result_path1 = f"result_filter/{model1}.json"
    result_path2 = f"result_filter/{model2}.json"
    
    save_path = f"result_filter/compare.txt"
    
    """ gemini 로딩 """
    # Set up the model
    generation_config = {
      "temperature": 0.1,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 2048,
    }
    gemini_model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)    
    """ GPT4 로딩 """
    client = OpenAI(
        api_key='***',
    )    
        
    with open(result_path1, 'r', encoding='utf-8') as f:
        result1 = json.load(f)
    with open(result_path2, 'r', encoding='utf-8') as f:
        result2 = json.load(f)
        
    test_query_set = set()
    result = defaultdict(int)
    filter_result = defaultdict(int)
    with open("error.txt", "w") as f:
        for ind in tqdm(range(len(result1))):
            data1 = result1[str(ind)]
            data2 = result2[str(ind)]
            assert data1['query'] == data2['query']

            query = data1['query']
            options_overall_label = data1['options_overall_label']
            if test_type == "duplicate":
                pass
            else: # unique
                if query in test_query_set:
                    continue
                else:
                    test_query_set.add(query)

            facet_list1 = data1['pred']
            facet_list2 = data2['pred']

            llm_input1, llm_input2 = make_prompt(query, facet_list1, facet_list2, LLM_type)
            try:
                response = generator(llm_input1, LLM_type, gemini_model, client)
                if response[0] == "A":
                    result[model1] += 1
                    if options_overall_label >= 1:
                        filter_result[model1] += 1
                elif response[0] == "B":
                    result[model2] += 1
                    if options_overall_label >= 1:
                        filter_result[model2] += 1
                else:
                    result["ERROR"] += 1
                    if options_overall_label >= 1:
                        filter_result["ERROR"] += 1
                    f.write("####RESPONE FORMAT####\n")
                    f.write(f"{llm_input}\n")
                    f.write(f"{response}\n\n")
            except:
                result["ERROR"] += 1
                f.write("####GEMINI ERROR####\n")
                f.write(f"{llm_input1}\n\n")                    
                    
            ## gemini에 대해서는 입력 format 거꾸로해서 한 번 더함
            if LLM_type == "gemini":
                try:                    
                    response = generator(llm_input2, LLM_type, gemini_model, client)
                    if response[0] == "A":
                        result[model2] += 1
                        if options_overall_label >= 1:
                            filter_result[model2] += 1
                    elif response[0] == "B":
                        result[model1] += 1
                        if options_overall_label >= 1:
                            filter_result[model1] += 1
                    else:
                        result["ERROR"] += 1
                        if options_overall_label >= 1:
                            filter_result["ERROR"] += 1
                        f.write("####RESPONE FORMAT####\n")
                        f.write(f"{llm_input}\n")
                        f.write(f"{response}\n\n")                    
                except:                    
                    result["ERROR"] += 1
                    f.write(f"####{LLM_type} ERROR####\n")
                    f.write(f"{llm_input2}\n\n")
    
    m1_win, m2_win, er_num = result[model1], result[model2], result["ERROR"]
    fm1_win, fm2_win, fer_num = filter_result[model1], filter_result[model2], filter_result["ERROR"]
    with open(f"{save_path}" ,"a") as f:
        f.write(f"#############Test Type: {test_type}#############\n")
        f.write(f"#############LLM Type: {LLM_type}#############\n")
        f.write(f"{model1} win: {m1_win}, {model2} win: {m2_win}, ERROR: {er_num}\n\n")
        
        f.write("Filter Result - options_overall_label >= 1\n")
        f.write(f"{model1} win: {fm1_win}, {model2} win: {fm2_win}, ERROR: {fer_num}\n\n")
    
if __name__ == '__main__':
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--test_type", type=str, help = "model", default = 'duplicate')
    parser.add_argument( "--model1", type=str, help = "compared model")
    parser.add_argument( "--model2", type=str, help = "compared model")
    parser.add_argument( "--LLM", type=str, help = "LLM type")
    
    args = parser.parse_args()
    
    main()        