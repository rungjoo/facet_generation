import json
import glob    
from tqdm import tqdm
""" 테스트할 쿼리 선정 """
with open("bartres_items_result.txt", "r") as f:
    dataset2 = f.readlines()    
comp_dataset = {}
for data in dataset2:
    query, result = data.split('\t')
    comp_dataset[query] = eval(result)    
    
""" 우리의 결과 쿼리 align """    
import glob
# test_result_list = glob.glob("result/*.json")
test_result_list = glob.glob("result/*noshot.json")
for data_path in tqdm(test_result_list):
    if data_path == 'result/gpt3_facets.json':
        continue
    else:
        with open(data_path, 'r') as f:
            dataset1 = json.load(f)       
        our_dataset = {}
        for ind, data in dataset1.items():
            query = data['query']
            our_dataset[query] = data
        """ 테스트할 파일로 저장 """
        save_path = data_path.replace("result", "result_filter")
        strucutred_dataset = {}
        new_our_dataset = {}
        for ind, (query, result) in enumerate(comp_dataset.items()):
            our_data = our_dataset[query]
            try:
                strucutred_dataset[ind] = {}
                strucutred_dataset[ind]['query'] = query
                strucutred_dataset[ind]['pred'] = result
                strucutred_dataset[ind]['label'] = our_data['label']
                strucutred_dataset[ind]['options_overall_label'] = our_data['options_overall_label']

                new_our_dataset[ind] = {}
                new_our_dataset[ind]['query'] = query
                new_our_dataset[ind]['pred'] = our_data['pred']
                new_our_dataset[ind]['label'] = our_data['label']
                new_our_dataset[ind]['options_overall_label'] = our_data['options_overall_label']
            except:
                import pdb
                pdb.set_trace()

        # with open("result_filter/structured.json", "w") as f:
        #     json.dump(strucutred_dataset, f, indent='\t')
        with open(save_path, "w") as f:
            json.dump(new_our_dataset, f, indent='\t')    