from tqdm import tqdm
import json
import glob
import pdb
import json

file_list = glob.glob("mimics_*")
for file_path in file_list:
    print(file_path)
    
    f = open(file_path)
    dataset = f.readlines()
    f.close()

    query_document = {}
    for k, data in enumerate(tqdm(dataset)):
        query_data = {}
        data = data.replace("true", "True")
        data = data.replace("false", "False")
        data = data.replace("null", "None")
        try:
            json_data = eval(data)
        except:
            pdb.set_trace()

        query = json_data['queryContext']['originalQuery']
        try:
            snippet_list = [x['snippet'] for x in json_data['webPages']['value']]
        except:
            snippet_list = []
        query_data[query] = snippet_list
        
        with open("MIMICS-BingAPI.jsonl", "a", encoding="utf-8") as f:
            json.dump(query_data, f)
            f.write("\n")