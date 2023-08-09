""" SERP loading """
import jsonlines
print("SERP loading")
dataset = []
with jsonlines.open("MIMICS-BingAPI.jsonl") as f:
    for line in f.iter():
        dataset.append(line)
        
import re
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

query_document = {}
for data in dataset:
    query = list(data.keys())[0]
    snippet_list = data[query]['snippet_list']
    related_list = data[query]['related_list']
    query_document[query] = {}
    query_document[query]['snippet_list'] = [cleanhtml(x) for x in snippet_list]
    query_document[query]['related_list'] = related_list
    
""" MIMICS loading """    
import glob, os
print("MIMICS loading")
folder = "../MIMICS/data/"
train_path = os.path.join(folder, "MIMICS-Click.tsv")
test_path = os.path.join(folder, "MIMICS-Manual.tsv")

import pandas as pd
tr_df = pd.read_csv(train_path, sep = '\t')
te_df = pd.read_csv(test_path, sep = '\t')

train_dataset = tr_df[['query', 'option_1', 'option_2', 'option_3', 'option_4', 'option_5']] # Click
test_dataset = te_df[['query', 'option_1', 'option_2', 'option_3', 'option_4', 'option_5', 'options_overall_label']] # Manual

""" Data filtering """
import math
print("Data filtering")
train_data = {}
for i in range(len(train_dataset)):
    data = train_dataset.iloc[i]
    query, option_1, option_2, option_3, option_4, option_5 = data['query'], data['option_1'], data['option_2'], data['option_3'], data['option_4'], data['option_5']    
    train_data[i] = {}
    train_data[i]['query'] = query
    options = [option_1, option_2, option_3, option_4, option_5]
    filter_options = []
    for option in options:
        if not (isinstance(option, float) and (math.isnan(option))):
            filter_options.append(option)    
    train_data[i]['facet'] = filter_options
    if query in query_document:
        train_data[i]['document'] = query_document[query]['snippet_list']
        train_data[i]['related'] = query_document[query]['related_list']
    
test_data = {}
for i in range(len(test_dataset)):
    data = test_dataset.iloc[i]
    query, option_1, option_2, option_3, option_4, option_5 = data['query'], data['option_1'], data['option_2'], data['option_3'], data['option_4'], data['option_5']    
    test_data[i] = {}
    test_data[i]['query'] = query
    options = [option_1, option_2, option_3, option_4, option_5]
    filter_options = []
    for option in options:
        if not (isinstance(option, float) and (math.isnan(option))):
            filter_options.append(option)
    test_data[i]['facet'] = filter_options
    
    options_overall_label = int(data['options_overall_label']) # Manual
    test_data[i]['options_overall_label'] = options_overall_label
    if query in query_document:
        test_data[i]['document'] = query_document[query]['snippet_list']
        test_data[i]['related'] = query_document[query]['related_list']
        
import json
with open("train.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent="\t")        
    
with open("test.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent="\t")    