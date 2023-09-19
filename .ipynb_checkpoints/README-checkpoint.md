## File Download (수정)
```
git clone https://github.com/microsoft/MIMICS
cd data
wget http://ciir.cs.umass.edu/downloads/mimics-serp/MIMICS-BingAPI-results.zip --no-check-certificate
```

# Data Preprocessing
## split data
Make sub files from MIMICS-BingAPI-results
```
cd data
unzip MIMICS-BingAPI-results.zip
cat MIMICS-BingAPI.result | wc -l # 479807
split -l 48000 MIMICS-BingAPI.result mimics_
```

## data/SERP_filter.py
Extract information from mimics_* and create MIMICS-BingAPI.jsonl
```
cd data
python3 SERP_filter.py
```

## data/data_preprocess.py
Create train.json and test.json in data folder
```
cd data
python3 data_preprocess.py
```

## rationale
Generate rationale from query and facet
```
cd information/LLM
python3 generate_information.py
python3 construct_train_dataset.py
```

## pick information document
Generate document from query

## related
Generate document from query

# Model Train

## query (baseline)
- input: query
- output: facet
```
python3 facet_generation_train.py --model_type {type}
```

## query_documet
- input: query+documet
- output: facet

## query_related
- input: query+related
- output: facet

# Test
All reulsts are included in result folder
```
python3 test.py --model_type {type}
```

# Evaluation
For evaluation
```
python3 evaluation.py --model_type {type}
```