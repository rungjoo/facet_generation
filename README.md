## File Download
```
git clone https://github.com/microsoft/MIMICS
mkdir data
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
python3 generate_information7B.py
python3 construct_train_dataset.py
```

## pick information document
Generate document from query

# Model Train

## query
- input: query
- output: facet
```
cd model/query
python3 facet_generation_train.py --batch 4 --epoch 10
```

## query_documet
- input: query+documet
- output: facet
```
cd model/query_document
python3 facet_generation_train.py
```

## query_related
- input: query+related
- output: facet
```
cd model/query_related
python3 facet_generation_train.py
```

## multi-task
- input: query
- output: facet / document / related
```
cd model/multi_task
python3 facet_generation_train.py --args
```

# LLM post-processing
- input: generated facets
- output: re-generated facets
```
cd model/LLM
python3 facet_generation_test.py --args
```

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