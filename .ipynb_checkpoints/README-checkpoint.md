## File Download
```
git clone https://github.com/microsoft/MIMICS
wget http://ciir.cs.umass.edu/downloads/mimics-serp/MIMICS-BingAPI-results.zip
```

# Data Preprocessing
## data/MIMICS-BingAPI.jsonl 
Make sub files from MIMICS-BingAPI-results
```
cat MIMICS-BingAPI-results | wc -l
split -n {line_num} MIMICS-BingAPI-results mimics_
```

## data/SERP_filter.py
Extract information from mimics_* and create MIMICS-BingAPI.jsonl
```
python3 SERP_filter.py
```

## data/data_preprocess.py
Create train.json and test.json in data folder
```
python3 data_preprocess.py
```

## document
Generate document from query

## related
Generate document from query

# Model Train

## baseline
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