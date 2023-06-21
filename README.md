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
Create train.json and test.json
```
python3 data_preprocess.py
```

# Train
```
python3 facet_generation_train.py --model_type {type}
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