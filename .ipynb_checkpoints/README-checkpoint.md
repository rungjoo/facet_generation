## File Download
```
git clone https://github.com/microsoft/MIMICS
wget http://ciir.cs.umass.edu/downloads/mimics-serp/MIMICS-BingAPI-results.zip
```

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