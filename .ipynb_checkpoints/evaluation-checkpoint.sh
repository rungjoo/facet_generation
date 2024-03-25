# python3 evaluation_LLM.py --model1 multitask_document --model2 LLM_upstage_llama-30b-instruct-2048_multitask_document --test_type unique
# python3 evaluation_LLM.py --model1 ictir_abstractive --model2 LLM_upstage_llama-30b-instruct-2048_multitask_document --test_type unique
# python3 evaluation_LLM.py --model1 structured --model2 LLM_upstage_llama-30b-instruct-2048_multitask_document --test_type unique

python3 evaluation_LLM.py --model1 LLM_upstage_llama-30b-instruct-2048_multitask_document --model2 multitask_document --test_type unique --LLM gpt4
python3 evaluation_LLM.py --model1 LLM_upstage_llama-30b-instruct-2048_multitask_document --model2 ictir_abstractive --test_type unique --LLM gpt4
python3 evaluation_LLM.py --model1 LLM_upstage_llama-30b-instruct-2048_multitask_document --model2 structured --test_type unique --LLM gpt4

# python3 evaluation.py --model_type LLM_upstage_llama-30b-instruct-2048_ictir_extractive --bert_type roberta-large --test_type unique
# python3 evaluation.py --model_type LLM_upstage_llama-30b-instruct-2048_ictir_classifier --bert_type roberta-large --test_type unique
# python3 evaluation.py --model_type LLM_upstage_llama-30b-instruct-2048_structured --bert_type roberta-large --test_type unique
# python3 evaluation.py --model_type LLM_upstage_llama-30b-instruct-2048_ictir_abstractive --bert_type roberta-large --test_type unique
