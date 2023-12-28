# python3 facet_generation_test.py --trained_model multitask_document_related --method post --model_name upstage/llama-30b-instruct-2048
# python3 facet_generation_test.py --trained_model multitask_document --method post --model_name upstage/llama-30b-instruct-2048
# python3 facet_generation_test.py --trained_model multitask_related --method post --model_name upstage/llama-30b-instruct-2048
# python3 facet_generation_test.py --trained_model multitask_related --method zero --model_name upstage/llama-30b-instruct-2048

# python3 facet_generation_test.py --trained_model multitask_document_related --method post --model_name HyperbeeAI/Tulpar-7b-v0
# python3 facet_generation_test.py --trained_model multitask_document --method post --model_name HyperbeeAI/Tulpar-7b-v0
# python3 facet_generation_test.py --trained_model multitask_related --method post --model_name HyperbeeAI/Tulpar-7b-v0
# python3 facet_generation_test.py --trained_model multitask_related --method zero --model_name HyperbeeAI/Tulpar-7b-v0

python3 facet_generation_test.py --trained_model multitask_document_related --method noshot --model_name upstage/llama-30b-instruct-2048
python3 facet_generation_test2.py --trained_model multitask_document_related --method noshot --model_name Open-Orca/OpenOrca-Platypus2-13B
python3 facet_generation_test.py --trained_model multitask_document_related --method noshot --model_name HyperbeeAI/Tulpar-7b-v0