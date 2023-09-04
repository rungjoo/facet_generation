python3 facet_generation_test.py --trained_model multitask_document_related --method post --model_name upstage/llama-30b-instruct-2048
python3 facet_generation_test.py --trained_model multitask_document --method post --model_name upstage/llama-30b-instruct-2048
python3 facet_generation_test.py --trained_model multitask_related --method post --model_name upstage/llama-30b-instruct-2048
python3 facet_generation_test.py --trained_model multitask_related --method zero --model_name upstage/llama-30b-instruct-2048

python3 facet_generation_test.py --trained_model multitask_document_related --method post --model_name JoSw-14/LoKuS-13B
python3 facet_generation_test.py --trained_model multitask_document --method post --model_name JoSw-14/LoKuS-13B
python3 facet_generation_test.py --trained_model multitask_related --method post --model_name JoSw-14/LoKuS-13B
python3 facet_generation_test.py --trained_model multitask_related --method zero --model_name JoSw-14/LoKuS-13B