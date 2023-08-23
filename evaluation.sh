# python3 evaluation.py --model_type baseline
# python3 evaluation.py --model_type baseline_document_label
# python3 evaluation.py --model_type query_document_label
# python3 evaluation.py --model_type query_document_pred
# python3 evaluation.py --model_type query_document_none
# python3 evaluation.py --model_type query_related_label
# python3 evaluation.py --model_type query_related_pred
# python3 evaluation.py --model_type query_document_pick
# python3 evaluation.py --model_type query_pick_document_label
python3 evaluation.py --model_type multitask

# python3 evaluation.py --model_type ictir_abstractive
# python3 evaluation.py --model_type ictir_abstractive_query
# python3 evaluation.py --model_type ictir_extractive
# python3 evaluation.py --model_type ictir_classifier
# python3 evaluation.py --model_type ictir_mmr
# python3 evaluation.py --model_type ictir_rank
# python3 evaluation.py --model_type ictir_round_robin

# python3 evaluation.py --model_type baseline --test_type unique
# python3 evaluation.py --model_type baseline_document_label --test_type unique
# python3 evaluation.py --model_type query_document_label --test_type unique
# python3 evaluation.py --model_type query_document_pred --test_type unique
# python3 evaluation.py --model_type query_document_none --test_type unique
# python3 evaluation.py --model_type query_related_label --test_type unique
# python3 evaluation.py --model_type query_related_pred --test_type unique
# python3 evaluation.py --model_type query_document_pick --test_type unique
# python3 evaluation.py --model_type query_pick_document_label --test_type unique
python3 evaluation.py --model_type multitask --test_type unique

# python3 evaluation.py --model_type ictir_abstractive --test_type unique
# python3 evaluation.py --model_type ictir_abstractive_query --test_type unique
# python3 evaluation.py --model_type ictir_extractive --test_type unique
# python3 evaluation.py --model_type ictir_classifier --test_type unique
# python3 evaluation.py --model_type ictir_mmr --test_type unique
# python3 evaluation.py --model_type ictir_rank --test_type unique
# python3 evaluation.py --model_type ictir_round_robin --test_type unique