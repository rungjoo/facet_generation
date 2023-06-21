import json
import argparse

import itertools
from tqdm import tqdm

from evaluate import load
bertscore_func = load("bertscore")
from nltk.translate.bleu_score import sentence_bleu

def best_bleu_cand(groundtruth, candidate):
    # assert len(groundtruth) >= len(candidate)
    all_permutations = list(itertools.permutations(candidate))
    max_bleu = 0.
    best_cand = all_permutations[0]
    for cand in all_permutations:
        bleu = 0.
        for i in range(min(len(groundtruth), len(cand))):
            bleu += sentence_bleu([groundtruth[i]], cand[i]) / len(groundtruth)
        if bleu > max_bleu:
            max_bleu = bleu
            best_cand = cand
    return list(best_cand)


def eval_bleu(groundtruth, cand):
    # Calculates the SET BLEU metrics, for 1-gram, 2-gram, 3-gram and 4-gram overlaps
    best_cand = best_bleu_cand(groundtruth, cand)
    bleu = [0., 0., 0., 0.]
    bleu_weights = [[1, 0, 0, 0], [0.5, 0.5, 0, 0], [0.33, 0.33, 0.33, 0], [0.25, 0.25, 0.25, 0.25]]
    for j in range(4):
        for i in range(min(len(groundtruth), len(best_cand))):
            bleu[j] += sentence_bleu([groundtruth[i]], best_cand[i], weights=bleu_weights[j]) / len(groundtruth)
    return bleu


def bertscore(groundtruth, cand):
    # Calculates the Set BERT-Score metrics for Precision, Recall & F1
    best_cand = best_bleu_cand(groundtruth, cand)
    if len(groundtruth) > len(best_cand):
        groundtruth = groundtruth[:len(best_cand)]
    else:
        best_cand = best_cand[:len(groundtruth)]
    results = bertscore_func.compute(predictions=best_cand, references=groundtruth, lang="en", device="cuda:0")
    precision, recall, f1 = results['precision'], results['recall'], results['f1']
    P, R, F = sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1)
    return P, R, F


def exact_match(groundtruth, cand):
    # Calculates the exact match Precision, Recall & F1
    c = 0.
    for x in cand:
        if x != '' and x in groundtruth:
            c += 1
    p = c / (len([x for x in cand if x != ''])+1e-8)
    r = c / (len([x for x in groundtruth if x != ''])+1e-8)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [p, r, f1]


def term_match(groundtruth, cand):
    # Calculates the term overlap Precision, Recall & F1
    gt_terms = set([])
    for x in groundtruth:
        if x == '':
            continue
        for t in x.strip().split():
            gt_terms.add(t)
    cand_terms = set([])
    for x in cand:
        if x == '':
            continue
        for t in x.strip().split():
            cand_terms.add(t)

    c = 0.
    for x in cand_terms:
        if x != '' and x in gt_terms:
            c += 1
    p = c / (len([x for x in cand_terms if x != ''])+1e-8)
    r = c / (len([x for x in gt_terms if x != ''])+1e-8)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [p, r, f1]


def main():
    model_type = args.model_type
    if model_type == 'baseline':
        result_path = "result/baseline.json"
        save_path = "result_baseline.txt"
    elif model_type == 'gpt3':
        result_path = "result/gpt3_facets.json"
        save_path = "result_gpt3.txt"
    elif model_type == "round_robin":
        result_path = "result/ictir_round_robin.json"
        save_path = "result_round_robin.txt"
    elif model_type == "abstractive":
        result_path = "result/ictir_abstractive.json"
        save_path = "abstractive.txt"
        
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    exact_p_list, exact_r_list, exact_f1_list = [], [], []
    term_p_list, term_r_list, term_f1_list = [], [], []
    bleu_list1, bleu_list2, bleu_list3, bleu_list4 = [], [], [], []
    bert_p_list, bert_r_list, bert_f1_list = [], [], []    
    for k, data in tqdm(result.items()):
        pred_list = data['pred']
        label_list = data['label']
        
        pred_list = best_bleu_cand(label_list, pred_list)

        exact_p, exact_r, exact_f1 = exact_match(label_list, pred_list)
        exact_p_list.append(exact_p)
        exact_r_list.append(exact_r)
        exact_f1_list.append(exact_f1)

        term_p, term_r, term_f1 = term_match(label_list, pred_list)
        term_p_list.append(term_p)
        term_r_list.append(term_r)
        term_f1_list.append(term_f1)
        
        bleu1, bleu2, bleu3, bleu4 = eval_bleu(label_list, pred_list)
        bleu_list1.append(bleu1)
        bleu_list2.append(bleu2)
        bleu_list3.append(bleu3)
        bleu_list4.append(bleu4)
        
        bert_p, bert_r, bert_f1 = bertscore(label_list, pred_list)
        bert_p_list.append(bert_p)
        bert_r_list.append(bert_r)
        bert_f1_list.append(bert_f1)

    exact_p_score = sum(exact_p_list)/len(exact_p_list)
    exact_r_score = sum(exact_r_list)/len(exact_r_list)
    exact_f1_score = sum(exact_f1_list)/len(exact_f1_list)

    term_p_score = sum(term_p_list)/len(term_p_list)
    term_r_score = sum(term_r_list)/len(term_r_list)
    term_f1_score = sum(term_f1_list)/len(term_f1_list)
    
    bleu_s1 = sum(bleu_list1)/len(bleu_list1)
    bleu_s2 = sum(bleu_list2)/len(bleu_list2)
    bleu_s3 = sum(bleu_list3)/len(bleu_list3)
    bleu_s4 = sum(bleu_list4)/len(bleu_list4)
    
    bert_p_score = sum(bert_p_list)/len(bert_p_list)
    bert_r_score = sum(bert_r_list)/len(bert_r_list)
    bert_f1_score = sum(bert_f1_list)/len(bert_f1_list)
    
    with open(f"result/{save_path}" ,"w") as f:
        f.write("Term-overlapping\n")
        f.write(f"precision: {term_p_score}, recall: {term_r_score}, f1: {term_f1_score}\n")
        f.write("Exact-matching\n")
        f.write(f"precision: {exact_p_score}, recall: {exact_r_score}, f1: {exact_f1_score}\n")
        f.write("Blue-score\n")
        f.write(f"bleu1: {bleu_s1}, bleu2: {bleu_s2}, bleu3: {bleu_s3}, bleu4: {bleu_s4}\n")
        f.write("BERTScore\n")
        f.write(f"precision: {bert_p_score}, recall: {bert_r_score}, f1: {bert_f1_score}\n")
    
if __name__ == '__main__':
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--model_type", type=str, help = "model", default = 'baseline')
    args = parser.parse_args()
    
    main()        