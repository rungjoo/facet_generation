import json
import argparse

import itertools
from tqdm import tqdm

from evaluate import load
bertscore_func = load("bertscore")
from nltk.translate.bleu_score import sentence_bleu

def best_bleu_cand_ori(groundtruth, candidate):
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

def best_bleu_cand(groundtruth, candidate):
    copy_candidate = candidate[:]
    best_cand = []
    for facet in groundtruth:
        max_bleu = 0
        max_ind = 0
        for i in range(len(copy_candidate)):
            bleu = sentence_bleu([facet], copy_candidate[i])
            if bleu > max_bleu:
                max_bleu = bleu
                max_ind = i
        if len(copy_candidate) > 0:
            best_cand.append(copy_candidate[max_ind])
            copy_candidate.pop(max_ind)
    return best_cand+copy_candidate

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

def cal_mean(result_list):
    return sum(result_list)/len(result_list)

def main():
    model_type = args.model_type
    if model_type == 'baseline':
        result_path = "result/baseline.json"
        save_path = "result_baseline.txt"
    elif model_type == 'gpt3':
        result_path = "result/gpt3_facets.json"
        save_path = "result_gpt3.txt"
    else: ## ictir
        result_path = f"result/ictir_{model_type}.json"
        save_path = f"result_{model_type}.txt"
        
        
    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    exact_p_list, exact_r_list, exact_f1_list = [], [], []
    term_p_list, term_r_list, term_f1_list = [], [], []
    bleu_list1, bleu_list2, bleu_list3, bleu_list4 = [], [], [], []
    bert_p_list, bert_r_list, bert_f1_list = [], [], []
    
    filter_exact_p_list, filter_exact_r_list, filter_exact_f1_list = [], [], []
    filter_term_p_list, filter_term_r_list, filter_term_f1_list = [], [], []
    filter_bleu_list1, filter_bleu_list2, filter_bleu_list3, filter_bleu_list4 = [], [], [], []
    filter_bert_p_list, filter_bert_r_list, filter_bert_f1_list = [], [], []    
    for k, data in tqdm(result.items()):
        pred_list = data['pred']        
        label_list = data['label']
        options_overall_label = data['options_overall_label']
        
        pred_list = pred_list[:5]
        pred_list = best_bleu_cand_ori(label_list, pred_list)

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
        
        if len(pred_list) == 0:
            bert_p, bert_r, bert_f1 = 0, 0, 0
        else:
            bert_p, bert_r, bert_f1 = bertscore(label_list, pred_list)
        bert_p_list.append(bert_p)
        bert_r_list.append(bert_r)
        bert_f1_list.append(bert_f1)
        
        if options_overall_label >= 1:
            filter_exact_p_list.append(exact_p)
            filter_exact_r_list.append(exact_r)
            filter_exact_f1_list.append(exact_f1)
            
            filter_term_p_list.append(term_p)
            filter_term_r_list.append(term_r)
            filter_term_f1_list.append(term_f1)
            
            filter_bleu_list1.append(bleu1)
            filter_bleu_list2.append(bleu2)
            filter_bleu_list3.append(bleu3)
            filter_bleu_list4.append(bleu4)
            
            filter_bert_p_list.append(bert_p)
            filter_bert_r_list.append(bert_r)
            filter_bert_f1_list.append(bert_f1)            

    exact_p_score, exact_r_score, exact_f1_score = cal_mean(exact_p_list), cal_mean(exact_r_list), cal_mean(exact_f1_list)
    filter_exact_p_score, filter_exact_r_score, filter_exact_f1_score = cal_mean(filter_exact_p_list), cal_mean(filter_exact_r_list), cal_mean(filter_exact_f1_list)

    term_p_score, term_r_score, term_f1_score = cal_mean(term_p_list), cal_mean(term_r_list), cal_mean(term_f1_list)
    filter_term_p_score, filter_term_r_score, filter_term_f1_score = cal_mean(filter_term_p_list), cal_mean(filter_term_r_list), cal_mean(filter_term_f1_list)
    
    bleu_s1, bleu_s2, bleu_s3, bleu_s4 = cal_mean(bleu_list1), cal_mean(bleu_list2), cal_mean(bleu_list3), cal_mean(bleu_list4)
    filter_bleu_s1, filter_bleu_s2, filter_bleu_s3, filter_bleu_s4 =\
        cal_mean(filter_bleu_list1), cal_mean(filter_bleu_list2), cal_mean(filter_bleu_list3), cal_mean(filter_bleu_list4)
    
    bert_p_score, bert_r_score, bert_f1_score = cal_mean(bert_p_list), cal_mean(bert_r_list), cal_mean(bert_f1_list)
    filter_bert_p_score, filter_bert_r_score, filter_bert_f1_score = cal_mean(filter_bert_p_list), cal_mean(filter_bert_r_list), cal_mean(filter_bert_f1_list)
    
    with open(f"result/{save_path}" ,"a") as f:
        f.write("Term-overlapping\n")
        f.write(f"precision: {term_p_score}, recall: {term_r_score}, f1: {term_f1_score}\n")
        f.write("Exact-matching\n")
        f.write(f"precision: {exact_p_score}, recall: {exact_r_score}, f1: {exact_f1_score}\n")
        f.write("Blue-score\n")
        f.write(f"bleu1: {bleu_s1}, bleu2: {bleu_s2}, bleu3: {bleu_s3}, bleu4: {bleu_s4}\n")
        f.write("BERTScore\n")
        f.write(f"precision: {bert_p_score}, recall: {bert_r_score}, f1: {bert_f1_score}\n\n")
        
        f.write("Filter Result - options_overall_label >= 1\n")
        f.write("Term-overlapping\n")
        f.write(f"precision: {filter_term_p_score}, recall: {filter_term_r_score}, f1: {filter_term_f1_score}\n")
        f.write("Exact-matching\n")
        f.write(f"precision: {filter_exact_p_score}, recall: {filter_exact_r_score}, f1: {filter_exact_f1_score}\n")
        f.write("Blue-score\n")
        f.write(f"bleu1: {filter_bleu_s1}, bleu2: {filter_bleu_s2}, bleu3: {filter_bleu_s3}, bleu4: {filter_bleu_s4}\n")
        f.write("BERTScore\n")
        f.write(f"precision: {filter_bert_p_score}, recall: {filter_bert_r_score}, f1: {filter_bert_f1_score}\n")
    
if __name__ == '__main__':
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "facet generation" )
    parser.add_argument( "--model_type", type=str, help = "model", default = 'baseline')
    args = parser.parse_args()
    
    main()        