#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install seqeval==0.0.13')


# In[ ]:


from collections import Counter
import numpy as np
import sys
import numpy as np
from scipy import stats
from seqeval.metrics import f1_score, precision_score, recall_score


# In[ ]:


def eval_sentence(words, y_pred):
    # This function based on the public code: https://github.com/SVAIGBA/WMSeg/blob/master/wmseg_eval.py
    seg_pred = []

    word_pred = ''
    y_pred_word = []
    for y_pred_label in y_pred:
        y_pred_word.append(y_pred_label[0])

    for i in range(len(y_pred_word)):
        if word_pred != '': word_pred += '' + words[i]
        else: word_pred += words[i]

        if y_pred_word[i][0] in ['S', 'E']:
            word_pos_pred = word_pred
            seg_pred.append(word_pos_pred.lower())
            word_pred = ''

    return seg_pred

def get_words_dict(path):
    lines = open(path, "r", encoding = "utf-8").read().strip().split("\n\n\n")
    sents, labels = [], []
    words_dict = dict()

    for line in lines:
        cur_sent, cur_label = [], []

        for i in line.split("\n"):
            i_list = i.split("\t")
            if i == '': continue
            i_w, i_l = i_list[0], i_list[1]
            cur_sent.append(i_w)
            cur_label.append(i_l)
        
        sents.append(cur_sent)
        labels.append(cur_label)
        words = eval_sentence(cur_sent, cur_label)
        for word in words:
            words_dict[word] = words_dict.get(word, 0) + 1
    
    return sents, labels, words_dict

def pos_evaluate_word_PRF(y_pred, y):
    # This function based on the public code: https://github.com/SVAIGBA/TwASP/blob/master/twasp_eval.py
    y_word = []
    y_pos = []
    y_pred_word = []
    y_pred_pos = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_pos.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    word_cor_num = 0
    pos_cor_num = 0
    yp_wordnum = y_pred_word.count('E')+y_pred_word.count('S')
    yt_wordnum = y_word.count('E')+y_word.count('S')
    start = 0
    for i in range(len(y_word)):
        if y_word[i] == 'E' or y_word[i] == 'S':
            word_flag = True
            pos_flag = True
            for j in range(start, i+1):
                if y_word[j] != y_pred_word[j]:
                    word_flag = False
                    pos_flag = False
                    break
                if y_pos[j] != y_pred_pos[j]:
                    pos_flag = False
            if word_flag:
                word_cor_num += 1
            if pos_flag:
                pos_cor_num += 1
            start = i+1

    wP = word_cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    wR = word_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    try:
        wF = 2 * wP * wR / (wP + wR)
    except:
        wF = 0

    pP = precision_score(y, y_pred)
    pR = recall_score(y, y_pred)
    pF = f1_score(y, y_pred)

    return (wP, wR, wF), (pP, pR, pF)

def significance_test(Gold_label, TwASP_label, Our_label, alpha = 0.05):
    # This function based on the public code: https://github.com/rtmdrr/testSignificanceNLP/blob/master/testSignificance.py

    TwASP_Seg, TwASP_Tag, Our_Seg, Our_Tag = [], [], [], []
    for Gold_label_i, TwASP_label_i, Our_label_i in zip(Gold_label, TwASP_label, Our_label):
        (_, _, wF_TwASP), (_, _, pF_TwASP) = pos_evaluate_word_PRF(TwASP_label_i, Gold_label_i)
        TwASP_Seg.append(wF_TwASP)
        TwASP_Tag.append(pF_TwASP)
        (_, _, wF_Our), (_, _, pF_Our) = pos_evaluate_word_PRF(Our_label_i, Gold_label_i)
        Our_Seg.append(wF_Our)
        Our_Tag.append(pF_Our)

    
    t_seg_results = stats.ttest_rel(TwASP_Seg, Our_Seg)
    # correct for one sided test
    pval = float(t_seg_results[1]) / 2
    if (float(pval) <= float(alpha)):
        print("\nTest result for Seg is significant with p-value: {}".format(pval))
    else:
        print("\nTest result for Seg is not significant with p-value: {}".format(pval))

    t_tag_results = stats.ttest_rel(TwASP_Tag, Our_Tag)
    # correct for one sided test
    pval = float(t_tag_results[1]) / 2
    if (float(pval) <= float(alpha)):
        print("\nTest result for Tag is significant with p-value: {}".format(pval))
    else:
        print("\nTest result for Tag is not significant with p-value: {}".format(pval))


# In[ ]:


best_dimensions = {'CTB6': '400', 'CTB7': '300', 'CTB9': '400'} # The dimension is chose from Table 3 of our paper

for dataset_name in best_dimensions: 
    print(19*'*')
    print(f'Significant test on {dataset_name} dataset')
    _, Gold_label, _ = get_words_dict(f"../data/{dataset_name}/test.tsv")
    _, TwASP_ZEN_prediction, _ = get_words_dict(f"./TwASP_prediction/{dataset_name}_ZEN_test.txt")
    _, our_BERT_prediction, _ = get_words_dict(f"./our_prediction/{dataset_name}_BERT_{best_dimensions[dataset_name]}_test.txt")
    significance_test(Gold_label, TwASP_ZEN_prediction, our_BERT_prediction)
    print(19*'*')

