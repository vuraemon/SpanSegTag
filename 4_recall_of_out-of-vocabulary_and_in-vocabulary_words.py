#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install seqeval==0.0.13')


# In[ ]:


from collections import Counter
import numpy as np
import sys
from seqeval.metrics import f1_score, precision_score, recall_score


# In[ ]:


def pos_evaluate(y_pred_list, y_list, sentence_list, word2id, oovivtype):
    # This fuction based on the public code https://github.com/SVAIGBA/TwASP/blob/master/twasp_eval.py

    word_cor_num = 0
    pos_cor_num = 0
    yt_wordnum = 0

    y_word_list = []
    y_pos_list = []
    y_pred_word_list = []
    y_pred_pos_list = []
    for y_label, y_pred_label in zip(y_list, y_pred_list):
        y_word = []
        y_pos = []
        y_pred_word = []
        y_pred_pos = []
        for y_l in y_label:
            y_word.append(y_l[0])
            y_pos.append(y_l[2:])
        for y_pred_l in y_pred_label:
            y_pred_word.append(y_pred_l[0])
            y_pred_pos.append(y_pred_l[2:])
        y_word_list.append(y_word)
        y_pos_list.append(y_pos)
        y_pred_word_list.append(y_pred_word)
        y_pred_pos_list.append(y_pred_pos)

    for y_w, y_p, y_p_w, y_p_p, sentence in zip(y_word_list, y_pos_list, y_pred_word_list, y_pred_pos_list, sentence_list):
        start = 0
        for i in range(len(y_w)):
            if y_w[i] == 'E' or y_w[i] == 'S':
                word = ''.join(sentence[start:i+1])
                if oovivtype == 'iv':
                    if word not in word2id:
                        start = i + 1
                        continue
                if oovivtype == 'oov':
                    if word in word2id:
                        start = i + 1
                        continue
                word_flag = True
                pos_flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y_w[j] != y_p_w[j]:
                        word_flag = False
                        pos_flag = False
                        break
                    if y_p[j] != y_p_p[j]:
                        pos_flag = False
                if word_flag:
                    word_cor_num += 1
                if pos_flag:
                    pos_cor_num += 1
                start = i + 1

    word_OOV = word_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    pos_OOV = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1

    return np.round(100*word_OOV,2), np.round(100*pos_OOV,2)

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


# In[ ]:


best_dimensions = {'CTB5': '500', 'CTB6': '400', 'CTB7': '300', 'CTB9': '400', 'UD1': '300'} # The dimension is chose from Table 3 of our paper

for dataset_name in best_dimensions: 
    print(19*'*')
    print(f'Recall of Out-of-vocabulary and in-vocabulary words on {dataset_name} dataset')
    _, _, train_words_dict = get_words_dict(f"../data/{dataset_name}/train.tsv")

    test_sent, Gold_label, _ = get_words_dict(f"../data/{dataset_name}/test.tsv")

    _, TwASP_BERT_prediction, _ = get_words_dict(f"./TwASP_prediction/{dataset_name}_BERT_test.txt")
    _, word_pos_oov = pos_evaluate(TwASP_BERT_prediction, Gold_label, test_sent, train_words_dict, 'oov')
    _, word_pos_iv = pos_evaluate(TwASP_BERT_prediction, Gold_label, test_sent, train_words_dict, 'iv')
    print(f'TwASP_BERT_prediction: word_pos_oov = {word_pos_oov}, word_pos_iv = {word_pos_iv}')

    _, TwASP_ZEN_prediction, _ = get_words_dict(f"./TwASP_prediction/{dataset_name}_ZEN_test.txt")
    _, word_pos_oov = pos_evaluate(TwASP_ZEN_prediction, Gold_label, test_sent, train_words_dict, 'oov')
    _, word_pos_iv = pos_evaluate(TwASP_ZEN_prediction, Gold_label, test_sent, train_words_dict, 'iv')
    print(f'TwASP_ZEN_prediction: word_pos_oov = {word_pos_oov}, word_pos_iv = {word_pos_iv}')
    
    _, our_BERT_prediction, _ = get_words_dict(f"./our_prediction/{dataset_name}_BERT_{best_dimensions[dataset_name]}_test.txt")
    _, word_pos_oov = pos_evaluate(our_BERT_prediction, Gold_label, test_sent, train_words_dict, 'oov')
    _, word_pos_iv = pos_evaluate(our_BERT_prediction, Gold_label, test_sent, train_words_dict, 'iv')
    print(f'our_BERT_prediction: word_pos_oov = {word_pos_oov}, word_pos_iv = {word_pos_iv}')
    
    print(19*'*')

