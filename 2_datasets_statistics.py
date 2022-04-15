#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter
import sys
import numpy as np


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

def oov_statistics(train, test):
    test_oov_words = dict()
    for word in test:
        if word not in train:
            test_oov_words[word] = test[word]

    return np.round(100*sum(test_oov_words.values())/sum(test.values()), 1), test_oov_words

def info(d):
    nb_chars = 0
    chars = set()
    for key in d.keys():
        nb_chars += len(key) * d[key]
        chars.update(set(key))
    nb_char_types = len(chars)

    return nb_chars, sum(d.values())


# In[ ]:


for dataset_name in ['CTB5', 'CTB6', 'CTB7', 'CTB9', 'UD1']:
    print(19*'*')
    print(f'Statistics of {dataset_name}')
    print()
    _, train_label, train_words_dict = get_words_dict(f"../data/{dataset_name}/train.tsv")

    _, dev_label, dev_words_dict = get_words_dict(f"../data/{dataset_name}/dev.tsv")
    dev_oov_rate, dev_oov_word = oov_statistics(train_words_dict, dev_words_dict)

    _, test_label, test_words_dict = get_words_dict(f"../data/{dataset_name}/test.tsv")
    test_oov_rate, test_oov_word = oov_statistics(train_words_dict, test_words_dict)

    print(f'Number of sentences in train set: {len(train_label)}')
    print(f'Number of sentences in dev set: {len(dev_label)}')
    print(f'Number of sentences in test set: {len(dev_label)}')
    print()

    nb_char, nb_word = info(train_words_dict)
    print(f'Number of characters in train set: {nb_char}')
    print(f'Number of words in train set: {nb_word}')
    print()

    nb_char, nb_word = info(dev_words_dict)
    print(f'Number of characters in dev set: {nb_char}')
    print(f'Number of words in dev set: {nb_word}')
    print()

    nb_char, nb_word = info(test_words_dict)
    print(f'Number of characters in test set: {nb_char}')
    print(f'Number of words in test set: {nb_word}')
    print()

    print(f'OOV rate of dev set is: {dev_oov_rate}')
    print(f'OOV rate of test set is: {test_oov_rate}')

    print(19*'*')

