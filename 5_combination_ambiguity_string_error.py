#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter
import numpy as np


# In[ ]:


def readfile(filename):
    f = open(filename)
    data = []
    sentence = []
    label = []

    for line in f:
        if len(line) == 0 or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split('\t')
        char = splits[0]
        l = splits[-1][0].replace('\n', '')
        sentence.append(char)
        label.append(l)

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


# In[ ]:


def print_cas():
    # This function based on the public paper: https://www.aclweb.org/anthology/J05-4005.pdf

    best_dimensions = {'CTB5': '500', 'CTB6': '400', 'CTB7': '300', 'CTB9': '400', 'UD1': '300'} # The dimension is chose from Table 3 of our paper
    for data_name in best_dimensions:
        train_gold = readfile(f'../data/{data_name}/train.tsv') # Read the train dataset
        bi_gram_dict = {} # The dictionary of two-character

        for sent_idx in range(len(train_gold)): # Iterate all index of training dataset
            if len(train_gold[sent_idx][0]) < 2: # Ignore sentence with length less than 2
                continue

            for i in range(len(train_gold[sent_idx][0]) - 1): # Iterate all consectutive two-character
                word = ''.join(train_gold[sent_idx][0][i:i+2]) # Get two-character string
                tag = ''.join(train_gold[sent_idx][1][i:i+2]) # Get tag of two-chracter
                if word not in bi_gram_dict: # Check whether this two-character is in bi_gram_dict
                    bi_gram_dict[word] = set()
                
                if tag in ['BE', 'SS']: # Check whther tag is in the template following the public paper: https://www.aclweb.org/anthology/J05-4005.pdf
                    bi_gram_dict[word].add(tag)

        cas_set = set() # combination_ambiguity_string set
        for key in bi_gram_dict:
            if len(bi_gram_dict[key]) == 2: # get only two-character occur in two cases 'BE', 'SS'
                cas_set.add(key)

        cas_counter = Counter() # counter for combination_ambiguity_string
        for sent_idx in range(len(train_gold)):
            if len(train_gold[sent_idx][0]) < 2:
                continue
            for i in range(len(train_gold[sent_idx][0]) - 1):
                word = ''.join(train_gold[sent_idx][0][i:i+2])
                tag = ''.join(train_gold[sent_idx][1][i:i+2])
                
                if tag in ['BE', 'SS'] and word in cas_set:
                    cas_counter.update({word: 1})

        most_common_cas = [word for word, word_count in cas_counter.most_common(70)] # get 70 high-frequency two-character CASs following the public paper: https://www.aclweb.org/anthology/J05-4005.pdf

        test_gold = readfile(f'../data/{data_name}/test.tsv') # read the gold label

        tian_bert = readfile(f'./TwASP_prediction/{data_name}_BERT_test.txt') # read best result of TwASP using BERT
        tian_zen = readfile(f'./TwASP_prediction/{data_name}_ZEN_test.txt') # read best result of TwASP using ZEN
        our_pred = readfile(f'./our_prediction/{data_name}_BERT_{best_dimensions[data_name]}_test.txt') # read our best predicted result


        tian_bert_cas_correct = 0
        tian_zen_cas_correct = 0
        our_cas_correct = 0

        total_case = 0
        for sent_idx in range(len(test_gold)):
            if len(test_gold[sent_idx][0]) < 2:
                continue

            for i in range(len(test_gold[sent_idx][0]) - 1):
                word = ''.join(test_gold[sent_idx][0][i:i+2])
                tag = ''.join(test_gold[sent_idx][1][i:i+2])
                
                tian_bert_tag = ''.join(tian_bert[sent_idx][1][i:i+2])
                tian_zen_tag = ''.join(tian_zen[sent_idx][1][i:i+2])
                pred_tag = ''.join(our_pred[sent_idx][1][i:i+2])
                
                if word in most_common_cas and tag in ['BE', 'SS']:
                    total_case += 1
                    if tag == tian_bert_tag: tian_bert_cas_correct += 1
                    if tag == tian_zen_tag: tian_zen_cas_correct += 1
                    if tag == pred_tag: our_cas_correct += 1

        print('*'*19)
        print(f'Combination Ambiguity String Accuracy on {data_name} dataset')
        print('TwASP_BERT_prediction:', np.round(100*(tian_bert_cas_correct)/total_case,2))
        print('TwASP_ZEN_prediction:', np.round(100*(tian_zen_cas_correct)/total_case,2))
        print('our_BERT_prediction: ', np.round(100*(our_cas_correct)/total_case, 2))
        print('*'*19)


# In[ ]:


print_cas()

