# This script based on the public code: https://github.com/SVAIGBA/WMSeg/blob/master/wmseg_helper.py

import re
import numpy as np
import json
from os import path
from collections import defaultdict
from math import log
import copy

def get_word2id(train_data_path):
    word2id = {'[PAD]': 0}         # Our code.
    tag_list = set()               # Our code.
    tag_list.add('[NON]')          # Our code.
    word = ''                      # Our code.
    index = 1
    for line in open(train_data_path):
        if len(line) == 0 or line[0] == "\n":
            continue
        splits = line.split('\t')
        character = splits[0]
        label = splits[-1][:-1]    # Our code.
        tag_list.add(label[2:])    # Our code.
        word += character
        if label[0] in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    return word2id, list(tag_list) # Our code.