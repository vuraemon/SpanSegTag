# This script based on the public code: https://github.com/SVAIGBA/WMSeg/blob/master/wmseg_model.py
# The code of SharedDropout module based on the public code: https://github.com/yzhangcs/crfpar/blob/crf-constituency/parser/modules/dropout.py
# The code of MLP module based on the public code: https://github.com/yzhangcs/crfpar/blob/crf-constituency/parser/modules/mlp.py
# The code of Biaffine module based on the public code: https://github.com/yzhangcs/crfpar/blob/crf-constituency/parser/modules/biaffine.py


from __future__ import absolute_import, division, print_function

import os
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pytorch_pretrained_zen as zen
from torch.nn import CrossEntropyLoss

# The code of SharedDropout module based on the public code: https://github.com/yzhangcs/crfpar/blob/crf-constituency/parser/modules/dropout.py
class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask

# The code of Biaffine module based on the public code: https://github.com/yzhangcs/crfpar/blob/crf-constituency/parser/modules/biaffine.py
class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

# The code of MLP module based on the public code: https://github.com/yzhangcs/crfpar/blob/crf-constituency/parser/modules/mlp.py
class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0):
        super(MLP, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'max_ngram_length': 5,
    'use_bert': False,
    'use_zen': False,
    'do_lower_case': False,
    'n_mlp_span': 500,        # Our code.
    'n_mlp_label': 100,       # Our code.
    'mlp_span_dropout': 0.1,  # Our code.
    'max_span_length': 7,     # Our code.
}


class WMSeg(nn.Module):

    def __init__(self, word2id, labelmap, tagmap, hpara, args): # Our code: labelmap, tagmap.
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec.pop('args')

        self.word2id = word2id
        self.tagmap = tagmap                       # Our code.
        self.idx2tag = {}                          # Our code.
        for key in self.tagmap:                    # Our code.
            self.idx2tag[self.tagmap[key]] = key   # Our code.

        self.labelmap = labelmap
        self.num_labels = len(self.labelmap) + 1
        self.hpara = hpara
        self.max_seq_length = self.hpara['max_seq_length']
        self.max_ngram_size = self.hpara['max_ngram_size']
        self.max_ngram_length = self.hpara['max_ngram_length']

        self.bert_tokenizer = None
        self.bert = None
        self.zen_tokenizer = None
        self.zen = None
        self.zen_ngram_dict = None


        if self.hpara['use_bert']:
            if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=self.hpara['do_lower_case'])
                self.bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                self.hpara['bert_tokenizer'] = self.bert_tokenizer
                self.hpara['config'] = self.bert.config
            else:
                self.bert_tokenizer = self.hpara['bert_tokenizer']
                self.bert = BertModel(self.hpara['config'])
            hidden_size = self.bert.config.hidden_size
            hidden_dropout_prob = self.bert.config.hidden_dropout_prob
            self.dropout = nn.Dropout(hidden_dropout_prob)

        elif self.hpara['use_zen']:
            if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(zen.PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.zen_tokenizer = zen.BertTokenizer.from_pretrained(args.bert_model, do_lower_case=self.hpara['do_lower_case'])
                self.zen_ngram_dict = zen.ZenNgramDict(args.bert_model, tokenizer=self.zen_tokenizer)
                self.zen = zen.modeling.ZenModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                self.hpara['zen_tokenizer'] = self.zen_tokenizer
                self.hpara['zen_ngram_dict'] = self.zen_ngram_dict
                self.hpara['config'] = self.zen.config
            else:
                self.zen_tokenizer = self.hpara['zen_tokenizer']
                self.zen_ngram_dict = self.hpara['zen_ngram_dict']
                self.zen = zen.modeling.ZenModel(self.hpara['config'])
            hidden_size = self.zen.config.hidden_size
            hidden_dropout_prob = self.zen.config.hidden_dropout_prob
            self.dropout = nn.Dropout(hidden_dropout_prob)
        else:
            raise ValueError()

        self.mlp_span_l = MLP(n_in=hidden_size,                   # Our code.
                              n_out=hpara['n_mlp_span'],          # Our code.
                              dropout=hpara['mlp_span_dropout'])  # Our code.
        self.mlp_span_r = MLP(n_in=hidden_size,                   # Our code.
                              n_out=hpara['n_mlp_span'],          # Our code.
                              dropout=hpara['mlp_span_dropout'])  # Our code.
        self.mlp_label_l = MLP(n_in=hidden_size,                  # Our code.
                               n_out=hpara['n_mlp_label'],        # Our code.
                               dropout=hpara['mlp_span_dropout']) # Our code.
        self.mlp_label_r = MLP(n_in=hidden_size,                  # Our code.
                              n_out=hpara['n_mlp_label'],         # Our code.
                              dropout=hpara['mlp_span_dropout'])  # Our code.
        self.span_attn = Biaffine(n_in=hpara['n_mlp_span'],       # Our code.
                                  bias_x=True,                    # Our code.
                                  bias_y=False)                   # Our code.
        self.label_attn = Biaffine(n_in=hpara['n_mlp_label'],     # Our code.
                                   n_out=len(self.tagmap),        # Our code: Size of POS tags set union with a non-word tag.
                                   bias_x=True,                   # Our code.
                                   bias_y=True)                   # Our code.


        if args.do_train:
            self.spec['hpara'] = self.hpara

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, word_seq=None, label_value_matrix=None, word_mask=None,
                input_ngram_ids=None, ngram_position_matrix=None, span_ids=None, span_label_ids=None): # Our code: span_ids, span_label_ids

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        sequence_output = self.dropout(sequence_output)
        batch_size, seq_len, hidden_size = sequence_output.shape # Our code.

        # Our code: sequence_output_f is forward vector, and sequence_output_b is backward vector
        sequence_output_f, sequence_output_b = sequence_output.chunk(2, dim=-1) 
        sequence_output = torch.cat((sequence_output_f[:, :-1], sequence_output_b[:, 1:]), -1)
        
        span_l = self.mlp_span_l(sequence_output)   # representation for left boundary for CWS task
        span_r = self.mlp_span_r(sequence_output)   # representation for right boundary for CWS task

        label_l = self.mlp_label_l(sequence_output) # representation for right boundary for Chinese POS tagging task
        label_r = self.mlp_label_r(sequence_output) # representation for right boundary for Chinese POS tagging task

        s_span = self.span_attn(span_l, span_r)     # Chinese word segmentation output
        s_span = torch.sigmoid(s_span)              # apply sigmoid to word segmentation output to get probabilty from 0 to 1

        s_label = self.label_attn(label_l, label_r)                     # Chinese POS tagging output
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1) # Chinese POS tagging output

        lens = input_ids.ne(0).sum(1) - 1
        mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
        mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)
        
        # Notably, our training and prediction progress,
        # we discard spans with length greater than 7 as the maximum n-gram length following (Diao et al., 2020)
        # to reduce negative spans.
        if self.hpara['max_span_length'] != 0:
            #  we discard spans with length greater than 7 as the maximum n-gram length following (Diao et al., 2020)
            span_mask = torch.tril(mask, self.hpara['max_span_length'])

        tag_seq, spans = self.span_decode(s_span, s_label, span_mask, lens) # tag_seq: output for joint CWS and POS tagging

        # CWS loss using binary cross entropy loss
        total_loss = nn.BCELoss()(s_span[span_mask], span_ids[span_mask].type_as(s_span))

        label_mask = mask & spans           # mask for compute loss for POS tagging task

        target = span_label_ids[label_mask] # the truth label after apply label_mask to it
        if target.shape[0] != 0:
            # POS tagging loss using cross entropy loss
            total_loss += nn.CrossEntropyLoss()(s_label[label_mask], span_label_ids[label_mask])

        return total_loss, tag_seq

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['mlp_span_dropout'] = args.mlp_span_dropout
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['max_ngram_length'] = args.max_ngram_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['n_mlp_span'] = args.n_mlp_span
        hyper_parameters['n_mlp_label'] = args.n_mlp_label
        hyper_parameters['max_span_length'] = args.max_span_length
        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model, args):
        spec = spec.copy()
        res = cls(args=args, **spec)
        res.load_state_dict(model)
        return res

    def segment_to_bies(self, begin, end, tag):
        # begin: the index of left boundary of span
        # end: the index of right boundary of span
        # tag: the predicted POS tag

        delta = end - begin
        if delta == 0:
            return []
        if delta == 1:
            return [self.labelmap[f'S-{tag}']]
        return [self.labelmap[f'B-{tag}']] + (delta - 2) * [self.labelmap[f'I-{tag}']] + [self.labelmap[f'E-{tag}']]

    # Our code for implementation of SpanPostProcessor algorithm in our paper
    def span_decode(self, pred, s_label, mask, lens):
        lens = lens.cpu().detach().numpy() # The lenghth of sentences
        pred = pred.permute(1, 2, 0) # The predicted spans for word segmentation task                
        batch_size = pred.shape[2] # Batch size
        seq_len = pred.shape[1] + 1 # Max sequence length of this batch size
        spans = torch.zeros_like(mask)
        # The list of valid predicted spans $\hat{S}_{novlp}$, satisfying non-overlapping between every two spans.
        mask = mask.permute(1, 2, 0) # Mask of this batch size
        pred_seg = mask * torch.gt(pred, 0.5) # Postive predicted spans
        segments = torch.nonzero(torch.eq(pred_seg, True)).cpu().detach().numpy() # Postive predicted spans
        start_end = [[(0, 0)] for _ in range(batch_size)] #  The list of predicted spans without overlapping ambiguity, this list can missed word boundary
        bies = [[self.labelmap["[CLS]"]] for _ in range(batch_size)] # Final result for joint CWS and POS tagging
        s_label = s_label.argmax(-1) # The predicted POS tags

        for segment in segments:                                                               # Our code.
            idx = segment[-1]                                                                  # Our code.
            s, e = segment[0], segment[1]                                                      # Our code.

            if s > start_end[idx][-1][1]:                                                      # Our code.
                start_end[idx].append((start_end[idx][-1][1], s))                              # Our code.
            if start_end[idx][-1][0] <= s and s < start_end[idx][-1][1]:                       # Our code.
                if pred[s, e, idx] > pred[start_end[idx][-1][0], start_end[idx][-1][1], idx]:  # Our code.
                    start_end[idx][-1] = (s, e)
            else:                                                                              # Our code.
                start_end[idx].append((s, e))                                                  # Our code.


        for idx in range(batch_size):                                                          # Our code.
            if start_end[idx][-1][1] != lens[idx] - 1:                                         # Our code.
                start_end[idx].append((start_end[idx][-1][1], lens[idx] - 1))                  # Our code.
            
            for j, se in enumerate(start_end[idx]):                                            # Our code.
                if j > 0 and se[0] != start_end[idx][j - 1][1]:                                # Our code.
                    tmp_seg = [start_end[idx][j - 1][1]]                                       # Our code.
                    for k in range(start_end[idx][j - 1][1], se[0]):                           # Our code.
                        if pred_seg[k, k + 1, idx]:                                            # Our code.
                           tmp_seg.append(k + 1)                                               # Our code.
                    tmp_seg.append(se[0])                                                      # Our code.
                    for k in range(len(tmp_seg) - 1):                                          # Our code.
                        bies[idx].extend(self.segment_to_bies(tmp_seg[k], tmp_seg[k + 1], self.idx2tag[s_label[idx, tmp_seg[k], tmp_seg[k + 1]].item()])) # Our code.
                        spans[idx, tmp_seg[k], tmp_seg[k + 1]] = True                          # Our code.
                
                bies[idx].extend(self.segment_to_bies(se[0], se[1], self.idx2tag[s_label[idx, se[0], se[1]].item()])) # Our code.
                spans[idx, se[0], se[1]] = True                                                # Our code.

            bies[idx].append(self.labelmap["[SEP]"])                                           # Our code.
            bies[idx].extend([0]*(seq_len - len(bies[idx])))                                   # Our code.

        all_label_ids = torch.tensor(bies, dtype=torch.long)                                   # Our code.
  
        return all_label_ids, spans                                                            # Our code.

    def load_data(self, data_path):
        flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        lines = readfile(data_path)

        data = []
        for sentence, label in lines:
            data.append((sentence, label))

        examples = []
        for i, (sentence, label) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                                         label=label, word=None, matrix=None))

        return examples

    def convert_examples_to_features(self, examples):

        max_seq_length = min(int(max([len(e.text_a.split(' ')) for e in examples]) * 1.1 + 2), self.max_seq_length)

        features = []

        tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            span_ids = torch.full((max_seq_length - 1, max_seq_length - 1), 0, dtype=torch.bool, device=torch.device('cpu')) # Our code.
            span_label_ids = torch.full((max_seq_length - 1, max_seq_length - 1), 0, dtype=torch.long, device=torch.device('cpu')) # Our code.
            valid = []
            label_mask = []

            begin_idx = 0                                                       # Our code.
            for end_idx, l in enumerate(labellist):                             # Our code.
                if l[0] == "B":                                                 # Our code.
                    begin_idx = end_idx                                         # Our code.
                elif l[0] == "E":                                               # Our code.
                    if end_idx + 1 >= max_seq_length - 1: break                 # Our code.
                    span_ids[begin_idx, end_idx + 1] = 1                        # Our code.
                    span_label_ids[begin_idx, end_idx + 1] = self.tagmap[l[2:]] # Our code.
                elif l[0] == "S":                                               # Our code.
                    if end_idx + 1 >= max_seq_length - 1: break                 # Our code.
                    span_ids[end_idx, end_idx + 1] = 1                          # Our code.
                    span_label_ids[end_idx, end_idx + 1] = self.tagmap[l[2:]]   # Our code.

            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.labelmap[labels[i]])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            
            word_ids = None
            matching_matrix = None

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])

                # random.shuffle(ngram_matches)
                ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

                max_ngram_in_seq_proportion = math.ceil(
                    (len(tokens) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                  dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

           
            features.append( InputFeatures(input_ids=input_ids,
                                                         input_mask=input_mask,
                                                         segment_ids=segment_ids,
                                                         label_id=label_ids,
                                                         valid_ids=valid,
                                                         label_mask=label_mask,
                                                         word_ids=word_ids,
                                                         matching_matrix=matching_matrix,
                                                         ngram_ids=ngram_ids,
                                                         ngram_positions=ngram_positions_matrix,
                                                         ngram_lengths=ngram_lengths,
                                                         ngram_tuples=ngram_tuples,
                                                         ngram_seg_ids=ngram_seg_ids,
                                                         ngram_masks=ngram_mask_array,
                                                         span_id=span_ids, # Our code.
                                                         span_label_id=span_label_ids # Our code.
                                                        ))

        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        all_span_ids = torch.stack([f.span_id for f in feature], dim=0) # Our code.
        all_span_label_ids = torch.stack([f.span_label_id for f in feature], dim=0) # Our code.


        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        span_ids  = all_span_ids.to(device) # Our code.
        span_label_ids  = all_span_label_ids.to(device) # Our code.
        
        word_ids = None
        matching_matrix = None
        word_mask = None

        if self.hpara['use_zen']:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, segment_ids, valid_ids, word_ids, word_mask, span_ids, span_label_ids # Our code: span_ids, span_label_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None, matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.matrix = matrix


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, matching_matrix=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None, span_id=None, span_label_id=None): # Ourcode: span_id, span_label_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.matching_matrix = matching_matrix

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks
        self.span_id = span_id              # Our code: represent the word segmentation span
        self.span_label_id = span_label_id  # Our code: represent the POS tagging span


def readfile(filename, return_tag_list=False):
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
        l = splits[-1][:-1]

        sentence.append(char)
        label.append(l)

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    

    return data


def readsentence(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            sentence = [char for char in line]
            label = ['<UNK>' for _ in sentence]
            data.append((sentence, label))
    return data

