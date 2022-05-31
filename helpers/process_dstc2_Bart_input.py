# -*- coding: utf-8 -*-

# @Time    : 2021-11-30 20:15
# @Author  : wangdeyuan
import os
import sys
import json
import argparse
from collections import Counter
import torch
import numpy as np

install_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(install_path)

import utils.Constants as Constants
from helpers.act_slot_split_map import SPLIT_MAP

interjections_words = [
    'ah', 'aha', 'ahh', 'eh', 'er', 'em', 'erm',
    'hmm', 'hum', 'mm', 'mmm', 'oh', 'oops',
    'uhm', 'uh', 'uhh', 'um', 'umm'
]


def get_data_fnlist(scp_fn):
    '''get file name list'''
    with open(scp_fn, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
    return lines


def get_label(sem):
    '''get label from semantic structure'''
    slots = sem['slots']
    act = sem['act']

    assert len(slots) in [0, 1]
    if len(slots) == 0:
        label = act
    else:
        slot = slots[0]
        assert len(slot) in [1, 2]
        if len(slot) == 1:
            label = '%s-%s' % (act, slot[0])
        else:
            if act == 'request':
                label = '%s-%s' % (act, slot[1])
            else:
                label = '%s-%s-%s' % (act, slot[0], slot[1])

    return label


def split_label(label):
    '''
    input: act/act-slot/act-slot-value
    output: (act, None)/(act-slot, None)/(act-slot, act-slot-value)
    '''
    sem_list = label.split('-')
    if len(sem_list) <= 2:
        return (label, None)
    else:
        act_slot = '-'.join(sem_list[:2])
        return (act_slot, label)


def rule_prun(arcs, thres, bin_norm=False, rm_null=False):
    ''' prune arcs in one word bin '''
    prune_filter = lambda arc: arc['word'] not in interjections_words \
                               and arc['score'] >= thres
    pruned_arcs = list(filter(prune_filter, arcs))

    if bin_norm and len(pruned_arcs) > 0:
        p_sum = sum([arc['score'] for arc in pruned_arcs])
        for i in range(len(pruned_arcs)):
            pruned_arcs[i]['score'] /= p_sum

    # first do normalization; then remove null
    if rm_null:
        null_filter = lambda arc: arc['word'] != '!null'
        pruned_arcs = list(filter(null_filter, pruned_arcs))

    return pruned_arcs


def process_sys_acts(sys_acts):
    '''
    type:
      1 - CLS (ROOT)
      2 - act
      3 - slot
      4 - value
    about sibling: same node with multiple words
      0 - no siblings
      n - same node with n
      e.g.
        system act: <cls> inform pricerange x
        split:      <cls> inform price range x
        index:      0     1      2     3     4
        parent:     -1    0      1     1     3  # `price` & `range` share the same parent; parent of `x` only set to `price`
        sibling:    0     0      0     2     0  # sibling of `range` is `price`; 0 means no sibling
    '''

    type_dict = {'<cls>': 1, 'ACT': 2, 'SLOT': 3, 'VALUE': 4}

    memory = {'act': [], 'slot': [], 'value': []}

    token_seq = ['<cls>']
    parent_idx_seq = [-1]
    sib_idx_seq = [0]
    type_seq = [type_dict['<cls>']]
    for term in sys_acts:
        act = term['act']
        if act not in SPLIT_MAP:
            token_seq.append(act)
            parent_idx_seq.append(0)  # parent is ROOT
            sib_idx_seq.append(0)  # no sibling
            type_seq.append(type_dict['ACT'])
            cur_act_idx = len(token_seq) - 1
            memory['act'].append(act)
        else:
            act_words = SPLIT_MAP[act]
            for j, aw in enumerate(act_words):
                token_seq.append(aw)
                parent_idx_seq.append(0)
                type_seq.append(type_dict['ACT'])
                memory['act'].append(aw)
                if j == 0:  # e.g. `request` in `reqmore`
                    sib_idx_seq.append(0)
                    cur_act_idx = len(token_seq) - 1
                else:  # e.g. `more` in `reqmore`, sibling is `request`
                    sib_idx_seq.append(len(token_seq) - 2)

        slots = term['slots']
        if len(slots) == 0:
            continue
        for slot, value in slots:
            # print(slot, value)
            if slot == 'slot':
                slot = value
                value = None

            if slot not in SPLIT_MAP:
                token_seq.append(slot)
                parent_idx_seq.append(cur_act_idx)  # idx of current act
                sib_idx_seq.append(0)  # no sibling
                type_seq.append(type_dict['SLOT'])
                cur_slot_idx = len(token_seq) - 1
                memory['slot'].append(slot)
            else:
                slot_words = SPLIT_MAP[slot]
                for j, sw in enumerate(slot_words):
                    token_seq.append(sw)
                    parent_idx_seq.append(cur_act_idx)
                    type_seq.append(type_dict['SLOT'])
                    memory['slot'].append(sw)
                    if j == 0:  # e.g. `price` in `pricerange`
                        sib_idx_seq.append(0)
                        cur_slot_idx = len(token_seq) - 1
                    else:  # e.g. `range` in `pricerange`, sibling is `range`
                        sib_idx_seq.append(len(token_seq) - 2)

            if value is not None:
                v_list = str(value).strip().split()
                for v in v_list:
                    token_seq.append(v)
                    parent_idx_seq.append(cur_slot_idx)  # idx of current slot
                    sib_idx_seq.append(0)  # no sibling
                    type_seq.append(type_dict['VALUE'])
                    memory['value'].append(v)

    return token_seq, parent_idx_seq, sib_idx_seq, type_seq, memory


def read_wcn_data_and_save(log_fn, label_fn, save_fp, bin_norm=False, rm_null=False,augment=True,add_transcription=True,upsample_count=3):
    log_data = json.loads(open(log_fn).read())
    label_data = json.loads(open(label_fn).read())
    assert log_data['session-id'] == label_data['session-id']

    # store all words and labels
    word_list = []  # allow duplication to count word freq
    label_set = set()

    log_turns = log_data['turns']
    label_turns = label_data['turns']

    seq_lens = []
    n_poss = []
    n_discarded_sent = 0

    # about system acts
    sys_acts_memory = {'act': [], 'slot': [], 'value': []}

    bart_encoder_input = []
    bart_decoder_input = []

    for turn, turn2 in zip(log_turns, label_turns):
        assert turn['turn-index'] == turn2['turn-index']
        # process system acts
        sys_acts = turn['output']['dialog-acts']
        _, _, _, _, memory = process_sys_acts(sys_acts)

        sys_acts_memory['act'].extend(memory['act'])
        sys_acts_memory['slot'].extend(memory['slot'])
        sys_acts_memory['value'].extend(memory['value'])

        # get wcn
        # each `cnet` represents an utterance

        # get batch 1best/nbest
        batch_nbest = turn['input']['batch']['asr-hyps']
        batch_nbest = [term['asr-hyp'] for term in batch_nbest if term['asr-hyp'].rstrip()!=""]
        batch_1best = batch_nbest[0]

        # get manual transcription
        manual = turn2['transcription']
        prev_sys_response = turn['output']['transcript']

        in_seq = '[system] ' + prev_sys_response \
                 + ' [user] ' + batch_1best.strip()

        # input seq based on transcription seq
        transcription_in_seq = '[system] ' + prev_sys_response \
                               + ' [user] ' + manual.strip()

        seq_lens.append(len(in_seq.split(' ')))

        for asr_hyp in [batch_1best]:
            word_list += asr_hyp.split() + prev_sys_response.split()

        # get semantic labels
        sems = turn2['semantics']['json']
        labels = [get_label(sem) for sem in sems]
        labels_seq = ';'.join(labels)
        label_set = label_set.union(labels)

        if in_seq.rstrip() == '' or labels_seq.rstrip() == '' or prev_sys_response.rstrip() == '':
            continue

        intent_slot_value_lst = ' ; '.join(labels)
        in_seq = in_seq + ' [belief] ' + intent_slot_value_lst + ' [end]'
        transcription_in_seq = transcription_in_seq + ' [belief] ' + intent_slot_value_lst + ' [end]'

        bart_input = in_seq + '<==>' + transcription_in_seq + '\n'
        save_fp.write(bart_input)

        # bart_encoder_input.append(in_seq)
        # bart_decoder_input.append(transcription_in_seq)
        # write sequence
        # if add_transcription:
        #     seq2write = '%s\t<=>\t%s\t<=>\t%s\n' % (in_seq,transcription_in_seq,labels_seq)
        # else:
        #     seq2write = '%s\t<=>\t%s\n' % (in_seq, labels_seq)
        # save_fp.write(seq2write)
        #
        # if augment:
        #     seq2write = '%s\t<=>\t%s\n' % (transcription_in_seq, labels_seq)
        #     for i in range(upsample_count):
        #         save_fp.write(seq2write)

    return word_list, label_set, sys_acts_memory


def build_vocab_and_save(words, labels, sys_acts, memory_fn, min_freq=1):
    # word vocab
    counter = Counter(words)
    ordered_freq_list = counter.most_common()
    print('num of words: %d' % (len(ordered_freq_list)))

    word2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.CLS_WORD: Constants.CLS
    }

    num = 0
    for (word, count) in ordered_freq_list:
        if count >= min_freq:
            num += 1
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    print('num of words with freq >= %d: %d' %(min_freq, num))
    print('word vocab size: %d' % (len(word2idx)))

    # label vocab
    label2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK
    }
    # top label means: act/act-slot
    # bottom label means: act/act-slot/act-slot-value
    toplabel2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK
    }
    # top2bottom: index map from top label to bottom label
    # for classifier construction
    top2bottom_dict = {
        Constants.PAD: [Constants.PAD],
        Constants.UNK: [Constants.UNK]
    }

    for label in list(labels):
        if label not in label2idx:
            bottom_idx = len(label2idx)
            label2idx[label] = bottom_idx

            top, bottom = split_label(label)
            if top in toplabel2idx:
                if bottom is not None:
                    top_idx = toplabel2idx[top]
                    top2bottom_dict[top_idx].append(bottom_idx)
            else:
                top_idx = len(toplabel2idx)
                toplabel2idx[top] = top_idx
                top2bottom_dict[top_idx] = [bottom_idx]

    # add act-slot-None to top labels (only act-slot-value triples)
    idx2label = {v:k for k,v in label2idx.items()}
    done_tops = []
    for label in list(labels):
        top, bottom = split_label(label)
        # skip act/act-slot, only retain act-slot-value
        if bottom is None:
            continue
        # only need once for each top-label
        if top in done_tops:
            continue

        top_idx = toplabel2idx[top]
        cur_bottom_ids = top2bottom_dict[top_idx]
        cur_bottom_labels = [idx2label[idx] for idx in cur_bottom_ids]

        none_bottom_label = '%s-NONE' % top
        assert none_bottom_label not in cur_bottom_labels

        # add to bottom label
        none_bottom_idx = len(label2idx)
        label2idx[none_bottom_label] = none_bottom_idx

        # add to top2bottom_dict
        top2bottom_dict[top_idx].append(none_bottom_idx)

        done_tops.append(top)

    # deduplicate and sort
    top2bottom_dict = {k: sorted(set(v))
                       for k, v in top2bottom_dict.items()}

    print('label vocab size: %d' % (len(label2idx)))
    print('top-label vocab size: %d' % (len(toplabel2idx)))

    sysact2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.CLS_WORD: Constants.CLS
    }
    for token in sys_acts:
        if token not in sysact2idx:
            sysact2idx[token] = len(sysact2idx)
    print('system act vocab size: %d' % (len(sysact2idx)))

    # act & slot & value
    acts, slots, value_words = [], [], []
    single_acts, double_acts, triple_acts = [], [], []
    for label in list(labels):
        lis = label.split('-', 2)
        acts.append(lis[0])
        if len(lis) == 1:
            single_acts.append(lis[0])
        elif len(lis) == 2:
            double_acts.append(lis[0])
            slots.append(lis[1])
        elif len(lis) == 3:
            triple_acts.append(lis[0])
            slots.append(lis[1])
            value_lis = lis[2].split(' ')
            value_words.extend(value_lis)

    acts = sorted(list(set(acts)))
    slots = sorted(list(set(slots)))
    value_words = sorted(list(set(value_words)))
    single_acts = list(set(single_acts))
    double_acts = list(set(double_acts))
    triple_acts = list(set(triple_acts))

    act2idx = {Constants.PAD_WORD: Constants.PAD}
    slot2idx = {Constants.PAD_WORD: Constants.PAD}
    value2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS
    }
    for a in acts:
        if a not in act2idx:
            act2idx[a] = len(act2idx)
    for s in slots:
        if s not in slot2idx:
            slot2idx[s] = len(slot2idx)
    for v in value_words:
        if v not in value2idx:
            value2idx[v] = len(value2idx)
    print('act vocab size: %d' % len(act2idx))
    print('slot vocab size: %d' % len(slot2idx))
    print('value vocab size: %d' % len(value2idx))


    # save memory
    memory = {}
    memory['word2idx'] = word2idx
    memory['idx2word'] = {v:k for k,v in word2idx.items()}
    memory['label2idx'] = label2idx
    memory['idx2label'] = {v:k for k,v in label2idx.items()}
    memory['toplabel2idx'] = toplabel2idx
    memory['idx2toplabel'] = {v:k for k,v in toplabel2idx.items()}
    memory['top2bottom_dict'] = top2bottom_dict
    memory['sysact2idx'] = sysact2idx
    memory['idx2sysact'] = {v:k for k,v in sysact2idx.items()}
    memory['single_acts'] = single_acts
    memory['double_acts'] = double_acts
    memory['triple_acts'] = triple_acts
    memory['act2idx'] = act2idx
    memory['idx2act'] = {v:k for k,v in act2idx.items()}
    memory['slot2idx'] = slot2idx
    memory['idx2slot'] = {v:k for k,v in slot2idx.items()}
    memory['value2idx'] = value2idx
    memory['idx2value'] = {v:k for k,v in value2idx.items()}

    torch.save(memory, memory_fn)
    print('memory saved in %s' % (memory_fn))


############################ directories & files ############################

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='raw data directory')
parser.add_argument('--prun_opt', choices=['no', 'rule'], default='no', help='pruning options')
parser.add_argument('--prun_score_thres', type=float, default=None)
parser.add_argument('--bin_norm', action='store_true', help='bin-level normalization')
parser.add_argument('--rm_null', action='store_true', help='remove null tokens')
parser.add_argument('--out_dir', required=True, help='output directory')
opt = parser.parse_args()
print(opt)

data_dir = opt.data_dir
train_scp_fn = os.path.join(data_dir, 'scripts/config/dstc2_train.flist')
valid_scp_fn = os.path.join(data_dir, 'scripts/config/dstc2_dev.flist')
test_scp_fn = os.path.join(data_dir, 'scripts/config/dstc2_test.flist')

processed_data_dir = os.path.join(opt.out_dir, 'processed_data')

subdir_dict = {
    'no': 'raw' + '_rmnull' * opt.rm_null,
    'rule': 'rule_prun_thres_%s' % (opt.prun_score_thres) + '_norm' * opt.bin_norm + '_rmnull' * opt.rm_null,
}
processed_wcn_data_dir = os.path.join(processed_data_dir, subdir_dict[opt.prun_opt])

if not os.path.exists(processed_data_dir): os.makedirs(processed_data_dir)
if not os.path.exists(processed_wcn_data_dir): os.makedirs(processed_wcn_data_dir)

train_data_fn = os.path.join(processed_wcn_data_dir, 'train')
valid_data_fn = os.path.join(processed_wcn_data_dir, 'valid')
test_data_fn = os.path.join(processed_wcn_data_dir, 'test')
memory_fn = os.path.join(processed_wcn_data_dir, 'memory.pt')
pre_log_fn = os.path.join(processed_wcn_data_dir, 'log')
label_word_fn = os.path.join(processed_wcn_data_dir, 'label_word.dic')

fnlist = {}
fnlist['train'] = get_data_fnlist(train_scp_fn)
fnlist['valid'] = get_data_fnlist(valid_scp_fn)
fnlist['test'] = get_data_fnlist(test_scp_fn)

fps = {}
fps['train'] = open(train_data_fn, 'w')
fps['valid'] = open(valid_data_fn, 'w')
fps['test'] = open(test_data_fn, 'w')

train_wcn_words = []
train_labels = set()
all_wcn_words = []
all_labels = set()

# system act vocabs
# train_sa = {'act': [], 'slot': [], 'value': []}
train_sa = []

seq_lens = []
n_poss = []

n_discarded_sent = {'train': 0, 'valid': 0, 'test': 0}


for mode in ['train', 'valid', 'test']:
    for fn in fnlist[mode]:
        fn = os.path.join(data_dir, 'ori_data', fn)
        label_fn = os.path.join(fn, 'label.json')
        log_fn = os.path.join(fn, 'log.json')
        augment = False
        if mode in ['train','valid']:
            augment = True
        word_list, label_set, sys_mem = read_wcn_data_and_save(
            log_fn, label_fn, fps[mode], bin_norm=opt.bin_norm, rm_null=opt.rm_null,augment=False,add_transcription=True
        )
        if mode == 'train':
            train_wcn_words += word_list
            train_labels = train_labels.union(label_set)
            train_sa += (sys_mem['act'] + sys_mem['slot'] + sys_mem['value'])
        all_wcn_words += word_list
        all_labels = all_labels.union(label_set)

    print('done writing %s file' % (mode))
    fps[mode].close()

label_set = set()
for _label in all_labels:
    label = _label.split('-')
    for i in label:
        if ' ' in i:
            lst = i.split(' ')
            for j in lst:
                label_set.add(j)
        else:
            label_set.add(i)

with open(label_word_fn, 'w', encoding='utf-8') as f:
    for i in label_set:
        f.write(i + '\n')

build_vocab_and_save(train_wcn_words, train_labels, train_sa, memory_fn, min_freq=1)

# statistic
with open(pre_log_fn, 'w') as fp:
    fp.write('train word vocab size: %d\n' % len(set(train_wcn_words)))  # 1741
    fp.write('all word vocab size: %d\n' % len(set(all_wcn_words)))  # 1847
    fp.write('out-of-train-vocab size: %d\n' % len(set(all_wcn_words) - set(train_wcn_words)))  # 106
    fp.write('#train labels: %d\n' % len(train_labels))  # 149
    fp.write('#all labels: %d\n' % len(all_labels))  # 176
    fp.write('out-of-train-label size: %d\n' % len(all_labels - train_labels))  # 27
