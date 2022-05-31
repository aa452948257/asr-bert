# -*- coding: utf-8 -*-

# @Time    : 2021-12-17 21:28
# @Author  : wangdeyuan
import os.path

import utils.Constants as Constants
from minet_data.raw.evaluate import make_sentence_pbsk
import random
from collections import Counter
import torch

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


with open('/home/wdy/project/ASR-Transformer/minet_data/data.txt', 'r', encoding='utf-8')as raw_data_fn:
    raw_data = raw_data_fn.readlines()
raw_data_fn.close()

random.shuffle(raw_data)

# 统计词典
words = []
# intent-slot-value集合
labels = set()

processed_data = []
for line in raw_data:
    line = line.rstrip('\n').split('\t')
    intent = line[2]
    utterance = line[3].replace(' ', '')
    words.extend(list(utterance))
    tags = line[4].split(' ')
    slot_lst = []
    value_lst = []
    span_start = []
    span_end = []

    now_slot = ''
    for idx, tag in enumerate(tags):
        if 'B-' in tag and now_slot == '':
            now_slot = tag.replace('B-', '')
            slot_lst.append(now_slot)
            span_start.append(idx)

        elif 'B-' in tag and now_slot is not '':
            span_end.append(idx)
            now_slot = tag.replace('B-', '')
            slot_lst.append(now_slot)
            span_start.append(idx)

        elif tag == 'O' and now_slot is not '':
            now_slot = ''
            span_end.append(idx)

        elif tag == 'O' and now_slot == '':
            continue

    if tag is not 'O':
        span_end.append(len(utterance))

    label_lst = []
    for idx, (start_id, end_id) in enumerate(zip(span_start, span_end)):
        value = utterance[start_id:end_id]
        if slot_lst[idx] == 'IP' or slot_lst[idx] == 'Frequency' or slot_lst[idx] == 'Location':
            intent_slot_value = intent + '-' + slot_lst[idx]
        else:
            intent_slot_value = intent + '-' + slot_lst[idx] + '-' + value
        label_lst.append(intent + '-' + slot_lst[idx] + '-' + value)
        labels.add(intent_slot_value)

    processed_data.append('[cls] ' + utterance + ' <==> ' + ';'.join(label_lst) + '\n')



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
    if count >= 1:
        num += 1
        if word not in word2idx:
            word2idx[word] = len(word2idx)
print('num of words with freq >= %d: %d' %(1, num))
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
top2bottom_dict = {k: sorted(set(v)) for k, v in top2bottom_dict.items()}

print('label vocab size: %d' % (len(label2idx)))
print('top-label vocab size: %d' % (len(toplabel2idx)))


sysact2idx = {
    Constants.PAD_WORD: Constants.PAD,
    Constants.UNK_WORD: Constants.UNK,
    Constants.CLS_WORD: Constants.CLS
}
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

write_fn = '/home/wdy/project/ASR-Transformer/minet_data/raw'
torch.save(memory, os.path.join(write_fn, 'memory.pt'))

# train_data = processed_data[: int(0.6*len(processed_data))]
train_data = processed_data[:]
dev_data = processed_data[int(0.6*len(processed_data)) : int(0.8*len(processed_data))]
test_data = processed_data[int(0.8*len(processed_data)) : ]

with open(os.path.join(write_fn, 'train'), 'w', encoding='utf-8') as f1:
    f1.writelines(train_data)

with open(os.path.join(write_fn, 'valid'), 'w', encoding='utf-8') as f2:
    f2.writelines(dev_data)

for snr in [10]:
    snr_test_data = []
    for line in test_data:
        start_token = '[cls] '
        split_str = ' <==> '
        sentence, label = line.split(split_str)
        sentence = sentence.split(' ')[1]
        sentence = make_sentence_pbsk(sentence, snr)
        # print(sentence)
        snr_test_data.append(start_token + sentence + split_str + label)

    with open(os.path.join(write_fn, str(snr) + '_test'), 'w', encoding='utf-8') as f3:
        f3.writelines(snr_test_data)