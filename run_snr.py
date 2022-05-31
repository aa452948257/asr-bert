# -*- coding: utf-8 -*-

# @Time    : 2021-12-31 14:34
# @Author  : wangdeyuan
import os
import sys
import json
import re
import random
import argparse

import numpy as np
import torch
from transformers import *
install_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(install_path)

from utils.gpu_selection import auto_select_gpu
from utils.STC_util import convert_labels, reverse_top2bottom, onehot_to_scalar
from models.model import make_model
from flask import Flask, render_template, request

MODEL_CLASSES = {
    "bert": (BertModel,BertTokenizer,'bert-base-chinese'),
    "roberta": (RobertaModel,RobertaTokenizer,'roberta-base'),
    "xlm-roberta": (XLMRobertaModel,XLMRobertaTokenizer,'xlm-roberta-base'),
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    ######################### model structure #########################
    parser.add_argument('--emb_size', type=int, default=256, help='word embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden layer dimension')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--n_layers', type=int, default=6, help='#transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='#attention heads')
    parser.add_argument('--d_k', type=int, default=64, help='dimension of k in attention')
    parser.add_argument('--d_v', type=int, default=64, help='dimension of v in attention')
    parser.add_argument('--score_util', default='pp', choices=['none', 'np', 'pp', 'mul'],
                        help='how to utilize scores in Transformer & BERT: np-naiveplus; pp-paramplus')
    parser.add_argument('--sent_repr', default='bin_sa_cls',
                        choices=['cls', 'maxpool', 'attn', 'bin_lstm', 'bin_sa', 'bin_sa_cls', 'tok_sa_cls'],
                        help='sentence level representation')
    parser.add_argument('--cls_type', default='stc', choices=['nc', 'tf_hd', 'stc'], help='classifier type')

    ######################### data & vocab #########################
    parser.add_argument('--dataset', required=True, help='<domain>')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--ontology_path', default=None, help='ontology')

    ######################## pretrained model (BERT) ########################
    parser.add_argument('--bert_model_name', default='bert-base-chinese',
                        choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
                                 'bert-base-chinese'])
    parser.add_argument('--fix_bert_model', action='store_true')

    ######################### training & testing options #########################
    parser.add_argument('--testing', action='store_true', help=' test your model (default is training && testing)')
    parser.add_argument('--deviceId', type=int, default=0, help='train model on ith gpu. -1:cpu, 0:auto_select')
    parser.add_argument('--random_seed', type=int, default=999, help='initial random seed')
    parser.add_argument('--l2', type=float, default=1e-8, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate at each non-recurrent layer')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='dropout rate for BERT')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--max_norm', type=float, default=5.0, help='threshold of gradient clipping (2-norm)')
    parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train')
    parser.add_argument('--experiment', default='exp', help='experiment directories for storing models and logs')
    parser.add_argument('--optim_choice', default='bertadam', choices=['adam', 'adamw', 'bertadam'], help='optimizer choice')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--bert_lr', default=3e-5, type=float, help='learning rate for bert')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup propotion')
    parser.add_argument('--init_type', default='uf', choices=['uf', 'xuf', 'normal'], help='init type')
    parser.add_argument('--init_range', type=float, default=0.02, help='init range, for naive uniform')

    ######################## system act #########################
    parser.add_argument('--coverage', type=float, default=1.0)

    ####################### Loss function setting ###############
    parser.add_argument('--add_l2_loss',action='store_true',help='whether to add l2 loss between pure and asr transcripts')

    ###################### Pre-trained model config ##########################
    parser.add_argument('--pre_trained_model', default='bert', help='pre-trained model name to use among bert,roberta,xlm-roberta')
    parser.add_argument('--tod_pre_trained_model',help = 'tod_pre_trained model checkpoint path')

    ##################### System act config ###################################
    parser.add_argument('--without_system_act',type=int, default=1, help='parameter to decide to add system act')

    ##################### Config to decide on segement ids ###################################
    parser.add_argument('--add_segment_ids',action='store_true', help = 'parameter to decide to add segment ids')


    opt = parser.parse_args()

    ######################### option verification & adjustment #########################
    # device definition
    if opt.deviceId >= 0:
        if opt.deviceId > 0:
            opt.deviceId, gpu_name, valid_gpus = auto_select_gpu(assigned_gpu_id=opt.deviceId - 1)
        elif opt.deviceId == 0:
            opt.deviceId, gpu_name, valid_gpus = auto_select_gpu()
        print('Valid GPU list: %s ; GPU %d (%s) is auto selected.' % (valid_gpus, opt.deviceId, gpu_name))
        torch.cuda.set_device(opt.deviceId)
        opt.device = torch.device('cuda')
    else:
        print('CPU is used.')
        opt.device = torch.device('cpu')

    # random seed set
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.random_seed)

    # d_model: just equals embedding size
    opt.d_model = opt.emb_size

    # ontology
    opt.ontology = None if opt.ontology_path is None else \
        json.load(open(opt.ontology_path))

    return opt


def pred_one_sample(i, ts, bottom_scores_dict, memory, opt):
    # ts: top scores
    # i: index of the sample in a batch
    pred_classes = []
    top_ids = [j for j, p in enumerate(ts) if p > 0.5]
    for ti in top_ids:
        bottom_ids = memory['top2bottom_dict'][ti]
        if len(bottom_ids) == 1:
            pred_classes.append(memory['idx2label'][bottom_ids[0]])
        else:
            bs = bottom_scores_dict['lin_%d' % ti][i]
            lbl_idx_in_vector = bs.data.cpu().numpy().argmax(axis=-1)
            real_lbl_idx = bottom_ids[lbl_idx_in_vector]
            lbl = memory['idx2label'][real_lbl_idx]
            if not lbl.endswith('NONE'):
                pred_classes.append(lbl)

    return pred_classes


def get_span(sentence, slot_name):
    device_id_lst = ['Terminal10', 'Terminal1', 'Terminal2', 'Terminal3', 'Terminal4', 'Terminal5', 'Terminal6',
                     'Router1', 'Router2', 'Router3']
    if slot_name == 'IP':
        result = re.findall(r"\d+\.\d+\.\d+\.\d+", sentence)
    elif slot_name == 'Frequency':
        for device_id in device_id_lst:
            sentence = sentence.replace(device_id, '')
        result = re.findall("\d+", sentence)
    elif slot_name == 'Location':
        result = re.findall(r"\[.*\]", sentence)
    if len(result) == 0:
        # print(sentence, slot_name)
        return 'None'
    else:
        return result[0]


def test_epoch(model, raw_in, opt, memory):
    model.eval()

    tok_seq_a = []
    tokenizer = opt.tokenizer
    for word in raw_in:
        tok_word = tokenizer.tokenize(word)
        tok_seq_a += tok_word

    tok_seq_a = [tokenizer.cls_token] + tok_seq_a
    seq_a_segments = [0] * len(tok_seq_a)
    bert_input_ids = [tokenizer.convert_tokens_to_ids(tok_seq_a)]
    bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.long, device=opt.device)
    seg_input_ids = torch.tensor(seq_a_segments, dtype=torch.long, device=opt.device)

    top_scores, bottom_scores_dict, batch_preds, asr_hidden_rep, trans_hidden_rep = model(opt, bert_input_ids,
                                                                                          bert_input_ids,
                                                                                          seg_ids=seg_input_ids,
                                                                                          trans_seg_ids=seg_input_ids,
                                                                                          classifier_input_type="asr")
    result = []
    for i, (ts, raw) in enumerate(zip(top_scores.tolist(), [raw_in])):
        pred_classes = pred_one_sample(i, ts, bottom_scores_dict, memory, opt)
        new_pred_classes = []
        sentence = raw_in
        for pre_label in pred_classes:
            if pre_label == '<unk>':
                new_pred_classes.append(pre_label)
                continue
            slot_name = pre_label.split('-')[1]
            if slot_name in ['IP', 'Frequency', 'Location']:
                processed_pred_label = get_span(sentence, slot_name)
                pre_label += ('-' + processed_pred_label)
                new_pred_classes.append(pre_label)
            else:
                new_pred_classes.append(pre_label)
        result.append(new_pred_classes)

    return result


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':

    opt = parse_arguments()

    pre_trained_model, pre_trained_tokenizer, model_name = MODEL_CLASSES.get(opt.pre_trained_model)
    opt.pretrained_model = pre_trained_model.from_pretrained(model_name)
    opt.tokenizer = pre_trained_tokenizer.from_pretrained(model_name)

    # memory
    memory = torch.load(os.path.join(opt.dataroot, 'memory.pt'))
    opt.word_vocab_size = opt.tokenizer.vocab_size  # subword-level
    opt.label_vocab_size = len(memory['label2idx'])
    opt.top_label_vocab_size = len(memory['toplabel2idx'])
    opt.top2bottom_dict = memory['top2bottom_dict']
    memory['bottom2top_mat'] = reverse_top2bottom(memory['top2bottom_dict'])
    print('word vocab size:', opt.word_vocab_size)
    print('#labels:', opt.label_vocab_size)
    print('#top-labels:', opt.top_label_vocab_size)
    print(opt)

    # exp dir
    opt.exp_dir = '/home/wdy/project/ASR-Transformer/exp/snr_bert/data_minet/nl_6__nh_4__dk_64__dv_64__bs_16__dp_0.3_0.1__opt_bertadam_0.1_3e-05_3e-05__mn_5.0__me_30__seed_999__score_pp__repr_bin_sa_cls__cls_stc__no_error'

    # model definition & num of params
    model = make_model(opt, memory)
    model = model.to(opt.device)

    model.load_model(os.path.join(opt.exp_dir, 'model.pt'))


    test_data = 'Terminal10更新频段为11'

    res = test_epoch(model, test_data, opt, memory)

    print(res)
