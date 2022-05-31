import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.hierarchical_classifier import HierarchicalClassifier
from transformers import BertModel
from transformers.modeling_bert import BertEmbeddings


def make_model(opt, memory):
    if opt.dataset == "dstc2":
        return TOD_ASR_Transformer_STC(opt, memory)
    else:
        return SNR_Transformer(opt, memory)


def is_filter(string):
    if string.isdigit():
        return True


class BertPhoneEmbeddings(BertEmbeddings):
    def __init__(self, config, memory, opt):
        super().__init__(config)
        self.id2word = memory['idx2word']
        self.word2phone = memory['phone_dict']
        self.word2phone['[cls]'] = ['[CLS]']
        self.word2phone['[sep]'] = ['[SEP]']
        self.word2phone['[pad]'] = ['[PAD]']
        self.phone2idx = memory['phone2idx']
        self.phone2idx['[UNK]'] = len(self.phone2idx)
        self.phone2idx['[SEP]'] = len(self.phone2idx)
        self.phone2idx['[PAD]'] = len(self.phone2idx)
        self.phone2idx['[CLS]'] = len(self.phone2idx)
        self.phone_embedding = nn.Embedding(len(self.phone2idx.keys()), config.hidden_size, padding_idx=len(self.phone2idx)-1)
        self.tokenizer = opt.tokenizer
        self.phone_weight = opt.phone_weight
        # 看一下哪些单词在词典中找不到
        self.not_in_phone_words = set()

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # add phone_embeddings
        batch_phone_embeddings = []
        for idx, one_batch_seq_ids in enumerate(input_ids):
            one_batch_phone_embed = []
            for word in [self.tokenizer.decode([i]) for i in one_batch_seq_ids]:
                word = word.lower()
                if word in self.word2phone:
                    phone_lst = self.word2phone[word]
                else:
                    if is_filter(word):
                        phone_lst = ['[UNK]']
                    else:
                        phone_lst = ['[UNK]']
                        self.not_in_phone_words.add(word)
                seq_phone = [self.phone2idx[phone] for phone in phone_lst]
                seq_phone = torch.FloatTensor(seq_phone).long().cuda()
                token_phone_embedding = self.phone_embedding(seq_phone)
                token_phone_embedding = token_phone_embedding.mean(dim=0)
                one_batch_phone_embed.append(token_phone_embedding)
            one_batch_phone_embed = torch.stack(one_batch_phone_embed, dim=0)
            batch_phone_embeddings.append(one_batch_phone_embed)
        batch_phone_embeddings = torch.stack(batch_phone_embeddings, dim=0)

        # final embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = (1 - self.phone_weight) * embeddings + self.phone_weight * batch_phone_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPhoneModel(BertModel):
    '''
        add phone embedding
    '''
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def set_phone_embeddings(self, memory, opt):
        self.embeddings = BertPhoneEmbeddings(self.config, memory, opt)


class TOD_ASR_Transformer_STC(nn.Module):
    '''TOD ASR Transformer Semantic Tuple Classifier'''
    def __init__(self, opt, memory):
        super(TOD_ASR_Transformer_STC, self).__init__()

        #self.pretrained_model_opts = pretrained_model_opts
        if opt.use_phone:
            self.bert_encoder = BertPhoneModel.from_pretrained('bert-base-uncased')
            self.bert_encoder.set_phone_embeddings(memory, opt)
            print('--------------using phone setting--------------')
        else:
            self.bert_encoder = opt.pretrained_model
            print('--------------not using phone setting--------------')

        self.dropout_layer = nn.Dropout(opt.dropout)

        self.device = opt.device
        self.score_util = opt.score_util
        self.sent_repr = opt.sent_repr
        self.cls_type = opt.cls_type

        # feature dimension
        fea_dim = 768

        self.clf = HierarchicalClassifier(opt.top2bottom_dict, fea_dim, opt.label_vocab_size, opt.dropout)
        self.lm_model = nn.Linear(fea_dim, len(opt.tokenizer))

    def forward(self,opt,input_ids,trans_input_ids=None,seg_ids=None,trans_seg_ids=None,return_attns=False,classifier_input_type="asr"):
        
        #linear input to fed to downstream classifier 
        lin_in=None 

        # encoder on asr out
        #If XLM-Roberta don't pass token type ids 
        if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta": 
            outputs = self.bert_encoder(input_ids=input_ids,attention_mask=input_ids>0)
        else:
            outputs = self.bert_encoder(input_ids=input_ids,attention_mask=input_ids>0,token_type_ids=seg_ids)    
        sequence_output = outputs[0]
        asr_lin_in = sequence_output[:, 0, :]

        #encoder on manual transcription
        trans_lin_in = None
        if trans_input_ids is not None:
            #If XLM-Roberta don't pass token type ids 
            if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta":
                trans_outputs = self.bert_encoder(input_ids=trans_input_ids,attention_mask=trans_input_ids>0)
            else:
                trans_outputs = self.bert_encoder(input_ids=trans_input_ids,attention_mask=trans_input_ids>0,token_type_ids=trans_seg_ids)  
            trans_sequence_output = trans_outputs[0]
            trans_lin_in = trans_sequence_output[:, 0, :]
        
        if classifier_input_type == "transcript":
            lin_in = trans_lin_in
        elif classifier_input_type == "asr":
            lin_in = asr_lin_in
        elif classifier_input_type == "warm_up":
            # [bz, max_len, vocab_size]
            lm_logits = self.lm_model(sequence_output)
            lm_log_softmax = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            return [], {}, [], lm_log_softmax, []

        # decoder / classifier
        if self.cls_type == 'stc':
            top_scores, bottom_scores_dict, final_scores = self.clf(lin_in)

        if return_attns:
            return top_scores, bottom_scores_dict, final_scores, asr_lin_in, trans_lin_in
        else:
            return top_scores, bottom_scores_dict, final_scores, asr_lin_in, trans_lin_in

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'),
                map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))


class SNR_Transformer(nn.Module):
    '''SNR Transformer Semantic Tuple Classifier'''
    def __init__(self, opt, memory):
        super(SNR_Transformer, self).__init__()

        self.bert_encoder = opt.pretrained_model
        self.dropout_layer = nn.Dropout(opt.dropout)
        self.device = opt.device
        self.score_util = opt.score_util
        self.sent_repr = opt.sent_repr
        self.cls_type = opt.cls_type

        # feature dimension
        fea_dim = 768

        self.clf = HierarchicalClassifier(opt.top2bottom_dict, fea_dim, opt.label_vocab_size, opt.dropout)


    def forward(self,opt,input_ids,trans_input_ids=None,seg_ids=None,trans_seg_ids=None,return_attns=False,classifier_input_type="asr"):

        #linear input to fed to downstream classifier
        lin_in=None

        # encoder on asr out
        #If XLM-Roberta don't pass token type ids
        if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta":
            outputs = self.bert_encoder(input_ids=input_ids,attention_mask=input_ids>0)
        else:
            outputs = self.bert_encoder(input_ids=input_ids,attention_mask=input_ids>0,token_type_ids=seg_ids)
        sequence_output = outputs[0]
        asr_lin_in = sequence_output[:, 0, :]

        #encoder on manual transcription
        trans_lin_in = None
        if trans_input_ids is not None:
            #If XLM-Roberta don't pass token type ids
            if opt.pre_trained_model and opt.pre_trained_model=="xlm-roberta":
                trans_outputs = self.bert_encoder(input_ids=trans_input_ids,attention_mask=trans_input_ids>0)
            else:
                trans_outputs = self.bert_encoder(input_ids=trans_input_ids,attention_mask=trans_input_ids>0,token_type_ids=trans_seg_ids)
            trans_sequence_output = trans_outputs[0]
            trans_lin_in = trans_sequence_output[:, 0, :]

        if classifier_input_type=="transcript":
            lin_in = trans_lin_in
        else:
            lin_in = asr_lin_in

            # decoder / classifier
        if self.cls_type == 'stc':
            top_scores, bottom_scores_dict, final_scores = self.clf(lin_in)


        if return_attns:
            return top_scores, bottom_scores_dict, final_scores, asr_lin_in,trans_lin_in
        else:
            return top_scores, bottom_scores_dict, final_scores, asr_lin_in,trans_lin_in

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'),
                                            map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))