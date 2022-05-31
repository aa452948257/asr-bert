#!/bin/bash
'''
Magic bash script to train your model.
'''
#################### model structure ####################
es=768  # also d_model; normal: 100 / bert: 768
hs=128
msl=512  # max_seq_len
nl=12
nh=12; dk=64; dv=64  # bert
score_util='pp'  # none/np/pp/mul
sent_repr='bin_sa_cls'
cls_type='stc'

#################### data & vocab dirs ####################
dataset="minet"

#You can keep this fixed 
dataroot="minet_data/raw"

#Pass path to export checkpoint and logs
exp_path="exp/snr_bert/"

#################### pretrained embedding ####################
fix_bert_model=false
if ! ${fix_bert_model}; then unset fix_bert_model; fi
bert_model_name='bert-base-chinese'  #  bert_model_name = {bert-base-uncased, xlnet-base-cased}

#################### training & testing options ####################
device=0
l2=1e-8
dp=0.3
bert_dp=0.1  # dropout rate for BERT
bs=32
mn=5.0
me=300             #max_epoch
optim="bertadam"  # bertadam/adamw
lr=3e-5  # for non-bert params
bert_lr=3e-5
wp=0.1  #warmup
init_type='uf'  # uf/xuf
init_range=0.02
seed=999

coverage=1.0

pre_trained_model='bert'

tod_pre_trained_model='multi-seq-asr-bert/ToD-BERT-jnt'

python3 snr_bert.py \
    --dataset ${dataset} --dataroot ${dataroot} \
    --bert_model_name ${bert_model_name} ${fix_bert_model:+--fix_bert_model} \
    --deviceId ${device} --random_seed ${seed} --l2 ${l2} --dropout ${dp} --bert_dropout ${bert_dp} \
    --optim_choice ${optim} --lr ${lr} --bert_lr ${bert_lr} --warmup_proportion ${wp} \
    --init_type ${init_type} --init_range ${init_range} \
    --batchSize ${bs} --max_norm ${mn} --max_epoch ${me} \
    --experiment ${exp_path} \
    --pre_trained_model ${pre_trained_model} \
    --coverage ${coverage} \
    --add_segment_ids \
    --add_info 'no_error' \

    #--without_system_act #Uncomment this if you wish to remove system act 
    #--add_l2_loss \ #Uncomment this if you wish to use l2 loss 
    #--tod_pre_trained_model ${pre_trained_model} #Uncomment this if you wish to multi-seq-asr-bert  

#python3 snr_bert.py \
#    --dataset ${dataset} --dataroot ${dataroot} \
#    --bert_model_name ${bert_model_name} ${fix_bert_model:+--fix_bert_model} \
#    --deviceId ${device} --random_seed ${seed} --l2 ${l2} --dropout ${dp} --bert_dropout ${bert_dp} \
#    --optim_choice ${optim} --lr ${lr} --bert_lr ${bert_lr} --warmup_proportion ${wp} \
#    --init_type ${init_type} --init_range ${init_range} \
#    --batchSize ${bs} --max_norm ${mn} --max_epoch ${me} \
#    --experiment ${exp_path} \
#    --pre_trained_model ${pre_trained_model} \
#    --coverage ${coverage} \
#    --add_segment_ids \
#    --add_info 'no_error' \
#    --testing \
#    --test_file '5_test'
