#!/bin/bash

# 配置代理
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

export PYTHONIOENCODING=utf-8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
my_port=23456

# pip3 install yaml
# pip3 install pynvml
# pip3 install tensorboard
# pip3 uninstall setuptools
# pip3 install setuptools==59.5.0
# pip3 install -r requirements.txt

# ========== metadata ========== #
dataset=wikitext103
model=copyisallyouneed
cuda=$1
# ========== metadata ========== #

prebatch_phrase_num=3
in_doc_neg_num=-1
temp=1.0
beta=0.95
loss_type=focal_loss
warmup_step=10000
lr=5e-4
# version=baseline
# version=new_inbatch_prebatch${prebatch_phrase_num}
# version=0516_cosine_prebatch${prebatch_phrase_num}_AllCandidates_${loss_type}_beta${beta}_inbatch_temp${temp}
# version=0517_valid_AllCandidates_prebatch${prebatch_phrase_num}_${loss_type}_beta${beta}_inbatch_temp${temp}
version=0519_update_lr${lr}_prebatch${prebatch_phrase_num}_beta${beta}_warmup${warmup_step}
# version=0516_inbatch_prebatch${prebatch_phrase_num}_indoc${in_doc_neg_num}_temp${temp}_${loss_type}_beta${beta}
# version=test

# pretrain_model_path=/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikitext103/copyisallyouneed/pretrain/best_test_400000.pt
# training_data_dir=/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref
training_data_dir=/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref_check_valid

root_dir=/apdcephfs/share_916081/ponybwcao/Copyisallyouneed
# backup
recoder_file=$root_dir/rest/$dataset/$model/recoder_train_$version.txt

# echo "find root_dir: $root_dir"
# echo "find version: $version"
# echo "write running log into recoder file: $recoder_file"
# mv $root_dir/ckpt/$dataset/$model/*_$version.pt $root_dir/bak/$dataset/$model
# # delete the previous tensorboard file
# rm $root_dir/rest/$dataset/$model/$version/* 
# rm -rf $root_dir/rest/$dataset/$model/$version 

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28445 train.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --total_workers ${#gpu_ids[@]} \
    --data_file_num 8 \
    --version $version \
    --training_data_dir $training_data_dir \
    --mode train \
    --prebatch_phrase_num ${prebatch_phrase_num} \
    --in_doc_neg_num ${in_doc_neg_num} \
    --temp ${temp} \
    --loss_type $loss_type \
    --beta $beta \
    --warmup_step $warmup_step \
    --lr $lr
    # --normalize true

