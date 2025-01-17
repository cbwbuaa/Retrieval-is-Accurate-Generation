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

prebatch_step=0
temp=2.0
beta=0.5
loss_type=focal_loss
warmup_step=50000
lr=1e-4
prebatch_num=0
phrase_dim=128
version=test

# pretrain_model_path=/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikitext103/copyisallyouneed/train_pipeline/best_0601_shuffle_queue5k_mergedQ_eval1k_dim128_focal_loss_lr1e-4_prebatch0_beta0.5_warmup50000_prenum0_temp2.0_400000.pt
training_data_dir=path/to/data/8split_all_phrase_ref_check_valid_merged

root_dir=./ #/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed
# backup
recoder_file=$root_dir/rest/$dataset/$model/recoder_train_$version.txt

# echo "find root_dir: $root_dir"
# echo "find version: $version"
# echo "write running log into recoder file: $recoder_file"
# mv $root_dir/ckpt/$dataset/$model/*_$version.pt $root_dir/bak/$dataset/$model
# # delete the previous tensorboard file
# rm $root_dir/rest/$dataset/$model/$version/* 
# rm -rf $root_dir/rest/$dataset/$model/$version 

# adjust total_step and save_every according to the number of your training examples.
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3.8 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28445 train_pipeline.py \
    --pretrain_model_path $pretrain_model_path \
    --total_step 220000 \
    --save_every 110000 \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --total_workers ${#gpu_ids[@]} \
    --data_file_num 8 \
    --version $version \
    --training_data_dir $training_data_dir \
    --mode train_pipeline \
    --prebatch_step ${prebatch_step} \
    --temp ${temp} \
    --loss_type $loss_type \
    --beta $beta \
    --prebatch_num $prebatch_num \
    --warmup_step $warmup_step \
    --lr $lr \
    --phrase_dim $phrase_dim
    #--random_initialize true #> $root_dir/log/train_pipeline/${version}.error 2>&1 &
