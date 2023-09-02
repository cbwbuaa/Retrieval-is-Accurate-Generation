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
dataset=wikipedia
model=copyisallyouneed
cuda=$1
# ========== metadata ========== #

temp=2.0
beta=0.5
loss_type=focal_loss
warmup_step=10000
lr=1e-4
phrase_dim=128
model_size=small
batch_size=4
version=0901_${model_size}_bs${batch_size}_temp${temp}_focal_lr${lr}

# pretrain_model_path=/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikitext103/copyisallyouneed/train_pipeline/best_0601_shuffle_queue5k_mergedQ_eval1k_dim128_focal_loss_lr1e-4_prebatch0_beta0.5_warmup50000_prenum0_temp2.0_400000.pt
training_data_dir=/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/match_result_tok

root_dir=/apdcephfs/share_916081/ponybwcao/tmp/copyisallyouneed_v2
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
CUDA_VISIBLE_DEVICES=$cuda python3.8 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28445 train_prefix_only.py \
    --total_step 150000 \
    --save_every 10000 \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --total_workers ${#gpu_ids[@]} \
    --version $version \
    --training_data_dir $training_data_dir \
    --mode train \
    --temp ${temp} \
    --loss_type $loss_type \
    --beta $beta \
    --warmup_step $warmup_step \
    --lr $lr \
    --phrase_dim $phrase_dim \
    --model_size $model_size \
    --batch_size $batch_size \
    --resume true
    #--random_initialize true #> $root_dir/log/train_pipeline/${version}.error 2>&1 &
    # --pretrain_model_path $pretrain_model_path \
