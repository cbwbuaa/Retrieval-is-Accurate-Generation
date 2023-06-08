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

prebatch_step=5
temp=3.0
beta=0.7
loss_type=focal_loss
warmup_step=50000
lr=1e-4
prebatch_num=200
# version=baseline
# version=new_inbatch_prebatch${prebatch_phrase_num}
# version=0516_cosine_prebatch${prebatch_phrase_num}_AllCandidates_${loss_type}_beta${beta}_inbatch_temp${temp}
# version=0517_valid_AllCandidates_prebatch${prebatch_phrase_num}_${loss_type}_beta${beta}_inbatch_temp${temp}
# version=0520_pipeline_update_lr${lr}_prebatch${prebatch_step}_beta${beta}_warmup${warmup_step}_prenum${prebatch_num}_temp${temp}
version=test

# pretrain_model_path=/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikitext103/copyisallyouneed/pretrain/best_test_400000.pt
# training_data_dir=/apdcephfs/share_916081/shared_info/ponybwcao/data/8split_all_phrase_ref
training_data_dir=/apdcephfs/share_916081/shared_info/ponybwcao/data/test_20

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
CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28445 train_pipeline.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --total_workers ${#gpu_ids[@]} \
    --data_file_num 1 \
    --version $version \
    --training_data_dir $training_data_dir \
    --mode train_pipeline \
    --prebatch_step ${prebatch_step} \
    --temp ${temp} \
    --loss_type $loss_type \
    --beta $beta \
    --prebatch_num $prebatch_num \
    --warmup_step $warmup_step \
    --lr $lr
    # --normalize true

