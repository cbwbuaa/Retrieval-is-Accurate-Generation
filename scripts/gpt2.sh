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
model=gpt2
cuda=$1
# ========== metadata ========== #

temp=2.0
warmup_step=50000
lr=1e-4
version=gpt2_temp${temp}_warmup_${warmup_step}_lr${lr}
# version=gpt2_randomInit_temp${temp}_warmup_${warmup_step}_lr${lr}

root_dir=/apdcephfs/share_916081/ponybwcao/Copyisallyouneed
# backup
recoder_file=$root_dir/rest/$dataset/$model/recoder_train_$version.txt


gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3.8 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28445 train.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --total_workers ${#gpu_ids[@]} \
    --version $version \
    --mode train \
    --temp ${temp} \
    --warmup_step $warmup_step \
    --lr $lr
    # --random_initialize true

