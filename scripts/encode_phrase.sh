#!/bin/bash
export NCCL_IB_DISABLE=1
cuda=$1
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python3.8 -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28206 build_phrase_index.py --mode encode --model_name best_0601_shuffle_queue5k_mergedQ_eval1k_dim128_focal_loss_lr1e-4_prebatch0_beta0.5_warmup50000_prenum0_temp2.0_400000