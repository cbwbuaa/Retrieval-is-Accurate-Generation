#!/bin/bash
# model_name=$2
# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling &
# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method greedy &


# CUDA_VISIBLE_DEVICES=0 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch10_beta0.7_warmup50000_prenum200_temp3.0_730000 &

# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch5_beta0.7_warmup50000_prenum500_temp2.0_840000 &

# CUDA_VISIBLE_DEVICES=2 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch5_beta0.7_warmup50000_prenum200_temp1.5_950000 &

# CUDA_VISIBLE_DEVICES=3 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch5_beta0.5_warmup50000_prenum200_temp1.5_950000 &

CUDA_VISIBLE_DEVICES=4 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch10_beta0.7_warmup50000_prenum200_temp3.0_400000 &

CUDA_VISIBLE_DEVICES=5 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch5_beta0.7_warmup50000_prenum500_temp2.0_400000 &

CUDA_VISIBLE_DEVICES=6 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch5_beta0.7_warmup50000_prenum200_temp1.5_400000 &

CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --training_mode train_pipeline --model_name best_0520_pipeline_update_lr1e-4_prebatch5_beta0.5_warmup50000_prenum200_temp1.5_400000 &


