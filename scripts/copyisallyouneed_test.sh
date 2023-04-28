#!/bin/bash
model_name=$2
# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling &
# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method greedy &
CUDA_VISIBLE_DEVICES=$1 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling --model_name ${model_name}
CUDA_VISIBLE_DEVICES=$1 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method greedy --model_name ${model_name}
echo $model_name
# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_v2.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling
