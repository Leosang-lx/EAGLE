# !/bin/bash

export CUDA_VISIBLE_DEVICES=0
TREE="90"
DEPTH="9"
TOPK="10"
TEMPERATURE="0.0"
BENCHMARK="mt_bench"
OUTPUTFILE="$BENCHMARK/V13B-ea3-t0-tree$TREE-d$DEPTH-topk$TOPK.jsonl"
# python gen_baseline_answer_vicuna.py \
#     --base-model-path /home/liux/big_file/lmsys/vicuna-13b-v1.3/ \
#     --model-id vicuna-13b-v1.3 \
#     --bench-name mt_bench \
#     --question-begin 30 \
#     --question-end 50 \
#     --temperature 0.0
python gen_ea_answer_vicuna.py \
    --ea-model-path /home/liux/big_file/yuhuili/EAGLE3-Vicuna1.3-13B \
    --base-model-path /home/liux/big_file/lmsys/vicuna-13b-v1.3/ \
    --model-id vicuna-13b-v1.3 \
    --bench-name $BENCHMARK \
    --temperature $TEMPERATURE \
    --total-token $TREE \
    --depth $DEPTH \
    --top-k $TOPK \
    --answer-file $OUTPUTFILE \
    --use-eagle3 \
    # --question-begin 30 \
    # --question-end 50 \