# !/bin/bash

export CUDA_VISIBLE_DEVICES=0
TREE="60"
DEPTH="6"
TOPK="10"
TEMPERATURE="0.0"
BENCHMARK="mt_bench"

OUTPUTFILE="$BENCHMARK/L31-8B-ea3-t0-tree$TREE-d$DEPTH-topk$TOPK.jsonl"
BASEDIR="/home/liux/big_file/meta-llama/Meta-Llama-3.1-8B-Instruct"
EAGLEDIR="/home/liux/big_file/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
MODELID="Meta-Llama-3.1-8B-Instruct"

python gen_ea_answer_llama3chat.py \
    --ea-model-path $EAGLEDIR \
    --base-model-path $BASEDIR \
    --model-id $MODELID \
    --bench-name $BENCHMARK \
    --temperature $TEMPERATURE \
    --total-token $TREE \
    --depth $DEPTH \
    --top-k $TOPK \
    --answer-file $OUTPUTFILE \
    --use-eagle3 \
    # --question-begin 30 \
    # --question-end 50 \

# OUTPUTFILE="$BENCHMARK/V13B-ea3-t0-tree$TREE-d$DEPTH-topk$TOPK.jsonl"
# BaseDir = "/home/liux/big_file/lmsys/vicuna-13b-v1.3/"
# EagleDir = "/home/liux/big_file/yuhuili/EAGLE3-Vicuna1.3-13B"
# ModelID = "vicuna-13b-v1.3"

# python gen_baseline_answer_vicuna.py \
#     --base-model-path /home/liux/big_file/lmsys/vicuna-13b-v1.3/ \
#     --model-id vicuna-13b-v1.3 \
#     --bench-name mt_bench \
#     --question-begin 30 \
#     --question-end 50 \
#     --temperature 0.0