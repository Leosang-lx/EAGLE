# !/bin/bash

export CUDA_VISIBLE_DEVICES=1

# parameter combinations for evaluation
declare -a DEPTH_VALUES=("5" "7" "8" "10")
declare -a TOKEN_VALUES=("50" "70" "80" "100")

# check equal length
if [ ${#DEPTH_VALUES[@]} -ne ${#TOKEN_VALUES[@]} ]; then
    echo "Error: depth_values and token_values arrays must have the same length"
    exit 1
fi

BENCHMARK="mt_bench"
TEMPERATURE="0.0"
TOPK="10"
BASEDIR="/home/liux/big_file/meta-llama/Meta-Llama-3.1-8B-Instruct"
EAGLEDIR="/home/liux/big_file/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
MODELID="Meta-Llama-3.1-8B-Instruct"

# loop through parameter combinations
for i in "${!DEPTH_VALUES[@]}"; do
    DEPTH=${DEPTH_VALUES[$i]}
    TOKEN=${TOKEN_VALUES[$i]}
    
    echo "Running with depth=$DEPTH, total_token=$TOKEN"
    
    OUTPUTFILE="$BENCHMARK/L31-8B-ea3-t0-tree${TOKEN}-d${DEPTH}-topk${TOPK}.jsonl"
    
    python gen_ea_answer_llama3chat.py \
        --ea-model-path $EAGLEDIR \
        --base-model-path $BASEDIR \
        --model-id $MODELID \
        --bench-name $BENCHMARK \
        --temperature $TEMPERATURE \
        --total-token $TOKEN \
        --depth $DEPTH \
        --top-k $TOPK \
        --answer-file $OUTPUTFILE \
        --use-eagle3
    
    # check success
    if [ $? -eq 0 ]; then
        echo "Successfully completed: depth=$DEPTH, total_token=$TOKEN"
    else
        echo "Failed: depth=$DEPTH, total_token=$TOKEN"
    fi
    
    echo "----------------------------------------"
done