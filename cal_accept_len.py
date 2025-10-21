import json
# from transformers import AutoTokenizer
import numpy as np
import argparse

# tokenizer=AutoTokenizer.from_pretrained("/home/lyh/weights/hf/llama2chat/13B/")
bench_name = 'mt_bench'
method = 'ea3'
model_tag = 'L31-8B'
# model_tag = 'V13B'
temperature = 0.0

# SD settings
total_token = 60
depth = 6
topk = 10

parser = argparse.ArgumentParser()
parser.add_argument('--bench-name', type=str, default=bench_name)
parser.add_argument('--method', type=str, default=method)
parser.add_argument('--model-tag', type=str, default=model_tag)
parser.add_argument('--temperature', type=float, default=temperature)
parser.add_argument('--total-token', type=int, default=total_token)
parser.add_argument('--depth', type=int, default=depth)
parser.add_argument('--topk', type=int, default=topk)

args = parser.parse_args()
model_tag = args.model_tag
bench_name = args.bench_name
temperature = args.temperature
total_token = args.total_token
depth = args.depth
topk = args.topk

prefix = 'eagle/evaluation/'
root_dir = ''
prefix = 'eagle/evaluation/'
question_range = None  # 默认记录是全的

# jsonl_file = f"{root_dir}{prefix}{bench_name}/{model_id}-temperature-{temperature}.jsonl"
jsonl_file = f'{bench_name}/{model_tag}-{method}-t{int(temperature)}-tree{total_token}-d{depth}-topk{topk}.jsonl'
jsonl_file = f'{root_dir}{prefix}{jsonl_file}'

def analyze_ans(jsonl_file_path):

    # jsonl_file_base = "llama-2-chat-70b-fp16-base-in-temperature-0.0.jsonl"
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    # test accept length per round
    # for one choice: total rounds + total new_tokens

    test_choices_cnt = 1  # default 1 test choice (random seed)
    turns_cnt = 0
    new_tokens_cnt = 0
    if question_range is not None:
        data = data[question_range[0]:question_range[1]]

    for datapoint in data:
        records = datapoint['choices'][0]
        idxs = records['idxs']
        new_tokens = records['new_tokens']
        idxs_sum = sum(idxs)
        new_tokens_sum = sum(new_tokens)
        # print(f'{new_tokens_sum} new tokens in {idxs_sum} turns')
        turns_cnt += idxs_sum
        new_tokens_cnt += new_tokens_sum

    print(f'=========Summary for {jsonl_file}')
    print(f'{new_tokens_cnt} new tokens in {turns_cnt} turns')
    print(f'Average accept length: {new_tokens_cnt/turns_cnt}')

depth_size = [(d, d*10) for d in range(3, 11)]
for depth, total_token in depth_size:
    jsonl_file = f'{bench_name}/{model_tag}-{method}-t{int(temperature)}-tree{total_token}-d{depth}-topk{topk}.jsonl'
    jsonl_file = f'{root_dir}{prefix}{jsonl_file}'

    analyze_ans(jsonl_file)
