import os
import torch
from eagle.model.cnets import Model
from eagle.model.cnets1 import Model as Model1
from eagle.model.configs import EConfig
import torch.nn as nn
import json
from tqdm import tqdm
from profiler import prof

# test eagle configuration
total_token = 64
depth = 6
top_k = 10
threshold = 1.0
dtype = torch.float16
device = torch.device("cuda:1")


# TODOs:
save_eagle3_fc = False
profile_eagle3 = True

# only load the lm_head
# lm_head = nn.Linear(4096, 128256).cuda()

# prefix = "C:/model_file"
prefix = "/home/liux/big_file"

draft_model_id = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# draft_model_id = "yuhuili/EAGLE3-Vicuna1.3-13B"
# base_model_id = "lmsys/vicuna-13b-v1.3"
# draft_model_id = "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
# base_model_id = "meta-llama/Llama-3.3-70B-Instruct"

ea_model_path = f'{prefix}/{draft_model_id}'
base_model_path = f'{prefix}/{base_model_id}'

config = EConfig.from_pretrained(ea_model_path)
assert hasattr(config, 'draft_vocab_size')

configpath = os.path.join(ea_model_path, "config.json")
with open(configpath, "r") as f:
    con = json.loads(f.read())
try:
    bias = con["bias"]
except:
    bias = True

# initialize draft model
print(f"Initialize Eagle3 model from {ea_model_path}...")
ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_path,load_emb=True)

load_model_path = os.path.join(ea_model_path, 'pytorch_model.bin')
ea_layer_state_dict = torch.load(load_model_path)

    ##### save eagle3-fc weight to server

if save_eagle3_fc:

    eagle3_fc_weight = ea_layer_state_dict['fc.weight']
    fc_dict = {
        "weight": eagle3_fc_weight,
        "in_dim": eagle3_fc_weight.shape[1],
        "out_dim": eagle3_fc_weight.shape[0],
    }
    torch.save(fc_dict, os.path.join(base_model_path, 'eagle3_fc.bin'))
    print(f"Eagle3 fc weight saved to {base_model_path}")
    


# eagle3
if config.vocab_size==config.draft_vocab_size:
    del ea_layer.d2t, ea_layer.t2d
load_ = ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
ea_layer.to(dtype).to(device)
ea_layer.init_tree()

print("Eagle3 model loaded")

### profile eagle3
if profile_eagle3:
    hidden_size = config.hidden_size

    warmup_repeat = 100
    test_repeat = 10

    hidden_state = torch.randn(1, 1, hidden_size).to(dtype).to(device)
    input_ids = torch.randint(1, 32000, (1, 2), dtype=torch.int64).to(device)
    for _ in tqdm(range(warmup_repeat), desc="warm up"):
        ea_layer.reset()
        ea_layer.reset_kv()
        _ = ea_layer.topK_genrate(hidden_state, input_ids, None, None)

    for _ in tqdm(range(test_repeat)):
        ea_layer.reset()
        ea_layer.reset_kv()
        with prof.profile_context('topK_genrate pipeline', cpu=False):
            _ = ea_layer.topK_genrate(hidden_state, input_ids, None, None, prof=prof)

    prof.print_all_elapsed_times()
