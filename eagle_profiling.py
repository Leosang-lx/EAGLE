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
device = torch.device("cuda:0")


# only load the lm_head
# lm_head = nn.Linear(4096, 128256).cuda()

ea_model_path = 'C:/model_file/yuhuili/EAGLE3-Vicuna1.3-13B/'
base_model_path = 'C:/model_file/lmsys/vicuna-13b-v1.3'

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
print("Initialize Eagle3 model...")
ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_path,load_emb=True)

load_model_path = os.path.join(ea_model_path, 'pytorch_model.bin')
ea_layer_state_dict = torch.load(load_model_path)

# eagle3
if config.vocab_size==config.draft_vocab_size:
    del ea_layer.d2t, ea_layer.t2d
load_ = ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
ea_layer.to(dtype).to(device)
ea_layer.init_tree()

print("Eagle3 model loaded")

hidden_size = config.hidden_size

warmup_repeat = 30
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
