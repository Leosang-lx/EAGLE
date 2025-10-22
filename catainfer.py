import os
import json
import torch
from transformers import AutoConfig, AutoTokenizer

from eagle.model.configs import EConfig
from eagle.model.cnets import Model
from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM


def load_draft_model(ea_model_path, dtype=torch.float16, base_model_path=None, device="cuda"):
    """
    Ea_layer may need the embedding layer of the base model: just the lm_head layer of eagle-3
    """
    config = EConfig.from_pretrained(ea_model_path)
    assert hasattr(config, "draft_vocab_size")

    configpath = os.path.join(ea_model_path, "config.json")
    with open(configpath, "r") as f:
        con = json.loads(f.read())
    try:
        bias = con['bias']
    except:
        bias = True
    
    ea_layer = Model(config, bias=bias, path=base_model_path, load_emb=True)
    load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
    
    if config.vocab_size == config.draft_vocab_size:
        del ea_layer.d2t, ea_layer.t2d
    ea_layer.load_state_dict(torch.load(load_model_path), strict=False)
    ea_layer.to(dtype).to(device)
    ea_layer.init_tree()

    return ea_layer

    
