import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from eagle.model.configs import EConfig
from eagle.model.cnets import Model
from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from catainfer import load_base_model, load_draft_model



class DraftClient(nn.Module):
    def __init__(
            self,
            use_eagle3,
            base_model_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            dtype=torch.float16,
            device="cuda",
    ):
        super().__init__()
        config = EConfig.from_pretrained(ea_model_path)
        assert hasattr(config, "draft_vocab_size")

        configpath = os.path.join(ea_model_path, "config.json")
        with open(configpath, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con['bias']
        except:
            bias = True
        
        self.ea_layer = load_draft_model(
            ea_model_path,
            dtype=dtype,
            base_model_path=base_model_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        self.use_eagle3 = use_eagle3


class BaseServer(nn.Module):
    def __init__(
            self,
            use_eagle3,
            base_model_path,
            # dtype=torch.float16,
            # device="cuda",
            **kwargs,
    ):
        super().__init__()
        self.use_eagle3 = use_eagle3
        self.base_model_name_or_path = base_model_path

        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        assert Type == "LlamaForCausalLM", "Only LlamaForCausalLM is supported for now"

        self.base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        self.config = self.base_model.config
        self.hidden_size = self.base_model.lm_head.weight.shape[-1]
        self.vocab_size = self.base_model.lm_head.weight.shape[0]

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        



    

