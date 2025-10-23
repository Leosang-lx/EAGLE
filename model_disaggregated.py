# python libs
import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
# eagle origin
from eagle.model.configs import EConfig
from eagle.model.cnets import Model
from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from eagle.model.utils import *
from eagle.model import initialize_past_key_values
# added
from catainfer import load_base_model, load_draft_model
from comm.tensor_socket import CommCS
from utils import *



class Drafter(nn.Module):
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
            init_comm=False,
            server_ip=None,
    ):
        super().__init__()
        self.ea_layer = load_draft_model(
            ea_model_path,
            dtype=dtype,
            base_model_path=base_model_path,
            device=device,
        )
        self.device = self.ea_layer.device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        self.use_eagle3 = use_eagle3

        if init_comm:
            assert server_ip is not None
            self.comm = CommCS(server_ip=server_ip)
            self.is_server = True if self.comm.is_server else False

    def generate_async(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
        ):
        ##### initialization #####
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # sync prefill: send, forward, recv
        token, mix_hidden_state = prefill_sync(self, input_ids)


        



# for single request only
class BaseVerifer(nn.Module):
    def __init__(
            self,
            use_eagle3,
            base_model_path,
            # dtype=torch.float16,
            # device="cuda",
            init_comm=False,
            server_ip=None,
            **kwargs,
    ):
        super().__init__()
        self.use_eagle3 = use_eagle3
        self.base_model_name_or_path = base_model_path

        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        assert Type == "LlamaForCausalLM", "Only LlamaForCausalLM is supported for now"

        self.base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        self.device = self.base_model.device

        self.config = self.base_model.config
        self.hidden_size = self.base_model.lm_head.weight.shape[-1]
        self.vocab_size = self.base_model.lm_head.weight.shape[0]

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        if use_eagle3:
            eagle3_fc_dict = torch.load(os.path.join(base_model_path, 'eagle3_fc.bin'))
            self.eagle3_fc = nn.Linear(eagle3_fc_dict["in_dim"], eagle3_fc_dict["out_dim"], bias=False)
            self.eagle3_fc.weight.data = eagle3_fc_dict["weight"]

        if init_comm:
            assert server_ip is not None
            self.comm = CommCS(server_ip=server_ip)
            self.is_server = True if self.comm.is_server else False

    def generate_async(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
        ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)

        self.base_model.model.tree_mask = None
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.stage_base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        should_stop = torch.tensor(0, dtype=torch.int32)
        device = self.base_model.device
        comm = self.comm
        reset_tree_mode(self)

        prefill_sync(
            self,
            past_key_values=past_key_values,
            logits_processor=logits_processor,
        )

        kv_cache = (past_key_values, past_key_values_data, current_length_data)









        



    

