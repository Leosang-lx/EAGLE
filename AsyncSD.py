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
from eagle.model.utils import (
    prepare_logits_processor,
    reset_tree_mode,
    evaluate_posterior,
)
from eagle.model import initialize_past_key_values
# added
from catainfer import load_base_model, load_draft_model
from comm.tensor_socket import CommCS
from utils import *
from ASDConfig import config as run_config


class AsyncSDWrapper(nn.Module):
    """
    For both drafter client and verifier server
    """
    def __init__(
            self,
            use_eagle3,
            base_model_path,
            ea_model_path=None,
            total_token=60,
            depth=6,
            top_k=10,
            threshold=1.0,
            dtype=torch.float16,
            device="cuda",
            init_comm=False,
            server_ip=None,
            is_server=None,
            **kwargs,
    ):
        super().__init__()
        self.use_eagle3 = use_eagle3
        self.base_model_name_or_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

        # first init comm handler
        if init_comm:
            assert server_ip is not None
            self.comm = CommCS(server_ip=server_ip)
            self.is_server = self.comm.is_server
        
            assert hasattr(self, "is_server") and self.is_server is not None, 'Cannot run comm without field "is_server"'
        else:
            assert is_server is not None
            self.is_server = is_server
        
        # init model based on self.is_server
        if not self.is_server:  # drafter client
            self.ea_layer = load_base_model(
                ea_model_path,
                dtype=dtype,
                base_model_path=base_model_path,
                device=device,
            )
            self.device = self.ea_layer.device
        
        else:  # verifier server
            Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
            assert Type == "LlamaForCausalLM", "Only LlamaForCausalLM is supported for now"

            self.base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
            self.device = self.base_model.device
            self.config = self.base_model.config
            self.vocab_size = self.base_model.lm_head.weight.shape[0]
            self.hidden_size = self.base_model.lm_head.weight.shape[-1]

            if use_eagle3:
                eagle3_fc_dict = torch.load(os.path.join(base_model_path, 'eagle3_fc.bin'))
                self.eagle3_fc = nn.Linear(eagle3_fc_dict["in_dim"], eagle3_fc_dict["out_dim"], bias=False)
                self.eagle3_fc.weight.data = eagle3_fc_dict["weight"]
            else:
                raise NotImplementedError("Only support eagle3 for now")
            
    def forward(self, input_ids, past_key_values, tree_mask=None, tree_position_ids=None, output_orig=True):
        outputs, orig, hidden_states = self.base_model(
            input_ids, past_key_values=past_key_values, output_orig=output_orig
        )
        # eagle3: mixing multi-layer hidden state and compress for lower transmission
        hidden_state = self.eagle3_fc(hidden_states)
        return orig, hidden_state
    
    def prefill_sync(
            self,
            input_ids=None,
            past_key_values=None,
            logits_processor=None,
            prof=None,
    ):
        comm = self.comm
        device = self.device

        if not comm.is_server:  # drafter client
            # send input_ids to server
            comm.send_to(input_ids)
            # recv mixed hidden_state and the new token
            mixed_hidden_state = comm.recv_from(device=device)
            token = comm.recv_from(device=device)
            
            return token, mixed_hidden_state
            
        else:  # server with base model
            input_ids = comm.recv_from(device=device)  # todo: fix recv_from() on server
            orig, mixed_hidden_state = self(
                input_ids, past_key_values, output_orig=True
            )
            comm.send_to(mixed_hidden_state)  # todo: fix send_to() on server
            
            token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
            comm.send_to(token)

    
    def co_generate(
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
        
        should_stop = torch.tensor(0, dtype=torch.int32)
        device = self.device
        comm = self.comm

        if not self.is_server:  # drafter client
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

        else:  # verifier server
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
            reset_tree_mode(self)

        ##### sync prefill #####
        if not self.is_server:  # drafter client
            token, mix_hidden_state = prefill_sync(self, input_ids)
        else:  # verifier server
            prefill_sync(
                self,
                past_key_values=past_key_values,
                logits_processor=logits_processor,
            )
            kv_cache = (past_key_values, past_key_values_data, current_length_data)

        ##### async decoding #####
        # outer loop:
        for round_idx in range(max_length):
            if not self.is_server:  # drafter client
                outputs = self.catainfer(
                    logits_processor=logits_processor,
                    input_ids=input_ids,
                    token=token,
                    hidden_state=hidden_state,
                    new_token=new_token,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    log=log,
                    prof=prof
                )
            else:
                self.catainfer(
                    kv_cache,
                    logits_processor,
                    prof=prof
                )

    def catainfer(
            self,
            kv_cache=None,
            logits_processor=None,
            input_ids=None,
            token=None,
            mixed_hidden_state=None,
            new_token=None,
            max_new_tokens=None,
            max_length=None,
            log=False,
            prof=None,
    ):
        comm = self.comm
        device = self.device

        padding = torch.zeros(1, 1, dtype=torch.int64, device=device)


        # drafter initialization
        if not self.is_server:  # drafter client
            input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            input_len = input_ids.size(-1)

            with prof.time_context(f'Drafter: topK_genrate', cpu=False) if prof is not None else nullcontext():
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                    mixed_hidden_state,
                    input_ids_ea,
                    self.ea_layer.lm_head,
                    logits_processor,
                    total_tokens=run_config.init_total_tokens,
                    depth=run_config.init_depth,
                    top_k=run_config.init_top_k,
                    return_last=run_config.none_expand,
                    # prof=prof,
                )
            if run_config.none_expand:
                last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)

            tree_position_ids = tree_position_ids + input_len
            comm.send_to(draft_tokens)
            comm.send_to(tree_mask)
            comm.send_to(tree_position_ids)  # tree_position_ids可能可以直接根据tree_mask得到
        else:  # verifier server
            past_key_values, past_key_values_data, current_length_data = kv_cache

        pruned = False  # no verification received for the first turn
        turn_idx = 0
        while True:
            turn_idx += 1
            
            ####################
            ## drafter client ##
            ####################
            if not self.is_server:
                # tree expansion
                with prof_or_null(f'Drafter: tree_expansion', prof, cpu=False):

                    if pruned:  # expand from the latest context
                        
                        new_ea_token = draft_tokens[:, :1].to(input_ids.device)
                        input_ids_ea = torch.cat((input_ids, new_ea_token), dim=-1)

                    else:  # expand the last tree
                        draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2, last_ea_state = self.ea_layer.expand_last(
                            last_ea_tree,
                            last_ea_state,
                            self.stage_base_model.lm_head,
                            logits_processor,
                            device,
                            expand_depth=run_config.none_expand_depth,
                            expand_size=run_config.none_expand_size,
                            return_last=run_config.none_expand,
                            # prof=prof
                        )
                        last_ea_tree = (draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2)
                        tree_position_ids2 = tree_position_ids2 + input_ids.size(-1)

                # organize newly generated part of draft tokens and send to verifier
                appended_draft_tokens = ...
                appended_tree_mask = ...
                appended_tree_position_ids = ...

                # todo: optimized send_muti() for transmission
                comm.send_multi([appended_draft_tokens, appended_tree_mask, appended_tree_position_ids])
                comm.

            #####################
            ## verifier server ##
            #####################
            else:  
                # todo: optimized recv_muti() for transmission
                draft_tokens, tree_mask, tree_position_ids = comm.recv_muti()
                # todo: check size
                assert True
                ### todo: check alignment (the new token aligns with the tree)

                self.base_model.model.tree_mask = appended_tree_mask
                ### verifier forward
                with prof_or_null('Verifier: forward', prof):
                    orig, mixed_hidden_state = self(  # todo: mix hidden_state now or later
                        appended_draft_tokens,
                        past_key_values=past_key_values,
                        position_ids=tree_position_ids,
                    )
                
                # evaluation
                with prof_or_null('Verifier: evaluation', prof):
                    draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                    candidates = draft_tokens[0, retrieve_indices]
                    # verification
                    best_candidate, accept_length, sample_p = evaluate_posterior(
                        orig, candidates, logits_processor
                    )
                
                # update cache
                input_ids, new_token, hidden_states, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    mixed_hidden_state,
                    sample_p
                )

        if not self.is_server:
            return input_ids, hidden_state, token, accpet_length_this_round, turn_idx

        
                
            




# class Drafter(AsyncSDWrapper):
#     def __init__(
#             self,
#             use_eagle3,
#             base_model_path,
#             ea_model_path,
#             dtype=torch.float16,
#             device="cuda",
#             init_comm=False,
#             server_ip=None,
#     ):
#         super().__init__()
#         self.ea_layer = load_draft_model(
#             ea_model_path,
#             dtype=dtype,
#             base_model_path=base_model_path,
#             device=device,
#         )
#         self.device = self.ea_layer.device
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
#         self.use_eagle3 = use_eagle3

#         if init_comm:
#             assert server_ip is not None
#             self.comm = CommCS(server_ip=server_ip)
#             self.is_server = True if self.comm.is_server else False

#     def generate_async(
#             self,
#             input_ids=None,
#             temperature=0.0,
#             top_p=0.0,
#             top_k=0.0,
#             max_new_tokens=512,
#             max_length=2048,
#             log=False,
#             is_llama3=False,
#         ):
#         ##### initialization #####
#         if is_llama3:
#             stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
#         if temperature > 1e-5:
#             logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
#         else:
#             logits_processor = None

#         padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
#         input_ids = input_ids.clone()
#         self.ea_layer.reset_kv()

#         # sync prefill: send, forward, recv
#         token, mix_hidden_state = prefill_sync(self, input_ids)

#         # outer loop:
#         for idx in range(max_length):
#             outputs = catainfer(
#                 self,
#                 kv_cache=self.ea_layer.kv_cache,
#                 logits_processor=logits_processor,
#                 input_ids=input_ids,
#                 token=token,
#                 mixed_hidden_state=mix_hidden_state,
#                 new_token=token,
#                 max_new_tokens=max_new_tokens,
#             )

# # for single request only
# class BaseVerifer(AsyncSDWrapper):
#     def __init__(
#             self,
#             use_eagle3,
#             base_model_path,
#             # dtype=torch.float16,
#             # device="cuda",
#             init_comm=False,
#             server_ip=None,
#             **kwargs,
#     ):
#         super().__init__()
#         self.use_eagle3 = use_eagle3
#         self.base_model_name_or_path = base_model_path

#         Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
#         assert Type == "LlamaForCausalLM", "Only LlamaForCausalLM is supported for now"

#         self.base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
#         self.device = self.base_model.device

#         self.config = self.base_model.config
#         self.hidden_size = self.base_model.lm_head.weight.shape[-1]
#         self.vocab_size = self.base_model.lm_head.weight.shape[0]

#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
#         if use_eagle3:
#             eagle3_fc_dict = torch.load(os.path.join(base_model_path, 'eagle3_fc.bin'))
#             self.eagle3_fc = nn.Linear(eagle3_fc_dict["in_dim"], eagle3_fc_dict["out_dim"], bias=False)
#             self.eagle3_fc.weight.data = eagle3_fc_dict["weight"]

#         if init_comm:
#             assert server_ip is not None
#             self.comm = CommCS(server_ip=server_ip)
#             self.is_server = True if self.comm.is_server else False

#     def generate_async(
#             self,
#             input_ids,
#             temperature=0.0,
#             top_p=0.0,
#             top_k=0.0,
#             max_new_tokens=512,
#             max_length=2048,
#             log=False,
#             is_llama3=False,
#         ):
#         if is_llama3:
#             stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
#         if temperature > 1e-5:
#             logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
#         else:
#             logits_processor = None

#         # padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)

#         self.base_model.model.tree_mask = None
#         if hasattr(self, "past_key_values"):
#             past_key_values = self.past_key_values
#             past_key_values_data = self.past_key_values_data
#             current_length_data = self.current_length_data
#             # Reset the past key and value states
#             current_length_data.zero_()
#         else:
#             (
#                 past_key_values,
#                 past_key_values_data,
#                 current_length_data,
#             ) = initialize_past_key_values(self.stage_base_model)
#             self.past_key_values = past_key_values
#             self.past_key_values_data = past_key_values_data
#             self.current_length_data = current_length_data

#         should_stop = torch.tensor(0, dtype=torch.int32)
#         device = self.base_model.device
#         comm = self.comm
#         reset_tree_mode(self)

#         prefill_sync(
#             self,
#             past_key_values=past_key_values,
#             logits_processor=logits_processor,
#         )

#         kv_cache = (past_key_values, past_key_values_data, current_length_data)
