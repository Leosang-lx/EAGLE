# python libs
import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
import logging

# eagle origin
from eagle.model.configs import EConfig
from eagle.model.cnets import Model
from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from eagle.model.utils import (
    prepare_logits_processor,
    reset_tree_mode,
    evaluate_posterior,
)
from eagle.model.kv_cache import initialize_past_key_values
# added
from comm.tensor_socket import CommCS
from utils import *
from ASDConfig import config as rconfig


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


class AsyncSDWrapper(nn.Module):
    """
    For both drafter client and verifier server
    """
    def __init__(
            self,
            use_eagle3,
            base_model_path,
            ea_model_path=None,
            # total_token=60,
            # depth=6,
            # top_k=10,
            # threshold=1.0,
            # dtype=torch.float16,
            # device="cuda",
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
            print(f'Loading draft model from {ea_model_path}...')
            dtype = kwargs['torch_dtype']
            device = kwargs['device_map']
            self.ea_layer = load_draft_model(
                ea_model_path,
                dtype=dtype,
                base_model_path=base_model_path,
                device=device,
            )
            self.device = torch.device(device)
            print("Loading draft model: done.")
        
        else:  # verifier server
            Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
            assert Type == "LlamaForCausalLM", "Only LlamaForCausalLM is supported for now"

            print(f'Loading base model from {base_model_path}...')
            self.base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
            self.device = self.base_model.device
            self.config = self.base_model.config
            self.vocab_size = self.base_model.lm_head.weight.shape[0]
            self.hidden_size = self.base_model.lm_head.weight.shape[-1]
            print("Loading base model: done.")

            if use_eagle3:
                eagle3_fc_dict = torch.load(os.path.join(base_model_path, 'eagle3_fc.bin'))
                self.eagle3_fc = nn.Linear(eagle3_fc_dict["in_dim"], eagle3_fc_dict["out_dim"], bias=False)
                self.eagle3_fc.weight.data = eagle3_fc_dict["weight"]
                self.eagle3_fc.to(self.device)
            else:
                raise NotImplementedError("Only support eagle3 for now")
            
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            output_orig=True,
        ):
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]
            
        # eagle3: mixing multi-layer hidden state and compress for lower transmission
        if self.use_eagle3:
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
            # if mix_hs:
            hidden_states = self.eagle3_fc(hidden_states)
            # return orig, mixed_hidden_state
        return orig, hidden_states
    
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
            prof=None,
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
            input_len = input_ids.shape[1]
            self.ea_layer.reset_kv()
            new_token = 0
            turns_cnt = 0

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
                ) = initialize_past_key_values(self.base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data
            reset_tree_mode(self)

        ##### sync prefill #####
        if not self.is_server:  # drafter client
            token, mix_hidden_state = prefill_sync(self, input_ids)
        else:  # verifier server
            input_ids = prefill_sync(
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
                    mixed_hidden_state=mix_hidden_state,
                    # next_token=new_token,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    log=log,
                    prof=prof,
                )
            else:  # verifier server
                self.catainfer(
                    kv_cache,
                    logits_processor,
                    input_ids,
                    prof=prof,
                )

            if not self.is_server:  # drafter client
                input_ids, mix_hidden_state, token, accept_length, turns = outputs
                new_token += accept_length
                if log:
                    turns_cnt += turns

                if input_ids is not None and self.tokenizer is not None:
                    if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif new_token > max_new_tokens:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif input_ids.shape[1] > max_length:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    
                    comm.send_to(should_stop, src=0)
                    if should_stop.item() == 1:
                        break
            else:
                should_stop = comm.recv_from()
                if should_stop.item() == 1:
                    break
        
        if self.is_draft_stage:
            if not log:
                return input_ids
            else:
                return input_ids, new_token, round_idx, turns_cnt


    def catainfer(
            self,
            kv_cache=None,
            logits_processor=None,
            input_ids=None,
            token=None,
            mixed_hidden_state=None,
            # new_token=None,
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
            accept_hidden_states = []
            accept_length_this_round = 0
            input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            input_len = input_ids.size(-1)

            with prof.time_context(f'Drafter: topK_genrate', cpu=False) if prof is not None else nullcontext():
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                    mixed_hidden_state,
                    input_ids_ea,
                    self.ea_layer.lm_head,
                    logits_processor,
                    total_tokens=rconfig.total_token,
                    depth=rconfig.depth,
                    top_k=rconfig.top_k,
                    return_last=True,
                    # prof=prof,
                )
            # if rconfig.none_expand:
                last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)

            tree_position_ids = tree_position_ids + input_len
            comm.send_to(draft_tokens)
            comm.send_to(tree_mask)
            comm.send_to(tree_position_ids)  # tree_position_ids可能可以直接根据tree_mask得到
            comm.send_to(retrieve_indices)
        else:  # verifier server
            past_key_values, past_key_values_data, current_length_data = kv_cache

        # pruned = False  # no verification received for the first turn
        turn_idx = -1
        while True:
            turn_idx += 1
            ####################
            # drafter client
            ####################
            if not self.is_server:
                existing_draft_length = draft_tokens.size(-1)
                # tree expansion
                with prof_or_null(f'Drafter: tree_expansion', prof, cpu=False):

                    if turn_idx == 0:  # first turn: expand the tree without pruning
                        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.expand_last(
                            last_ea_tree,
                            last_ea_state,
                            self.ea_layer.lm_head,
                            logits_processor,
                            device,
                            expand_depth=rconfig.depth,
                            expand_size=rconfig.expand_size,
                            # return_last=rconfig.none_expand,  # no need for return_last
                            # prof=prof
                        )
                        # last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                        tree_position_ids = tree_position_ids + input_ids.size(-1)

                    else:  # following turns: expand the tree based on the latest context
                        # recv the latest context
                        pruning_info, mixed_hidden_state = comm.recv_multi(device)
                        # pruning based on the latest context first
                        accept_tokens, truncate, pruned_tree = drafter_prune_draft(  # todo: fix this
                            pruning_info,
                            last_ea_tree
                        )
                        input_ids = torch.cat((input_ids, accept_tokens.to(input_ids.device)), dim=-1)
                        new_sample_token = pruning_info[0].item()
                        token = torch.tensor([[new_sample_token]], device=input_ids.device)

                        if truncate:  # drafter breaks the loop in the next iteration when verifier truncates
                            accept_hidden_states.append(mixed_hidden_state)
                            break

                        input_ids_ea = torch.cat((input_ids, token), dim=-1)
                        # prune the newly generated draft based on the latest context
                        # draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer

                        # draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2 = self.ea_layer.topK_genrate(
                        #     mixed_hidden_state,
                        #     input_ids_ea
                        # )  # todo: fix this

                        # generate a new draft tree based on the latest context
                        draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2, _ = self.ea_layer.topK_genrate(
                            mixed_hidden_state,
                            input_ids_ea,
                            self.ea_layer.lm_head,
                            logits_processor,
                            total_tokens=rconfig.total_token,
                            depth=rconfig.depth,
                            top_k=rconfig.top_k,
                            # return_last=True,
                            prof=prof,
                        )
                        # merge the two draft trees (both roots at the latest context)
                        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, appended_length = merge_two_tree(  # todo: fix this
                            (draft_tokens, retrieve_indices, tree_mask, tree_position_ids),
                            (draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2),
                        )

                # organize newly generated part of draft tokens and send to verifier
                appended_draft_tokens = draft_tokens[..., existing_draft_length:]
                appended_tree_mask = tree_mask[..., existing_draft_length:, :]
                appended_tree_position_ids = tree_position_ids[existing_draft_length:]

                # todo: optimized send_muti() for transmission
                # comm.send_multi([appended_draft_tokens, appended_tree_mask, appended_tree_position_ids, retrieve_indices])
                comm.send_to(appended_draft_tokens)
                comm.send_to(appended_tree_mask)
                comm.send_to(appended_tree_position_ids)
                comm.send_to(retrieve_indices)
                # explain: verifier requires the retrieve_indices to verify accurate tokens

            #####################
            ## verifier server
            #####################
            else:  
                # with prof_or_null(f'Verifer: recv draft', prof, cpu=True):
                #     # todo: optimized recv_muti() for transmission
                #     # draft_tokens, tree_mask, tree_position_ids, retrieve_indices = comm.recv_muti()  # maybe retrieve_indices?
                #     draft_tokens = comm.recv_from(device=device)
                #     tree_mask = comm.recv_from(device=device)
                #     tree_position_ids = comm.recv_from(device=device)
                #     retrieve_indices = comm.recv_from()
                # # todo: check size
                # assert True

                if turn_idx == 0:
                    draft_tokens = comm.recv_from(device=device)
                    tree_mask = comm.recv_from(device=device)
                    tree_position_ids = comm.recv_from(device=device)
                    retrieve_indices = comm.recv_from()

                else:
                    appended_draft_tokens = comm.recv_from(device=device)
                    appended_tree_mask = comm.recv_from(device=device)
                    appended_tree_position_ids = comm.recv_from(device=device)
                    retrieve_indices = comm.recv_from()
                    # do not check alignment for the first turn
                    
                    ### todo: check alignment (the new token aligns with the tree)
                    # explain: 当前有前半棵树的验证结果，新采样出的token，以及接收到的后半棵树；以最低开销进行剪枝，去掉新树部分的不合法的节点

                    # merge the newly received tree (TODO: this can be down asynchronously)
                    draft_tokens, tree_mask, tree_position_ids = merge_appended_draft(
                        draft_tokens, tree_mask, tree_position_ids, appended_draft_tokens, appended_tree_mask, appended_tree_position_ids,
                    )

                    # left_indices: including the newly appended part
                    # left_indices, truncate = cal_pruning_info(draft_tokens, retrieve_indices, best_candidate, accept_length, token, subseq_ri_cum_depths)  # todo: fix this

                    truncate, pruned_tree = verifier_prune_draft(draft_tokens, tree_mask, tree_position_ids, retrieve_indices, best_candidate, accept_indices, sample_token)
                    
                    # comm.send_to(left_indices)
                    if truncate:
                        break
                    else:
                        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = pruned_tree

                # last_tree_wo_ri = (draft_tokens, tree_mask, tree_position_ids)


                self.base_model.model.tree_mask = tree_mask
                ### verifier forward
                with prof_or_null('Verifier: forward', prof):
                    orig, mixed_hidden_state = self(  # todo: mix hidden_state now or later
                        draft_tokens,
                        past_key_values=past_key_values,
                        position_ids=tree_position_ids,
                        # mix_hs=True,
                    )
                
                # evaluation
                with prof_or_null('Verifier: evaluation', prof):
                    draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                    candidates = draft_tokens[0, retrieve_indices]
                    tree_logits = orig[0, retrieve_indices]
                    # verification
                    best_candidate, accept_length, sample_p = evaluate_posterior(
                        tree_logits, candidates, logits_processor
                    )
                    accept_length += 1
                # next_token = gen_token(prob=sample_p, logits_processor=logits_processor)
                # Do the following:
                # - add accepted tokens to input_ids
                # - sample the new token
                # - get hidden states of the accepted tokens
                accept_indices, input_ids, hidden_states, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    # new_token,
                    past_key_values_data,
                    current_length_data,
                    # self,
                    mixed_hidden_state,
                    sample_p
                )

                # mixed_hidden_state: compress for lower transmission; pass self.eagle3_fc in forward in default
                # mixed_hidden_state = self.eagle3_fc(hidden_states)
                pruning_info = torch.cat((accept_indices, sample_token.to(accept_indices.device)), dim=0)
                # send the pruning info (accept_draft_indices + sample_token) and the mixed hidden state to drafter at once after completing verification
                # comm.send_multi((pruning_info, mixed_hidden_state))  # maybe send before update_inference_inputs?
                comm.send_to(pruning_info)
                comm.send_to(mixed_hidden_state)

        if not self.is_server:
            gt_hidden_state = torch.cat(accept_hidden_states, dim=-2)
            return input_ids, gt_hidden_state, token, accept_length_this_round, turn_idx + 1



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
