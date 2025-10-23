import torch
from contextlib import nullcontext

# [ADD] logits processor
def gen_token(logits=None, prob=None, logits_processor=None):
    if logits_processor is not None:
        if logits is not None:
            logits = logits_processor(None, logits)
            prob = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(prob, 1)

    else:
        if logits is not None:
            prob = logits
        token = torch.argmax(prob, dim=-1)
        token = token[None]

    return token

def prefill_sync(
        model_wrapper,
        input_ids=None,
        past_key_values=None,
        logits_processor=None,
        prof=None,
):
    comm = model_wrapper.comm
    device = model_wrapper.device

    if not comm.is_server:  # drafter client
        # send input_ids to server
        comm.send_to(input_ids)
        # recv mixed hidden_state and the new token
        mixed_hidden_state = comm.recv_from(device=device)
        token = comm.recv_from(device=device)
        
        return token, mixed_hidden_state
        
    else:  # server with base model
        input_ids = comm.recv_from(device=device)  # todo: fix recv_from() on server
        outputs, orig, hidden_states = model_wrapper.base_model(
            input_ids, past_key_values=past_key_values, output_orig=True
        )
        # mixing multi-layer hidden state and compress transmission
        mixed_hidden_state = model_wrapper.eagle3_fc(hidden_states)
        comm.send_to(mixed_hidden_state)  # todo: fix send_to() on server
        
        token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
        comm.send_to(token)

        # no return for server


def catainfer(
        model_wrapper,
        kv_cache=None,
        logits_processor=None,
        input_ids=None,
        token=None,
        mixed_hidden_state=None,
        new_token=None,
        max_new_tokens=None,
        max_length=None,
        log=False,
        prof=None
):
    comm = model_wrapper.comm
    device = model_wrapper.device

    if not comm.is_server:  # drafter client
        input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        input_len = input_ids.shape[1]

        with prof.time_context(f'Drafter: topK_genrate', cpu=False) if prof is not None else nullcontext():
            # todo: fix topK_genrate of eagle3
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = model_wrapper.ea_layer.topK_genrate(
                mixed_hidden_state,
                input_ids_ea,
                model_wrapper.ea_layer.lm_head,
                logits_processor,
                total_tokens=run_config.total_tokens,
                depth=run_config.depth,
                top_k=run_config.top_k,
                return_last=run_config.none_expand,
                prof=prof,
            )
        tree_position_ids = tree_position_ids + input_ids.size(-1)
    
        





        



