import torch
from contextlib import nullcontext

def prof_or_null(name, prof=None, cpu=False):
    if prof is not None:
        return prof.time_context(name, cpu=cpu)
    else:
        return nullcontext()

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
        orig, mixed_hidden_state = model_wrapper(
            input_ids, past_key_values, output_orig=True
        )
        comm.send_to(mixed_hidden_state)  # todo: fix send_to() on server
        
        token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
        comm.send_to(token)

        # no return for server


def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        # model,
        hidden_state_new,
        sample_p,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    
    new_token += accept_length + 1

    return input_ids, new_token, None, token


def map_retrieve_indices(retrieve_indices, a, b):
    # consider a is sorted, transform elements in retrieve_indices by mapping a->b
    assert a.size(0) == b.size(0), f'a.size(0)={a.size(0)}, b.size(0)={b.size(0)}'
    flat = retrieve_indices.reshape(-1)
    mask = flat != -1
    if not mask.any():
        return torch.full_like(retrieve_indices, -1)
    indices = torch.searchsorted(a, flat[mask])
    valid_mask = indices < len(a)
    result = torch.full_like(flat, -1)
    result[mask] = b[indices[valid_mask]]
    return result.view(retrieve_indices.shape)


