import torch

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
        mixed_hidden_state = model_wrapper.eagle3_fc(hidden_states)
        comm.send_to(mixed_hidden_state)  # todo: fix send_to() on server
        
        token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
        comm.send_to(token)

        # no return for server





        



