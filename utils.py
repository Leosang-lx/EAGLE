import torch
from contextlib import nullcontext
import numpy as np
from typing import Tuple, List

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

        return input_ids

def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        # new_token,
        past_key_values_data_list,
        current_length_data,
        # model,
        hidden_state_new,
        sample_p,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    accept_indices = retrieve_indices[best_candidate, : accept_length]
    # select_indices = (
    #         retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    # )
    select_indices = accept_indices + prev_input_len
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length].to(input_ids.device)], dim=-1
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
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    # prob = sample_p
    # if logits_processor is not None:
    #     token = torch.multinomial(prob, 1)
    #     token = token[None]
    # else:
    #     token = torch.argmax(prob)
    #     token = token[None, None]
    
    next_token = gen_token(prob=sample_p, logits_processor=logits_processor)

    # new_token += accept_length + 1

    return accept_indices, input_ids, accept_hidden_state_new, next_token


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


def find_prefix_match(retrieve_indices, accept_indices):
    prefixes = retrieve_indices[:, :accept_indices.size(0)]

    matches = torch.all(prefixes == accept_indices.unsqueeze(0), dim=1)

    match_paths = torch.nonzero(matches).squeeze(1)

    return match_paths


def process_retrieve_indices(retrieve_indices):
    flattened = retrieve_indices.reshape(-1)
    mask = flattened != -1
    filtered = flattened[mask]

    unique_values = torch.unique(filtered)
    sorted_values = torch.sort(unique_values).values

    return sorted_values

# old version
def cal_pruning_info(draft_tokens, retrieve_indices, best_candidate, accept_len, new_token, subseq_ri_cum_depths):
    """
    Prune the token tree on the retrieve_indices
    :param retrieve_indices:
    :param best_candidate:
    :param accept_len:
    :return:
    """
    # accept_len > 0
    accepted_indices = retrieve_indices[best_candidate, :accept_len]

    # judge whether the global leaf node is reached
    cur_path_depth = (retrieve_indices[best_candidate, :] != -1).sum().item()
    if accept_len == retrieve_indices.size(-1) or retrieve_indices[best_candidate, accept_len] == -1:  # the next token of the accepted
        # truncate: reach the global leaf: occurs in the eagenerate_pruned_pipeline()
        # print('- leaf has been reached')
        return accepted_indices, True

    # judge whether the new token follows the tree
    matched_candidates = find_prefix_match(retrieve_indices, accepted_indices)  # non-zero
    next_indices_draft = retrieve_indices[matched_candidates, accept_len]  # todo: retrieve_indices会比last stage的draft tokens多，每次appended只会把append的token给到last stage
    next_tokens_draft = draft_tokens[0, next_indices_draft]
    # found the paths with prefix of "accept_tokens + new_token"

    same_indices = torch.nonzero(next_tokens_draft.cpu() == new_token.cpu()).squeeze(1)
    if same_indices.numel() == 0:
        # truncate: unmatched token
        # print(f'- - no match token found in the tree, accept_len/global_depth={accept_len}/{cur_path_depth}')
        # print(f'- - next_tokens_draft={next_tokens_draft.unique()}')
        # print(f'- - new_token={new_token}')
        return accepted_indices, True

    # pruning
    left_candidates = matched_candidates[same_indices]
    left_retrieve_indices = retrieve_indices[left_candidates, accept_len:]  # todo: left_retrieve_indices all -1

    # [update]: retrieve_indices may be larger than draft_tokens
    # left_indices_global: for the global context
    # left_indices: for the tree in pipeline
    max_depth = (left_retrieve_indices != -1).sum(dim=1).max().item()
    left_retrieve_indices = left_retrieve_indices[:, :max_depth]  # drop the all -1 parts
    left_indices_global = process_retrieve_indices(left_retrieve_indices)  # for pruning for other stages
    left_indices_global = torch.cat((accepted_indices, left_indices_global), dim=0)
    left_indices_from_zero = torch.arange(left_indices_global.size(-1) - accept_len, dtype=torch.long)

    left_indices = left_indices_global[left_indices_global < draft_tokens.size(-1)]

    return left_indices, False


def prune_retrieve_indices(draft_tokens, retrieve_indices, accept_indices, new_token):
    """
    Prune retrieve_indices
    return [the pruned and processed (mapping to indices starts from zero)] and [the left indices]
    """
    accept_len = len(accept_indices)
    matched_candidates = find_prefix_match(retrieve_indices, accept_indices)
    next_indices_draft = retrieve_indices[matched_candidates, accept_len]
    next_tokens_draft = draft_tokens[0, next_indices_draft]

    same_indices = torch.nonzero(next_tokens_draft.cpu() == new_token.cpu()).squeeze(1)  # try to accelerate with numpy
    if same_indices.numel() == 0:
        # truncate: the sampled next token is unmatched with the current
        return None, None

    left_candidates = matched_candidates[same_indices]
    left_retrieve_indices = retrieve_indices[left_candidates, accept_len + 1:]
    left_draft_indices = process_retrieve_indices(left_retrieve_indices)

    left_indices_from_zero = torch.arange(left_draft_indices.size(-1), dtype=torch.long)
    mapped_retrieve_indices = map_retrieve_indices(left_retrieve_indices, left_draft_indices, left_indices_from_zero)

    return mapped_retrieve_indices, left_draft_indices
 

def verifier_prune_draft(draft_tokens, tree_mask, tree_pos_ids, retrieve_indices, best_candidate, accept_indices, next_token):
    accept_len = len(accept_indices)
    # cur_path_depth = (retrieve_indices[best_candidate, :] != -1).sum().item()
    if accept_len == retrieve_indices.size(-1) or retrieve_indices[best_candidate, accept_len] == -1:
        # reach the leaf node
        return True, None
    
    # prune retrieve_indices and get left_indices (not including the accept_indices)
    pruned_retrieve_indices, left_draft_indices = prune_retrieve_indices(draft_tokens, retrieve_indices, accept_indices, next_token)
    if pruned_retrieve_indices is None:
        return True, None

    # # judge whether the new token follows the tree
    # matched_candidates = find_prefix_match(retrieve_indices, accept_indices)
    # next_indices_draft = retrieve_indices[matched_candidates, accept_len]
    # next_tokens_draft = draft_tokens[0, next_indices_draft]

    # same_indices = torch.nonzero(next_tokens_draft.cpu() == next_token.cpu()).squeeze(1)
    # if same_indices.numel() == 0:
    #     # truncate: unmatched token
    #     return True, None
    
    # ### draft pruning
    # # todo: cal left_indices to prune the draft
    # left_candidates = matched_candidates[same_indices]
    # left_retrieve_indices = retrieve_indices[left_candidates, accept_len + 1:]
    # left_draft_indices = process_retrieve_indices(left_retrieve_indices)

    # left_indices_from_zero = torch.arange(left_draft_indices.size(-1), dtype=torch.long)
    # mapped_retrieve_indices = map_retrieve_indices(left_retrieve_indices, left_draft_indices, left_indices_from_zero)

    draft_tokens = draft_tokens[:, left_draft_indices]
    tree_pos_ids = tree_pos_ids[:, left_draft_indices]
    tree_mask = tree_mask[..., left_draft_indices[:, None], left_draft_indices]

    return False, (draft_tokens, tree_mask, tree_pos_ids, pruned_retrieve_indices)


def drafter_prune_draft(draft_tokens, tree_mask, tree_pos_ids, retrieve_indices, accept_indices, next_token):
    # fixme: when the accepted tokens reach the leaf node
    pruned_retrieve_indices, left_draft_indices = prune_retrieve_indices(draft_tokens, retrieve_indices, accept_indices, next_token)
    if pruned_retrieve_indices is None:
        return True, None
    
    draft_tokens = draft_tokens[:, left_draft_indices]
    tree_pos_ids = tree_pos_ids[:, left_draft_indices]
    tree_mask = tree_mask[..., left_draft_indices[:, None], left_draft_indices]
    return False, (draft_tokens, tree_mask, tree_pos_ids, pruned_retrieve_indices)


def verifier_pruning(
        past_key_values_data_list,
        current_length_data,
        # lens_split,
        last_hidden_state,
        tree_mask,
        tree_pos_ids,
        left_indices,
        global_accept_len,
        accept_len,
        # stage,
):
    """
    prune the tokens related data: kv-cache, last_hidden_state, tree_mask
    [update] last_hidden_state is the output of last_rank
    """
    cur_kv_len = current_length_data[0].item()
    cache_device = past_key_values_data_list[0][0].device

    # prune cache
    # ***这里的global_accept_len还是没有加上accept_len长度的
    left_indices_global = left_indices + global_accept_len
    left_indices_in_cache = (left_indices_global[left_indices_global < cur_kv_len])
    left_indices_after_cache = left_indices_global[left_indices_in_cache.size(-1):]
    
    # copy and set cache length
    left_indices_in_cache_size = left_indices_in_cache.size(-1)
    for past_key_values_data in past_key_values_data_list:
        left_kv_cache = past_key_values_data[..., left_indices_in_cache, :]
        cache_dst = past_key_values_data[..., global_accept_len:global_accept_len+left_indices_in_cache_size, :]
        cache_dst.copy_(left_kv_cache, non_blocking=True)
    current_length_data.fill_(global_accept_len + left_indices_in_cache_size)

    # prune lens_split
    # if lens_split is not None:
    #     # test kv_len
    #     assert cur_kv_len == global_accept_len + torch.sum(lens_split[:5-stage]), f'stage{stage} wrong kv_len={cur_kv_len} while global_accept={global_accept_len} and lens_split={lens_split}'

    #     cum_lens = torch.cumsum(lens_split, dim=0)
    #     lens_split = torch.tensor([torch.sum((left_indices >= cum_lens[i-1]) & (left_indices < cum_lens[i])) for i in range(1, cum_lens.size(-1))], dtype=torch.long)

    # prune last_hidden_state
    # the last_hidden_state here are the output of the current stage_model
    if last_hidden_state is not None:
        cur_hs_len = last_hidden_state.size(1) 

        hs_indices_start_idx = cur_kv_len  # [update]: last_hidden_state is new to current stage
        hs_indices_end_idx = cur_kv_len + cur_hs_len
        left_indices_in_input = left_indices_after_cache[left_indices_after_cache < hs_indices_end_idx] - hs_indices_start_idx  # prune last_hidden_state, tree_mask, tree_pos_ids
        # left_indices_in_hs = left_indices_in_hs.to(last_hidden_state.device)
        if left_indices_in_input.numel() > 0:
            assert tree_pos_ids.size(0) == tree_mask.size(2) > max(left_indices_in_input), f'last_hidden_state.shape={last_hidden_state} tree_pos_ids.shape={tree_pos_ids.shape}, tree_mask.shape={tree_mask.shape}, left_indices_in_input={left_indices_in_input}'

        if left_indices_in_input.numel() > 0:
            assert torch.max(left_indices_in_input) < last_hidden_state.size(1), f'stage{stage} left_indices_in_input={left_indices_in_input} is out of range'
        if len(last_hidden_state.shape) == 3:
            last_hidden_state = last_hidden_state[..., left_indices_in_input.to(last_hidden_state.device), :]  # todo: bug occurs when hidden_state is empty tensor
        else:
            last_hidden_state = last_hidden_state[..., left_indices_in_input.to(last_hidden_state.device)]  # todo: bug occurs when hidden_state is empty tensor
        
    # prune tree_mask
    if tree_mask is not None:
        tree_mask_cpu = tree_mask.cpu()
        local_tree_mask_left_indices = left_indices_in_input
        global_tree_mask_left_indices = left_indices[accept_len:]
        global_tree_mask_left_indices = global_tree_mask_left_indices[global_tree_mask_left_indices < tree_mask_cpu.size(-1)]
        # assert torch.max(tree_mask_left_indices) < tree_mask_cpu.size(-1), f'stage{stage} tree_mask_left_indices={tree_mask_left_indices} is out of range'
        tree_mask = tree_mask_cpu[..., local_tree_mask_left_indices[:, None], global_tree_mask_left_indices].to(tree_mask.device)

    # prune tree_pos_ids
    if tree_pos_ids is not None:
        tree_pos_ids_cpu = tree_pos_ids.cpu()
        
        tree_pos_ids = tree_pos_ids_cpu[local_tree_mask_left_indices].to(tree_pos_ids.device)

    return past_key_values_data_list, current_length_data, last_hidden_state, tree_mask, tree_pos_ids





def merge_appended_draft(
    draft_tokens,
    tree_mask,
    tree_pos_ids,
    appended_draft_tokens,
    appended_tree_mask,
    appended_tree_pos_ids,
):
    draft_tokens = torch.cat((draft_tokens, appended_draft_tokens), dim=-1)
    tree_pos_ids = torch.cat((tree_pos_ids, appended_tree_pos_ids), dim=-1)
    # assert right mask shape
    orig_size = tree_mask.size(-1)
    appended_size, full_size = appended_tree_mask.size(-2), appended_tree_mask.size(-1)
    assert orig_size + appended_size == full_size, f'Unmatched tree_mask size with tree_mask1 shape: {tree_mask.shape} and tree_mask2 shape: {appended_tree_mask.shape}'
    # merge tree_mask
    merged_tree_mask = torch.zeros(full_size, full_size, dtype=tree_mask.dtype, device=tree_mask.device)
    merged_tree_mask[orig_size:, orig_size:].copy_(tree_mask, non_blocking=True)
    merged_tree_mask[orig_size:, :].copy_(appended_tree_mask[0, 0], non_blocking=True)

    return draft_tokens, merged_tree_mask, tree_pos_ids


def get_parent_indices_np(tree_mask):
    """
    Compute parent indices for each node in a tree using NumPy.
    
    Args:
        tree_mask (np.ndarray): Shape [n, n], boolean matrix indicating parent-child relationships.

    Returns:
        np.ndarray: Shape [n], where parent_indices[i] = j means j is the last parent of i, or -1 if no parent.
    """
    n = tree_mask.shape[0]
    
    tree_mask = tree_mask.astype(np.bool_)
    offset_mask = np.tri(n, n, k=-1, dtype=np.bool_)
    masked = np.logical_and(tree_mask, offset_mask)
    flipped_masked = np.fliplr(masked)
    col_indices = np.argmax(flipped_masked, axis=1)
    parent_indices = n - 1 - col_indices
    all_false_rows = ~flipped_masked[np.arange(n), col_indices]
    parent_indices[all_false_rows] = -1

    return parent_indices


def merge_two_tree(
        tree1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tree2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        # lens_split,
        # subseq_ri_cum_depths,
        prof=None
):
    """
    Merge two tree that share the same root node, tree1 is the old tree, tree2 is the new tree
    The merged tree has the draft_tokens of: {draft_tokens1, added_tokens}
    Ensure that **the whole tree is on CPU**
    """
    with prof.time_context(f"init", cpu=True) if prof is not None else nullcontext():
        draft_tokens1, retrieve_indices1, tree_mask1, tree_pos_ids1 = tree1
        draft_tokens2, retrieve_indices2, tree_mask2, tree_pos_ids2 = tree2

        tree1_depth = retrieve_indices1.size(1)
        tree2_depth = retrieve_indices2.size(1)
        tree_mask1 = tree_mask1[0, 0, ...]
        tree_mask2 = tree_mask2[0, 0, ...]
        tree1_size = draft_tokens1.size(-1)

        tree_mask1_np = tree_mask1.numpy()
        tree_mask2_np = tree_mask2.numpy()
        draft_tokens1_np = draft_tokens1[0].numpy()
        draft_tokens2_np = draft_tokens2[0].numpy()
        retrieve_indices1_np = retrieve_indices1.numpy()
        retrieve_indices2_np = retrieve_indices2.numpy()

    # get paths of draft_tokens1
    with prof.time_context(f"get paths of draft_tokens1", cpu=True) if prof is not None else nullcontext():
        tree_mask1_np = tree_mask1.numpy()
        tree1_token_paths_idx = [(tuple(draft_tokens1_np[np.flatnonzero(tree_mask1_np[i])]), i) for i in range(tree_mask1_np.shape[0])]
        paths_tree1_idx = dict(tree1_token_paths_idx)

    # go through draft_tokens2
    with prof.time_context(f"go through draft_tokens2", cpu=True) if prof is not None else nullcontext():
        # paths_tree2 = set()
        index_mapping_2_to_merged = np.zeros(draft_tokens2.size(1), dtype=np.int64)
        append_indices = []
        tree2_token_paths = [tuple(draft_tokens2_np[np.flatnonzero(tree_mask2_np[i])]) for i in range(tree_mask2_np.shape[0])]
        paths_tree2 = set(tree2_token_paths)

        for i, token_path in enumerate(tree2_token_paths):
            if len(token_path) <= tree1_depth and token_path in paths_tree1_idx:
                index_mapping_2_to_merged[i] = paths_tree1_idx[token_path]
            else:
                mapped_idx = tree1_size + len(append_indices)
                append_indices.append(i)
                index_mapping_2_to_merged[i] = mapped_idx

    append_length = len(append_indices)
    with prof.time_context(f"merge tokens and positions", cpu=True) if prof is not None else nullcontext():
        append_indices = torch.tensor(append_indices, dtype=torch.long)
        draft_tokens_merged = torch.cat((draft_tokens1, draft_tokens2[:, append_indices]), dim=1)
        merged_tree_pos_ids = torch.cat((tree_pos_ids1, tree_pos_ids2[append_indices]), dim=0)
    assert draft_tokens_merged.size(-1) == merged_tree_pos_ids.size(0), f'draft_tokens_merged != merged_tree_pos_ids: {draft_tokens_merged.size(-1)} and {merged_tree_pos_ids.size(0)}'
    # [merge tree_mask]
    with prof.time_context(f"merge tree_mask", cpu=True) if prof is not None else nullcontext():
        merged_size = draft_tokens_merged.size(-1)
        # init merged_tree_mask as tree_mask1
        merged_tree_mask = np.zeros((merged_size, merged_size), dtype=tree_mask1_np.dtype)
        merged_tree_mask[:tree1_size, :tree1_size] = tree_mask1_np

        with prof.time_context(f"get parent indices", cpu=True) if prof is not None else nullcontext():
            parent_indices = get_parent_indices_np(tree_mask2_np)
        with prof.time_context(f"iterative merge tree_mask", cpu=True) if prof is not None else nullcontext():
            for i, append_idx in enumerate(append_indices):
                mapped_idx = index_mapping_2_to_merged[append_idx]
                parent_idx = index_mapping_2_to_merged[parent_indices[append_idx]]
                mapped_parent_mask_row = merged_tree_mask[parent_idx, :parent_idx+1]
                merged_tree_mask[mapped_idx, :parent_idx+1] = mapped_parent_mask_row
                merged_tree_mask[mapped_idx, mapped_idx] = 1

    with prof.time_context(f"merge retrieve_indices", cpu=True) if prof is not None else nullcontext():
        leaf_depths1 = (retrieve_indices1_np != -1).sum(axis=1)
        leaf_depths2 = (retrieve_indices2_np != -1).sum(axis=1)
        leave_paths1 = [(tuple(draft_tokens1_np[retrieve_indices1_np[i, :leaf_depths1[i]]]), i) for i in range(retrieve_indices1_np.shape[0])]
        leave_paths1 = dict(leave_paths1)
        leave_paths2 = [(tuple(draft_tokens2_np[retrieve_indices2_np[i, :leaf_depths2[i]]]), i) for i in range(retrieve_indices2_np.shape[0])]
        leave_paths2 = dict(leave_paths2)

        selected_leaves1 = np.zeros(retrieve_indices1.size(0), dtype=np.bool_)
        selected_leaves2 = np.zeros(retrieve_indices2.size(0), dtype=np.bool_)
        for leaf_path, leaf_path_idx in leave_paths1.items():
            if leaf_path in paths_tree2 and leaf_path not in leave_paths2:
                pass
            else:
                selected_leaves1[leaf_path_idx] = True
        
        for leaf_path, leaf_path_idx in leave_paths2.items():
            if leaf_path not in paths_tree1_idx:
                selected_leaves2[leaf_path_idx] = True

        tree1_selected_sum = selected_leaves1.sum()
        tree2_selected_sum = selected_leaves2.sum()
        merged_selected_sum = tree1_selected_sum + tree2_selected_sum
        max_depth = max(tree1_depth, tree2_depth)
        ri_merged = np.full((merged_selected_sum, max_depth), -1, dtype=np.int64)
        ri_merged[:tree1_selected_sum, :tree1_depth] = retrieve_indices1_np[selected_leaves1]
        ri2_merged = retrieve_indices2_np[selected_leaves2]
        valid_mask = ri2_merged != -1
        ri2_merged[valid_mask] = index_mapping_2_to_merged[ri2_merged[valid_mask]]
        
        ri_merged[tree1_selected_sum:merged_selected_sum, :tree2_depth] = ri2_merged
        retrieve_indices_merged = torch.from_numpy(ri_merged)
    # [merge retrieve_indices] finish

    # # update lens_split and subseq_ri_cum_depths
    # with prof.time_context(f"update lens_split and subseq_ri_cum_depths", cpu=True) if prof is not None else nullcontext():
    #     lens_split = torch.cat((lens_split, torch.tensor([append_indices.size(0)], dtype=torch.long)))
    #     # todo: len_split多长？subseq_ri_cum_depths应该多长？
    #     n_leaves = retrieve_indices_merged.size(0)
    #     subseq_ri_cum_depths = []
    #     cum_seq_lens = np.cumsum(lens_split[:-1].numpy(), axis=0)
    #     bottom = np.full((n_leaves, 1), -1, dtype=np.int64)
    #     retrieve_indices_filled = np.concatenate((retrieve_indices_merged.numpy(), bottom), axis=1)  # add -1 to bottom to prevent overflow

    #     ri_depth_cum = np.zeros(n_leaves, dtype=np.int64)
    #     for i, cum_seq_len in enumerate(cum_seq_lens):
    #         for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
    #             row_indices = np.arange(n_leaves, dtype=np.int64)
    #             cum_ri_leaves = retrieve_indices_filled[row_indices, ri_depth_cum]
    #             ri_depth_cum[cum_ri_leaves == j] += 1
    #         # update: 只计算到在pipeline里的draft token tree部分，即将输入的最新一段单独算
    #         subseq_ri_cum_depths.append(ri_depth_cum.copy())
    #     subseq_ri_cum_depths = np.stack(subseq_ri_cum_depths, axis=0)
    
    # return draft_tokens_merged, retrieve_indices_merged, torch.from_numpy(merged_tree_mask)[None, None], merged_tree_pos_ids, lens_split, torch.from_numpy(subseq_ri_cum_depths)
    return draft_tokens_merged, retrieve_indices_merged, torch.from_numpy(merged_tree_mask)[None, None], merged_tree_pos_ids, append_length

