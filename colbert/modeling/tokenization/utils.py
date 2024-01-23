import torch
import random
import numpy as np

def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize, 
                      switching_prob, seed, lexicons=None):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
    
    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)   

    stat = {}
    if 'lexicon_for_query' in lexicons.keys():
        code_switched_query, query_cs_index, stat['query'] = code_switch_bilingually(query_tokenizer, Q_ids, lexicons['lexicon_for_query'], seed, switching_prob)
        code_switched_query_batches = _split_into_batches(code_switched_query, Q_mask, bsize, query_cs_index)
    else:
        code_switched_query_batches = [(None,None)]

    if 'lexicon_for_doc' in lexicons.keys():
        code_switched_positive_ids, pos_cs_index, stat['positive'] = code_switch_bilingually(doc_tokenizer, positive_ids, lexicons['lexicon_for_doc'], seed, switching_prob)
        code_switched_negative_ids, neg_cs_index, stat['negative'] = code_switch_bilingually(doc_tokenizer, negative_ids, lexicons['lexicon_for_doc'], seed, switching_prob)
        code_switched_positive_batches = _split_into_batches(code_switched_positive_ids, positive_mask, bsize, pos_cs_index)
        code_switched_negative_batches = _split_into_batches(code_switched_negative_ids, negative_mask, bsize, neg_cs_index)
    else:
        code_switched_positive_batches = [(None,None)]
        code_switched_negative_batches = [(None,None)]

    return make_batch(query_batches, positive_batches, negative_batches, code_switched_query_batches, code_switched_positive_batches, code_switched_negative_batches), stat

def make_batch(query_batches, positive_batches, negative_batches, code_switched_query_batches=None, code_switched_positive_batches=None, code_switched_negative_batches=None):

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), (cs_q_ids, cs_q_idx), (cs_p_ids, cs_p_idx), (cs_n_ids, cs_n_idx) in zip(query_batches, positive_batches, negative_batches, code_switched_query_batches, code_switched_positive_batches, code_switched_negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))

        CS_Q = (cs_q_ids, q_mask, cs_q_idx)
        if cs_p_ids is not None:
            CS_D = (torch.cat((cs_p_ids, cs_n_ids)), torch.cat((p_mask, n_mask)), torch.cat((cs_p_idx, cs_n_idx)))
        else:
            CS_D = (cs_p_ids, cs_n_ids)
    
        batches.append((Q, D, CS_Q, CS_D))

    return batches

def code_switch_bilingually(tokenizer, token_ids, lexicon, seed, switching_prob=0.5):
    random.seed(seed)
    code_switched_token_ids = token_ids.clone()
    cs_index = torch.zeros(token_ids.size(0), token_ids.size(1))
    
    token_count_list = []
    cs_count_list = []
    for query_idx in range(token_ids.size(0)):
        token_count = 0
        cs_count = 0
        for token_idx in range(token_ids.size(1)):
            token_id = code_switched_token_ids[query_idx, token_idx].item()
            if token_id in (tokenizer.tok.all_special_ids + [5, 250002, 250003]):
                continue
            token_count += 1
            if str(token_id) in lexicon and random.random() < switching_prob:
                cs_count+=1
                code_switched_token_ids[query_idx, token_idx] = lexicon[str(token_id)][0]
                cs_index[query_idx, token_idx] = 1
        token_count_list.append(token_count)
        cs_count_list.append(cs_count)
    
    stat = {'num_token': np.mean(token_count_list), 'num_cs_token': np.mean(cs_count_list)}
    return code_switched_token_ids, cs_index, stat

def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize, cs_index=None):
    batches = []
    if cs_index is not None:
        for offset in range(0, ids.size(0), bsize):
            batches.append((ids[offset:offset+bsize], cs_index[offset:offset+bsize]))
        return batches
    
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
