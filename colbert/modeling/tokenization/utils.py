import torch
import random
import numpy as np
from colbert.modeling.tokenization.codeswtiching_method import golden_dictionary_codeswitch, mplm_dictionary_codeswitch

def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize, 
                      switching_prob, seed, lexicon_type=None, lexicons=None):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0
    
    N = len(queries)

    # Original Query & Postive & Negative pair
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    positive_ids, positive_mask = doc_tokenizer.tensorize(positives)
    negative_ids, negative_mask = doc_tokenizer.tensorize(negatives)
    
    # D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    # D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    # maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    # indices = maxlens.sort().indices
    # Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    # D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    # (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
    
    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)   

    stat = {}
    if lexicons['query'] is not None:
        if lexicon_type == "muse_golden":
            code_switched_queries, stat['query'] = golden_dictionary_codeswitch(queries, lexicons['query'], switching_prob, seed)
            code_switched_query_ids, code_switched_query_mask = query_tokenizer.tensorize(code_switched_queries)
            code_switched_query_batches = _split_into_batches(code_switched_query_ids, code_switched_query_mask, bsize)
        elif lexicon_type == "mplm_vocab":
            code_switched_query, query_cs_index, stat['query'] = mplm_dictionary_codeswitch(query_tokenizer, Q_ids, lexicons['lexicon_for_query'], switching_prob, seed)
            code_switched_query_batches = _split_into_batches(code_switched_query, Q_mask, bsize, query_cs_index)
    else:
        code_switched_query_batches = [(None,None)]

    if lexicons['doc'] is not None:
        if lexicon_type == "muse_golden":
            code_switched_positives, stat['positive'] = golden_dictionary_codeswitch(positives, lexicons['doc'], switching_prob, seed)
            code_switched_negatives, stat['negative'] = golden_dictionary_codeswitch(negatives, lexicons['doc'], switching_prob, seed)
            code_switched_positive_ids, code_switched_positive_mask = doc_tokenizer.tensorize(code_switched_positives)
            code_switched_negative_ids, code_switched_negative_mask = doc_tokenizer.tensorize(code_switched_negatives)    
            code_switched_positive_batches = _split_into_batches(code_switched_positive_ids, code_switched_positive_mask, bsize)
            code_switched_negative_batches = _split_into_batches(code_switched_negative_ids, code_switched_negative_mask, bsize)        
        elif lexicon_type == "mplm_vocab":
            code_switched_positive_ids, pos_cs_index, stat['positive'] = mplm_dictionary_codeswitch(doc_tokenizer, positive_ids, lexicons['lexicon_for_doc'], switching_prob, seed)
            code_switched_negative_ids, neg_cs_index, stat['negative'] = mplm_dictionary_codeswitch(doc_tokenizer, negative_ids, lexicons['lexicon_for_doc'], switching_prob, seed)
            code_switched_positive_batches = _split_into_batches(code_switched_positive_ids, positive_mask, bsize, pos_cs_index)
            code_switched_negative_batches = _split_into_batches(code_switched_negative_ids, negative_mask, bsize, neg_cs_index)
    else:
        code_switched_positive_batches = [(None,None)]
        code_switched_negative_batches = [(None,None)]

    return make_batch(query_batches, positive_batches, negative_batches, code_switched_query_batches, code_switched_positive_batches, code_switched_negative_batches), stat

def make_batch(query_batches, positive_batches, negative_batches, code_switched_query_batches=None, code_switched_positive_batches=None, code_switched_negative_batches=None):

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), (cs_q_ids, cs_q_mask), (cs_p_ids, cs_p_mask), (cs_n_ids, cs_n_mask) in zip(query_batches, positive_batches, negative_batches, code_switched_query_batches, code_switched_positive_batches, code_switched_negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))

        #CS_Q = (cs_q_ids, cs_q_mask, cs_q_idx)
        CS_Q = (cs_q_ids, cs_q_mask)
        if cs_p_ids is not None:
            #CS_D = (torch.cat((cs_p_ids, cs_n_ids)), torch.cat((p_mask, n_mask)), torch.cat((cs_p_idx, cs_n_idx)))
            CS_D = (torch.cat((cs_p_ids, cs_n_ids)), torch.cat((cs_p_mask, cs_n_mask)))
        else:
            CS_D = (cs_p_ids, cs_n_ids)
    
        batches.append((Q, D, CS_Q, CS_D))

    return batches

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
