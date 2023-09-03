import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize, _use_gradient_reversal, sources, targets):
    assert len(queries) == len(positives) == len(negatives) # == len(sources) == len(targets)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)
    
    # parallel sentences (source & target) tensorize
    if _use_gradient_reversal == True:
        SRC_ids, SRC_mask = doc_tokenizer.tensorize(sources)
        TRG_ids, TRG_mask = doc_tokenizer.tensorize(targets)

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
    
    batches = []
    
    if _use_gradient_reversal == True:
        source_batches = _split_into_batches(SRC_ids, SRC_mask, bsize)
        target_batches = _split_into_batches(TRG_ids, TRG_mask, bsize)
        for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask), (s_ids, s_mask), (t_ids, t_mask) in zip(query_batches, positive_batches, negative_batches, source_batches, target_batches):
            Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
            D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
            S = (torch.cat((s_ids, s_ids)), torch.cat((s_mask, s_mask)))
            T = (torch.cat((t_ids, t_ids)), torch.cat((t_mask, t_mask)))

            #S = (s_ids, s_mask)
            #T = (t_ids, t_mask)
            batches.append((Q, D, S, T))  
    else:
        #source_batches = (1,1)
        #target_batches = (1,1)

        for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches, positive_batches, negative_batches):
            Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
            D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
            batches.append((Q, D, None, None))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
