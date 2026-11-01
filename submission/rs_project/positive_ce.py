import torch

def positive_aware_loss(model,tokens,targets,active_tokens,user_times,positions,mode):
    mask=targets[...,0]>0
    hidden=model.encode(tokens)[mask]
    logits=hidden@model.item_embedding(active_tokens).T
    row_times=user_times[:,None,:].expand(-1,tokens.shape[1],-1)[mask]
    row_positions=positions[mask]
    if mode=='past':excluded=row_times<row_positions[:,None]
    elif mode=='all_positive':excluded=row_times>=0
    else:raise ValueError('Unknown exclusion mode')
    # Never-observed items have time -1 and must remain eligible negatives.
    excluded &= row_times>=0
    wanted=targets[mask];valid=wanted>0
    lookup=torch.where(valid,wanted,active_tokens[0])
    indices=torch.searchsorted(active_tokens,lookup.contiguous())
    if (indices>=len(active_tokens)).any() or not torch.equal(active_tokens[indices],lookup):
        raise ValueError('Target outside active fit catalog')
    rows=torch.arange(len(indices),device=indices.device)[:,None].expand_as(indices)
    excluded[rows[valid],indices[valid]]=False
    log_probs=torch.log_softmax(logits.masked_fill(excluded,-torch.inf),dim=-1)
    selected=log_probs.gather(1,indices).masked_fill(~valid,0)
    return -(selected.sum(1)/valid.sum(1)).mean()
