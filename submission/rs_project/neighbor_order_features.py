import numpy as np

def aligned_neighbor_scores(histories,similarity,tail,k,horizon,anchor_mode='last_query'):
    n_users,n_items=similarity.shape[0],max(int(h.max()) for h in histories)+1
    positions=np.full((n_users,n_items),-1,np.int32);lengths=np.array([len(h) for h in histories])
    event_users=np.repeat(np.arange(n_users),lengths);event_items=np.concatenate(histories)
    event_positions=np.concatenate([np.arange(n) for n in lengths])
    for u,h in enumerate(histories):positions[u,h]=np.arange(len(h))
    result=np.zeros((n_users,n_items),np.float32)
    for u,h in enumerate(histories):
        matches=positions[:,h[-tail:]];valid=matches>=0
        if anchor_mode=='last_query':
            last=np.where(valid,np.arange(matches.shape[1])[None,:],-1).max(1)
            anchor=np.take_along_axis(matches,last.clip(0)[:,None],axis=1).ravel()
        else:anchor=matches.max(1)
        weights=similarity[u].copy();weights[(anchor<0)|(lengths<=anchor+1)]=0;weights[u]=0
        if k<len(weights):
            selected=np.argsort(-weights,kind='stable')[:k];kept=np.zeros_like(weights);kept[selected]=weights[selected];weights=kept
        weights=weights**2
        delta=event_positions-anchor[event_users]
        eligible=(delta>0)&(delta<=horizon)&(anchor[event_users]>=0)
        remaining=np.minimum(lengths-anchor-1,horizon).clip(1)
        event_weights=weights[event_users]/remaining[event_users]*eligible
        result[u]=np.bincount(event_items,weights=event_weights,minlength=n_items)
    return result

def neighbor_features(ctx,users,items,smoke=False):
    pop=ctx.x.sum(0).clip(1);scale=pop**(-.25);z=ctx.x*scale
    z/=np.linalg.norm(z,axis=1,keepdims=True).clip(1e-10)
    values=[];names=[]
    for query in ['all','recent20']:
        q=np.zeros_like(ctx.x)
        for u,h in enumerate(ctx.histories):q[u,h if query=='all' else h[-20:]]=1
        q*=scale;q/=np.linalg.norm(q,axis=1,keepdims=True).clip(1e-10)
        sim=q@z.T;np.fill_diagonal(sim,0)
        for tail in ([5] if smoke else [5,20]):
            for k in ([20] if smoke else [20,len(ctx.histories)]):
                for horizon in ([20] if smoke else [5,20,50]):
                    for anchor in (['last_query'] if smoke else ['last_query','max_neighbor']):
                        scores=aligned_neighbor_scores(ctx.histories,sim,tail,k,horizon,anchor)
                        # A column can be absent from this past fold, yet appear in the global mapping.
                        if scores.shape[1]<ctx.x.shape[1]:scores=np.pad(scores,((0,0),(0,ctx.x.shape[1]-scores.shape[1])))
                        values.append(scores[users,items]);names.append(f'aligned_neighbor_{query}_tail{tail}_k{k}_h{horizon}_{anchor}')
    return np.stack(values,1).astype(np.float32),names
