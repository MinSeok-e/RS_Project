import numpy as np

from scipy import sparse

from .common import weighted_history

from .linear import transitions

from .ensemble import transform

def recent_similarity(ctx,users,items):
    x=ctx.x;pop=x.sum(0);features=[];names=[]
    fields=['cosine_last','cosine_max','cosine_top3','cosine_mean5','cosine_mean20',
            'jaccard_last','jaccard_max','jaccard_top3','jaccard_mean5','jaccard_mean20',
            'lift_last','lift_max','lift_top3','lift_mean5','lift_mean20']
    output=np.zeros((len(users),len(fields)),np.float32)
    for u,h in enumerate(ctx.histories):
        idx=np.flatnonzero(users==u);candidates=items[idx];recent=h[-20:]
        overlap=x[:,candidates].T@x[:,recent]
        matrices=[overlap/np.maximum(np.sqrt(pop[candidates,None]*pop[None,recent]),1),
                  overlap/np.maximum(pop[candidates,None]+pop[None,recent]-overlap,1),
                  overlap/np.maximum(pop[candidates,None]*pop[None,recent],1)]
        col=0
        for sim in matrices:
            output[idx,col]=sim[:,-1];output[idx,col+1]=sim.max(1)
            output[idx,col+2]=np.sort(sim,axis=1)[:,-min(3,len(recent)):].mean(1)
            output[idx,col+3]=sim[:,-5:].mean(1);output[idx,col+4]=sim.mean(1);col+=5
    return output,fields

def directional_features(ctx,users,items):
    eligible=~ctx.seen & ctx.active[None,:];features=[];names=[]
    # Row/column normalization has a different interpretation for directed edges.
    for horizon in [1,5,20,50]:
        counts=transitions(ctx.histories,ctx.x.shape[1],horizon)
        degree=np.asarray(counts.sum(1)).ravel();incoming=np.asarray(counts.sum(0)).ravel()
        weights=sparse.diags(1/np.maximum(degree,1))@counts
        for tail in [1,3,5,20,50]:
            score=(sparse.csr_matrix(weighted_history(ctx,tail=tail))@weights).toarray()
            for kind in ['zscore','rrf']:
                values=transform(score,eligible,kind)
                features.append(values[users,items]);names.append(f'directed_h{horizon}_tail{tail}_{kind}')
        # Population baseline lets the ranker distinguish frequent from selective transitions.
        features.append(np.log1p(incoming)[items]);names.append(f'directed_h{horizon}_incoming')
    return np.stack(features,axis=1).astype(np.float32),names
