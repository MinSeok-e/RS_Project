from dataclasses import replace

import json

from types import SimpleNamespace

import numpy as np

from scipy import sparse

import torch

from .common import fast_topk, weighted_history

from .linear import ease_predict, user_knn, transitions

from .ensemble import transform

from .sequence_training import make_windows, full_catalog_loss

from .sasrec import SASRec, inference_array, predict

from .models import SLIM

def context(data, rows, future):
    d=replace(data,train_rows=np.asarray(rows),valid_rows=np.asarray(future))
    x=d.matrix(d.train_rows,dtype=np.float32)
    histories=[d.items[d.train_rows[d.users[d.train_rows]==u]] for u in range(d.shape[0])]
    return SimpleNamespace(data=d,x=x,histories=histories,seen=x.astype(bool),active=x.any(0))

def inner_context(data,fraction):
    past,future=[],[]
    for u in range(data.shape[0]):
        rows=data.train_rows[data.users[data.train_rows]==u]
        n=max(2,min(len(rows)-1,int(len(rows)*fraction)))
        past.extend(rows[:n]);future.extend(rows[n:])
    if set(past)&set(future) or (set(past)|set(future))!=set(data.train_rows):
        raise AssertionError('Inner folds must partition only outer training rows')
    return context(data,past,future)

def fixed_sas(ctx,seed,epochs):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    windows=make_windows(ctx.histories,100,'coverage',5,50)
    active=torch.as_tensor(np.flatnonzero(ctx.active)+1,device='cuda')
    model=SASRec(ctx.x.shape[1],maxlen=100,dim=64,blocks=2,heads=2,dropout=.2).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.98))
    for _ in range(epochs):
        model.train();order=rng.permutation(len(windows.users))
        for offset in range(0,len(order),128):
            idx=order[offset:offset+128]
            tokens=torch.as_tensor(windows.tokens[idx],device='cuda')
            labels=torch.as_tensor(windows.targets[idx],device='cuda')
            loss=full_catalog_loss(model,tokens,labels,active)
            if not torch.isfinite(loss):raise ValueError('Nonfinite inner SAS loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    scores=predict(model,inference_array(ctx.histories,100),torch.device('cuda'))
    del model,opt;torch.cuda.empty_cache()
    return scores

def base_scores(ctx,outer_root=None,smoke=False):
    x=ctx.x;eligible=~ctx.seen & ctx.active[None,:];out={}
    q=weighted_history(ctx,half_life=20)
    out['ease']=ease_predict(x,x,300).astype(np.float32)
    out['recent_ease']=ease_predict(x,q,300).astype(np.float32)
    r=weighted_history(ctx,rating_power=2)
    out['rating_ease']=ease_predict(r,r,100).astype(np.float32)
    slim=SLIM(alpha=.3,l1_ratio=.01,topk=100,workers=2).fit(x)
    out['slim']=slim.predict(x).astype(np.float32);del slim
    out['userknn']=user_knn(x,q,x,20,20).astype(np.float32)
    for horizon in [5,20,50]:
        counts=transitions(ctx.histories,x.shape[1],horizon)
        degrees=np.asarray(counts.sum(1)).ravel()
        weights=sparse.diags(1/np.maximum(degrees,1))@counts
        out[f'markov{horizon}']=(sparse.csr_matrix(weighted_history(ctx,tail=10))@weights).toarray()
    bags=[]
    old_runs={2020:81089,2021:81090,2022:81091}
    for seed,epochs in [(2020,120),(2021,110),(2022,95)]:
        if outer_root:
            run=outer_root/f'results/sasrec_dev_{old_runs[seed]}'
            manifest=json.loads((run/'manifest.json').read_text())
            if manifest['data']['file_sha256']!=ctx.data.hashes:
                raise AssertionError('Saved SAS data mismatch')
            with np.load(run/f'future5_ce_s{seed}/predictions.npz') as f:scores=f['scores']
        else:scores=fixed_sas(ctx,seed,2 if smoke else epochs)
        bags.append(transform(scores,eligible,'zscore'))
    out['sasbag']=np.mean(bags,axis=0)
    return out

def make_features(ctx,scores,with_labels):
    eligible=~ctx.seen & ctx.active[None,:]
    pool_models=['sasbag','slim','recent_ease','rating_ease','markov20','userknn']
    pools=[fast_topk(scores[name],ctx.seen,ctx.active,k=100) for name in pool_models]
    candidates=[np.unique(np.concatenate([pool[u] for pool in pools])) for u in range(len(ctx.x))]
    groups=np.array([len(c) for c in candidates],dtype=np.int32)
    users=np.repeat(np.arange(len(ctx.x)),groups);items=np.concatenate(candidates)
    features=[];names=[]
    for name,a in scores.items():
        z=transform(a,eligible,'zscore');rrf=transform(a,eligible,'rrf')
        features.extend([a[users,items],z[users,items],rrf[users,items]])
        names.extend([name,name+'_z',name+'_rrf'])
    # Simple fold-local item and user statistics; no ID memorization features.
    pop=ctx.x.sum(0);rowlen=ctx.x.sum(1)
    rating=np.zeros_like(ctx.x);relative=np.zeros_like(ctx.x)
    for u,h in enumerate(ctx.histories):
        rows=ctx.data.train_rows[ctx.data.users[ctx.data.train_rows]==u]
        rating[u,h]=ctx.data.ratings[rows]
        relative[u,h]=np.arange(len(h))/max(len(h)-1,1)
    stats={'log_pop':np.log1p(pop)[items],
           'item_rating_mean':(rating.sum(0)/np.maximum(pop,1))[items],
           'item_mean_position':(relative.sum(0)/np.maximum(pop,1))[items],
           'recent_item_frequency':((relative>.8).sum(0)/np.maximum(pop,1))[items],
           'log_user_history':np.log1p(rowlen)[users],
           'user_mean_rating':(rating.sum(1)/rowlen)[users],
           'user_mean_pop':((ctx.x*pop).sum(1)/rowlen)[users]}
    features.extend(stats.values());names.extend(stats)
    f=np.stack(features,axis=1).astype(np.float32)
    if not np.isfinite(f).all():raise ValueError('Nonfinite ranking features')
    result={'features':f,'groups':groups,'users':users,'items':items}
    if with_labels:
        truth=ctx.data.matrix(ctx.data.valid_rows,dtype=bool)
        if (truth&ctx.seen).any():raise AssertionError('Inner labels overlap training')
        result['labels']=truth[users,items].astype(np.int32)
    return result,names
