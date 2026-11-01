"""Fixed-epoch refits of the selected base recommenders, with no holdout access."""

from pathlib import Path

import numpy as np

import torch

from torch.nn import functional as F

from .rerank import context

from .common import weighted_history

from .neural import Autoencoder, temporal_examples, infer_features, GraphFactor

from .factorization import sample_unseen

from .sequence import GRURec

from .sasrec import SASRec, inference_array, predict

from .sequence_training import make_windows, full_catalog_loss

from .positive_ce import positive_aware_loss

from .rating_sequence import RatingSASRec, RatedBatch, ratings_for_tokens, predict_rated

from .filter_sequence import FilterRec, sample_prefixes, configured_prefix_batch, inference_tokens

from .multi_interest import MultiInterest, interest_loss

def full_context(data):
    rows=[]
    for u in range(data.shape[0]):
        ids=np.flatnonzero(data.users==u);rows.extend(ids[np.argsort(data.times[ids],kind='stable')])
    ctx=context(data,np.asarray(rows,dtype=np.int64),np.empty(0,dtype=np.int64))
    if int(ctx.x.sum())!=len(data.users) or ctx.data.valid_rows.size:
        raise AssertionError('Full fitting context must contain every provided training row and no holdout')
    return ctx

def seed_all(seed=2020):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)

def step(loss,model,optimizer,clip=True):
    if not torch.isfinite(loss):raise ValueError('Nonfinite fixed-refit loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    if clip:torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer.step()

def fit_sequence(ctx,c,epochs,kind='sas',seed=2020):
    rng=seed_all(seed);maxlen=c.get('maxlen',100);horizon=c.get('horizon',5)
    windows=make_windows(ctx.histories,maxlen,'coverage',horizon,c.get('stride',50))
    active_np=np.flatnonzero(ctx.active);active=torch.as_tensor(active_np+1,device='cuda')
    arch=dict(maxlen=maxlen,dim=c['dim'],blocks=c.get('blocks',2 if kind!='gru' else 1),heads=c.get('heads',2),dropout=c.get('dropout',.2))
    if kind=='rated':model=RatingSASRec(ctx.x.shape[1],rating_scale=c['rating_scale'],**arch).cuda()
    else:model=(GRURec if kind=='gru' else SASRec)(ctx.x.shape[1],**arch).cuda()
    if kind=='proportional':
        from .horizon_features import proportional_targets
        windows.targets=proportional_targets(windows.targets,windows.positions)
    past=c.get('mode')=='past' or kind=='proportional'
    times=np.full(ctx.x.shape,-1,np.int32)
    for u,h in enumerate(ctx.histories):times[u,h]=np.arange(len(h))
    times=torch.as_tensor(times[:,active_np],device='cuda')
    ratings=None
    if kind=='rated':
        ratings=np.zeros(ctx.x.shape,np.int64);r=ctx.data.train_rows
        ratings[ctx.data.users[r],ctx.data.items[r]]=np.rint(ctx.data.ratings[r]*2).astype(np.int64)
        ratings=torch.as_tensor(ratings,device='cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=c.get('lr',c.get('learning_rate',.001)),betas=(.9,.98),weight_decay=c.get('weight_decay',0.))
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(windows.users))
        for start in range(0,len(order),128):
            ix=order[start:start+128];tokens=torch.as_tensor(windows.tokens[ix],device='cuda');targets=torch.as_tensor(windows.targets[ix],device='cuda')
            users=torch.as_tensor(windows.users[ix],device='cuda')
            batch=RatedBatch(model,ratings_for_tokens(ratings,users,tokens)) if kind=='rated' else model
            if past:loss=positive_aware_loss(batch,tokens,targets,active,times[users],torch.as_tensor(windows.positions[ix],device='cuda'),'past')
            else:loss=full_catalog_loss(batch,tokens,targets,active,c.get('temperature',1.),c.get('smooth',c.get('label_smoothing',0.)))
            step(loss,model,optimizer)
    if kind=='rated':scores=predict_rated(model,torch.as_tensor(inference_array(ctx.histories,maxlen),device='cuda'),ratings)
    else:scores=predict(model,inference_array(ctx.histories,maxlen),torch.device('cuda'))
    return model,scores

def fit_prefix(ctx,c,epochs,multi=False):
    rng=seed_all(c.get('seed',2020));active_np=np.flatnonzero(ctx.active);active=torch.as_tensor(active_np+1,device='cuda')
    lengths=np.array([len(h) for h in ctx.histories]);times=np.full(ctx.x.shape,-1,np.int32)
    for u,h in enumerate(ctx.histories):times[u,h]=np.arange(len(h))
    times=torch.as_tensor(times[:,active_np],device='cuda')
    model=(MultiInterest(ctx.x.shape[1],c['dim'],c['length'],c['heads'],c['dropout']) if multi else
           FilterRec(ctx.x.shape[1],c['length'],c['dim'],c['dropout'],c['layers'])).cuda()
    if c.get('initialization')=='svd':
        from .filter_initialization import initialize_filter_items
        initialize_filter_items(model,ctx.x,c)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.0001)
    for epoch in range(epochs):
        model.train();users,ends=sample_prefixes(lengths,c,rng)
        tokens,targets=configured_prefix_batch(ctx.histories,users,ends,c)
        for start in range(0,len(users),256):
            token=torch.as_tensor(tokens[start:start+256],device='cuda');wanted=torch.as_tensor(targets[start:start+256],device='cuda');valid=wanted>0
            logits=model.encode(token)@model.items(active).T
            if c['exclude_past']:
                u=torch.as_tensor(users[start:start+256],device='cuda');end=torch.as_tensor(ends[start:start+256],device='cuda')
                excluded=(times[u]>=0)&(times[u]<end[:,None])
                logits=logits.masked_fill(excluded[:,None,:] if multi else excluded,-torch.inf)
            lookup=torch.where(valid,wanted,active[0]);indices=torch.searchsorted(active,lookup.contiguous())
            if (indices>=len(active)).any() or not torch.equal(active[indices],lookup):raise AssertionError('Training target outside fit catalog')
            if multi:loss=interest_loss(logits,indices,valid,c['objective'])
            else:
                logp=torch.log_softmax(logits,dim=1).gather(1,indices).masked_fill(~valid,0)
                loss=-(logp.sum(1)/valid.sum(1)).mean()
            step(loss,model,optimizer)
    model.eval();tokens=torch.as_tensor(inference_tokens(ctx.histories,c['length']),device='cuda')
    with torch.no_grad():
        if multi:
            scores=torch.cat([model.score(tokens[j:j+128],'mixture' if c['objective']=='mixture' else 'max',torch.as_tensor(~ctx.seen[j:j+128]&ctx.active[None,:],device='cuda')) for j in range(0,len(tokens),128)]).cpu().numpy()
        else:scores=torch.cat([model.score(tokens[j:j+128]) for j in range(0,len(tokens),128)]).cpu().numpy()
    return model,scores

def fit_autoencoder(ctx,c,epochs):
    rng=seed_all();active=np.flatnonzero(ctx.active);mapping=np.full(ctx.x.shape[1],-1,np.int64);mapping[active]=np.arange(len(active))
    histories=[mapping[h] for h in ctx.histories]
    inputs,targets=temporal_examples(histories,len(active),c['horizon'],c['tail'],c['half_life'])
    infer=infer_features(histories,len(active),c['tail'],c['half_life'])
    inputs,targets,infer=[torch.as_tensor(a,device='cuda') for a in (inputs,targets,infer)]
    model=Autoencoder(len(active),c['hidden'],c['latent'],c['dropout'],False).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(inputs))
        for start in range(0,len(order),128):
            ix=torch.as_tensor(order[start:start+128],device='cuda');logits,_=model(inputs[ix])
            loss=-(F.log_softmax(logits,dim=-1)*targets[ix]).sum(1).mean();step(loss,model,optimizer)
    model.eval()
    with torch.no_grad():a=torch.cat([model(infer[j:j+128])[0] for j in range(0,len(infer),128)]).cpu().numpy()
    scores=np.zeros_like(ctx.x);scores[:,active]=a
    return model,scores

def fit_bpr(ctx,c,epochs):
    seed_all();active=np.flatnonzero(ctx.active);x=torch.as_tensor(ctx.x[:,active],device='cuda')
    pairs=torch.nonzero(x,as_tuple=False);model=GraphFactor(x,c['dim'],c.get('layers',0)).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=c.get('lr',.003))
    for epoch in range(epochs):
        order=torch.randperm(len(pairs),device='cuda')
        for start in range(0,len(order),8192):
            batch=pairs[order[start:start+8192]];users,positive=batch[:,0],batch[:,1];negative=sample_unseen(x,users)
            u,i=model.embeddings();difference=(u[users]*(i[positive]-i[negative])).sum(1)
            penalty=(model.user[users].square().sum()+model.item[positive].square().sum()+model.item[negative].square().sum())/(2*len(users))
            step(F.softplus(-difference).mean()+c['reg']*penalty,model,optimizer,clip=False)
    model.eval()
    with torch.no_grad():a=model().cpu().numpy()
    scores=np.zeros_like(ctx.x);scores[:,active]=a
    return model,scores

def fit_masked(ctx,c,epochs):
    from .bert import MaskedRec,fit_windows,mask_inputs,score
    rng=seed_all();active=torch.as_tensor(np.flatnonzero(ctx.active)+1,device='cuda')
    train=fit_windows(ctx.histories,c['maxlen']);model=MaskedRec(ctx.x.shape[1],c['maxlen'],c['dim'],c['dropout']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01)
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(train))
        for start in range(0,len(order),128):
            raw=train[order[start:start+128]];inputs,selected=mask_inputs(raw,model.mask_id,c['mask_rate'],rng)
            h=model.encode(torch.as_tensor(inputs,device='cuda'))[torch.as_tensor(selected,device='cuda')]
            labels=torch.as_tensor(raw[selected],device='cuda');logits=h@model.item(active).T+model.bias[active-1]
            step(F.cross_entropy(logits,torch.searchsorted(active,labels)),model,optimizer)
    return model,score(model,ctx.histories,c['maxlen'],c['inference_masks'])

def active_context(ctx):
    active=np.flatnonzero(ctx.active);mapping=np.full(ctx.x.shape[1],-1,np.int64);mapping[active]=np.arange(len(active))
    return active,[mapping[h] for h in ctx.histories],torch.as_tensor(ctx.x[:,active],device='cuda')

def history_tail(histories,length):
    out=np.zeros((len(histories),length),np.int64)
    for u,h in enumerate(histories):
        tail=h[-length:]+1;out[u,-len(tail):]=tail
    return torch.as_tensor(out,device='cuda')

def expand_active(ctx,active,values):
    scores=np.zeros_like(ctx.x);scores[:,active]=values
    return scores

def fit_hybrid(ctx,c,epochs):
    from .hybrid_markov import HybridMarkov,examples
    if c['objective']!='ce':raise ValueError('Only selected CE hybrid is supported')
    seed_all();active,histories,x=active_context(ctx)
    users,past,future,starts,lengths=[torch.as_tensor(a,device='cuda') for a in examples(histories,c['context'],c['horizon'])]
    counts=(future>=0).sum(1);infer=history_tail(histories,c['context'])
    model=HybridMarkov(len(histories),len(active),c['dim'],c['personal'],c['dropout']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['reg'])
    for epoch in range(epochs):
        model.train();drawusers=torch.randint(len(histories),(len(histories)*16,),device='cuda')
        chosen=starts[drawusers]+(torch.rand(len(drawusers),device='cuda')*lengths[drawusers]).long()
        for start in range(0,len(chosen),512):
            ids=chosen[start:start+512];truth=future[ids];logits=model.all_scores(users[ids],past[ids])
            per_target=-F.log_softmax(logits,dim=1).gather(1,truth.clamp_min(0))
            step(((per_target*(truth>=0)).sum(1)/counts[ids]).mean(),model,optimizer)
    model.eval()
    with torch.no_grad():prediction=model.all_scores(torch.arange(len(histories),device='cuda'),infer).cpu().numpy()
    return model,expand_active(ctx,active,prediction)

def fit_drift(ctx,c,epochs):
    from .drift_factorization import DriftFactor,temporal_pairs
    from .bpr_refine import draw_negatives
    seed_all();active,histories,x=active_context(ctx)
    users,positive,times=[torch.as_tensor(a,device='cuda') for a in temporal_pairs(histories)]
    model=DriftFactor(len(histories),len(active),c['dim']).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'])
    for epoch in range(epochs):
        order=torch.randperm(len(users),device='cuda')
        for start in range(0,len(order),8192):
            ids=order[start:start+8192];u=users[ids];pos=positive[ids];neg=draw_negatives(x,u)
            vector=model.user_at(u,times[ids]);ip=model.item[pos];ineg=model.item[neg]
            difference=(vector*(ip-ineg)).sum(1)+model.bias[pos]-model.bias[neg]
            static=(model.user[u].square().sum()+ip.square().sum()+ineg.square().sum()+model.bias[pos].square().sum()+model.bias[neg].square().sum())/(2*len(ids))
            drift=model.drift[u].square().sum()/(2*len(ids))
            step(F.softplus(-difference).mean()+c['base_reg']*static+c['drift_reg']*drift,model,optimizer,clip=False)
    with torch.no_grad():prediction=model.scores(c['predict_t']).cpu().numpy()
    return model,expand_active(ctx,active,prediction)

def fit_similarity(ctx,c,epochs):
    from .attentive_similarity import AttentiveSimilarity,full_prediction
    from .hybrid_markov import examples
    from .bpr_refine import draw_negatives
    if c['mode']!='causal' or c['objective']!='bce':raise ValueError('Only selected causal BCE similarity is supported')
    seed_all();active,histories,x=active_context(ctx);infer=history_tail(histories,c['context'])
    exusers,past,future,starts,counts=[torch.as_tensor(a,device='cuda') for a in examples(histories,c['context'],c['horizon'])]
    future_counts=(future>=0).sum(1)
    model=AttentiveSimilarity(len(active),c['dim'],c['attention'],c['beta']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    for epoch in range(1,epochs+1):
        model.train()
        for _ in range(40):
            users=torch.randint(len(histories),(256,),device='cuda')
            ids=starts[users]+(torch.rand(len(users),device='cuda')*counts[users]).long()
            column=(torch.rand(len(users),device='cuda')*future_counts[ids]).long();positive=future[ids,column]
            negatives=draw_negatives(x,users.repeat_interleave(4)).reshape(len(users),4)
            candidates=torch.cat([positive[:,None],negatives],1)
            prediction=model(past[ids],candidates,warmup=epoch<=c['warmup_epochs'])
            step((F.softplus(-prediction[:,0])+F.softplus(prediction[:,1:]).mean(1)).mean(),model,optimizer)
    return model,expand_active(ctx,active,full_prediction(model,infer,len(active)))

def fit_contrastive_graph(ctx,c,epochs):
    from .graph_contrastive import CrossLayerGraph,identity_contrast
    seed_all();active,_,x=active_context(ctx);pairs=torch.nonzero(x,as_tuple=False)
    model=CrossLayerGraph(x,c['dim'],c['layers'],c['epsilon'],c['contrast_layer']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'])
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(pairs),device='cuda')
        for start in range(0,len(order),c['batch_size']):
            selected=pairs[order[start:start+c['batch_size']]];users,positive=selected[:,0],selected[:,1];negative=sample_unseen(x,users)
            u,i,uc,ic=model.embeddings(True);unique_users=torch.unique(users);unique_items=torch.unique(positive)
            contrast=identity_contrast(u[unique_users],uc[unique_users],c['temperature'])+identity_contrast(i[unique_items],ic[unique_items],c['temperature'])
            uv,iv=u[users],i[positive];bpr=F.softplus(-(uv*(iv-i[negative])).sum(1)).mean()
            regularizer=(torch.linalg.vector_norm(uv)+torch.linalg.vector_norm(iv))/len(users)
            step(bpr+c['reg']*regularizer+c['contrast_weight']*contrast,model,optimizer)
    model.eval()
    with torch.no_grad():prediction=model().cpu().numpy()
    return model,expand_active(ctx,active,prediction)

def fit_diffusion(ctx,c,epochs):
    from .diffusion import DenoisingMLP,SmallNoiseDiffusion
    rng=seed_all();active,_,x=active_context(ctx)
    model=DenoisingMLP(len(active),c['hidden'],c['dropout'],c['normalize']).cuda()
    diffusion=SmallNoiseDiffusion(c['steps'],c['noise_scale'],c['noise_min'],c['noise_max']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(x))
        for start in range(0,len(order),c['batch']):
            ids=torch.as_tensor(order[start:start+c['batch']],device='cuda')
            step(diffusion.loss(model,x[ids],c['reweight']),model,optimizer)
    model.eval();prediction=torch.cat([diffusion.predict(model,x[j:j+128]) for j in range(0,len(x),128)]).cpu().numpy()
    return model,expand_active(ctx,active,prediction)

def train_selected(node,ctx,epochs_override=None):
    import json
    from scipy import sparse
    from .models import SLIM
    from .linear import transitions
    family=node['family'];row=node['row'];c=dict(row.get('parameters',{}));seed=2020
    if family=='predictions.npz':
        row=json.loads((Path(node['prediction']).parent/'result.json').read_text())
        c=dict(row['parameters']);family='sasrec_ce';seed=row['seed']
    epochs=epochs_override if epochs_override is not None else row.get('best_epoch',c.get('epochs'))
    if family=='slim':
        model=SLIM(**dict(c,workers=2)).fit(ctx.x);scores=model.predict(ctx.x)
    elif family=='markov':
        counts=transitions(ctx.histories,ctx.x.shape[1],c['horizon']);degrees=np.asarray(counts.sum(1)).ravel()
        weights=sparse.diags(1/np.maximum(degrees,1))@counts@sparse.diags(np.maximum(ctx.x.sum(0),1)**(-c['pop_power']))
        kind=c['query'];query=(ctx.x if kind=='all' else weighted_history(ctx,half_life=int(kind[4:])) if kind.startswith('half') else weighted_history(ctx,tail=int(kind[4:])))
        scores=(sparse.csr_matrix(query)@weights).toarray();model=None
    elif family in ('sasrec_ce','gru_ce'):
        model,scores=fit_sequence(ctx,c,epochs,'gru' if family=='gru_ce' else 'sas',seed)
    elif family.startswith('horizon_sas_'):
        from .stronger_sequence_features import sequence_views
        config=dict(dim=c['dim'],maxlen=100,horizon=200 if c['kind']=='proportional' else 20,lr=.001,dropout=.2,mode='past')
        model,_=fit_sequence(ctx,config,epochs,'proportional' if c['kind']=='proportional' else 'sas')
        scores=sequence_views(model,ctx)[c['view']]
    elif family.startswith('rating_sasrec_'):model,scores=fit_sequence(ctx,c,epochs,'rated')
    elif family.startswith('filter_mlp_'):model,scores=fit_prefix(ctx,c,epochs)
    elif family.startswith('multi_interest_'):model,scores=fit_prefix(ctx,c,epochs,multi=True)
    elif family=='temporal_ae':model,scores=fit_autoencoder(ctx,c,epochs)
    elif family=='bpr_mf':model,scores=fit_bpr(ctx,c,epochs)
    elif family=='bert4rec_adapted':model,scores=fit_masked(ctx,c,epochs)
    elif family=='hybrid_markov_ce':model,scores=fit_hybrid(ctx,c,epochs)
    elif family=='drift_mf_extrapolated':model,scores=fit_drift(ctx,c,epochs)
    elif family=='mean_similarity_causal_bce':model,scores=fit_similarity(ctx,c,epochs)
    elif family=='xsim_graph':model,scores=fit_contrastive_graph(ctx,c,epochs)
    elif family=='diffusion_x0':model,scores=fit_diffusion(ctx,c,epochs)
    else:raise ValueError(f'Unsupported selected base family: {family}')
    if scores.shape!=ctx.x.shape or not np.isfinite(scores).all():raise ValueError('Invalid refitted base prediction')
    return model,scores,dict(family=family,parameters=c,fit_epochs=epochs,seed=seed)

def main():
    import argparse
    import hashlib
    import json
    import os
    from time import perf_counter
    from threadpoolctl import threadpool_limits
    from .data import load_data,sha256
    from .evaluation import topk
    from .submission_rankers import is_ranker
    from .io_utils import dump
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--data',required=True);parser.add_argument('--out',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Use a Slurm GPU allocation')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    plan=json.loads(Path(args.plan).read_text());data=load_data(args.data);ctx=full_context(data)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);records={}
    with threadpool_limits(limits=2):
        for key in plan['leaves']:
            node=plan['nodes'][key]
            if is_ranker(node):continue
            print('FIT '+json.dumps(dict(run=node['run'],family=node['family'],smoke=args.smoke)),flush=True)
            started=perf_counter();model,scores,spec=train_selected(node,ctx,2 if args.smoke else None)
            ranking=topk(scores,ctx.seen,ctx.active)
            if (ctx.seen[np.arange(len(ranking))[:,None],ranking]).any():raise AssertionError('Seen item in recommendations')
            details=dict(spec,fit_seconds=perf_counter()-started,fit_interactions=int(ctx.x.sum()),active_items=int(ctx.active.sum()),finite_scores=True)
            if not args.smoke:
                label=hashlib.sha256(key.encode()).hexdigest()[:20];path=out/(label+'.npy');np.save(path,scores)
                details.update(path=str(path.resolve()),sha256=sha256(path))
                if isinstance(model,torch.nn.Module):
                    checkpoint=out/(label+'.pt');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},spec=spec,active_items=np.flatnonzero(ctx.active).tolist(),fit_interactions=len(data.users),data_hashes=data.hashes),checkpoint)
                    details['checkpoint']=str(checkpoint.resolve())
            records[key]=details;dump(out/'bases.json',records);print('DONE '+json.dumps(details),flush=True)
            del model,scores,ranking;torch.cuda.empty_cache()
    dump(out/'complete.json',dict(selected_base_count=len(records),smoke=args.smoke,fit_interactions=len(data.users),data_hashes=data.hashes))

if __name__=='__main__':main()
