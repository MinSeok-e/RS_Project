"""Fit one fixed ranking configuration on all available historical training queries.

The last train.csv temporal block may be used in this FINAL refit. Consequently
its former validation score is not a holdout score for this fitted model.
No test relevance is available or used. There is no hyperparameter search.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rs_project.data import load_data,sha256
from rs_project.neural_ranker import pack,ranking_loss,QueryRanker
from rs_project.mixture_ranker import MixtureQueryRanker
from rs_project.ensemble import transform
from rs_project.evaluation import topk
from rs_project.io_utils import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Use a Slurm GPU')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    project=Path(args.project);config=json.loads(Path(args.config).read_text());data=load_data(project.parent/'data')
    source=Path(config['features_directory']);meta=json.loads((source/'features.json').read_text())
    if meta['smoke'] or meta['data_hashes']!=data.hashes:raise ValueError('Invalid historical features')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    pieces=[];labels=[];groups=[];weights=[];checks=[];target_counts=[]
    lengths=np.bincount(data.users[data.train_rows],minlength=data.shape[0]);targets=data.targets()
    for fold in meta['checks']:
        name=fold['fold'];path=source/(name+'.npz')
        if name=='outer' and not config['include_last_training_block']:continue
        with np.load(path) as f:arrays={k:f[k].copy() for k in f.files}
        if name=='outer':
            if 'labels' in arrays:raise ValueError('Original holdout cache was modified')
            truth=np.zeros(data.shape,np.float32)
            truth[data.users[data.valid_rows],data.items[data.valid_rows]]=1
            y=truth[arrays['users'],arrays['items']];counts=np.array([len(t) for t in targets]);del truth
            factor=float(config.get('last_block_weight',1.))
        else:
            if fold.get('outer_label_overlap',0)!=0:raise ValueError('Invalid historical fold')
            y=arrays['labels'];fraction=fold.get('fraction',int(name.removeprefix('inner'))/100)
            past=np.maximum(2,np.minimum(lengths-1,(lengths*fraction).astype(int)));counts=lengths-past
            if 'future_policy' in meta:counts=np.minimum(counts,np.ceil(past*.25).astype(int))
            factor=1.
        if len(counts)!=len(arrays['groups']):raise ValueError('Unexpected query identities')
        pieces.append(arrays['features']);labels.append(y);groups.append(arrays['groups']);target_counts.append(counts);weights.extend([factor]*len(counts))
        checks.append({'fold':name,'sha256':sha256(path),'queries':len(counts),'positive_candidates':int(y.sum()),'weight':factor})
    features=np.concatenate(pieces);y=np.concatenate(labels);g=np.concatenate(groups)
    del pieces,labels,groups,arrays
    mean=features.mean(0);std=features.std(0).clip(1e-5)
    train,truth,mask=pack(features,y,g,int(g.max()),mean,std);del features,y,g
    disc=np.cumsum(1/np.log2(np.arange(2,22)));idcg=torch.tensor(disc[np.minimum(np.concatenate(target_counts),20)-1],device='cuda',dtype=torch.float32)
    weights=torch.tensor(weights,device='cuda',dtype=torch.float32)
    inference=Path(config['inference_features'])
    if sha256(inference)!=config['inference_features_sha256']:raise ValueError('Serving features changed')
    with np.load(inference) as f:
        if 'labels' in f:raise ValueError('No test relevance may appear in inference features')
        full,_,full_mask=pack(f['features'],None,f['groups'],int(f['groups'].max()),mean,std);users=f['users'];items=f['items']
    seen=data.matrix(np.arange(len(data.users)),dtype=bool);active=seen.any(0);eligible=~seen&active[None,:]
    if seen[users,items].any():raise ValueError('Seen items in full-history inference pool')
    predictions=[];records=[]
    with threadpool_limits(limits=2):
        for seed in config['seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            cls=MixtureQueryRanker if config['kind'].startswith('mixture') else QueryRanker
            model=cls(train.shape[-1],config['hidden'],config['kind']).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);history=[]
            for epoch in range(1,config['epochs']+1):
                model.train();order=rng.permutation(len(train));losses=[]
                for start in range(0,len(order),32):
                    ix=torch.tensor(order[start:start+32],device='cuda')
                    loss=ranking_loss(model(train[ix],mask[ix]),truth[ix],mask[ix],idcg[ix],config['objective'],weights[ix])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(loss.item())
                history.append({'epoch':epoch,'loss':float(np.mean(losses))})
                if epoch%5==0 or epoch==1:print('TRAIN',seed,history[-1],flush=True)
            model.eval()
            with torch.no_grad():values=torch.cat([model(full[j:j+32],full_mask[j:j+32]) for j in range(0,len(full),32)])[full_mask].cpu().numpy()
            scores=np.full(data.shape,-1e6,np.float32);scores[users,items]=values
            if not np.isfinite(scores).all():raise ValueError('Nonfinite final scores')
            path=out/f'seed{seed}.npy';np.save(path,scores);checkpoint=out/f'seed{seed}.pt'
            torch.save({'state_dict':model.cpu().state_dict(),'config':config,'seed':seed,'mean':torch.tensor(mean),'std':torch.tensor(std)},checkpoint)
            records.append({'seed':seed,'path':str(path.resolve()),'sha256':sha256(path),'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':sha256(checkpoint)})
            predictions.append(transform(scores,eligible,'rrf'));dump(out/f'seed{seed}_history.json',history)
            del model,opt,scores;torch.cuda.empty_cache()
        path=out/'bag_rrf.npy';np.save(path,np.mean(predictions,axis=0));topk(np.load(path),seen,active)
    source_file=Path(__file__).resolve()
    result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),training_policy='Final refit only; previous outer train.csv block included as an additional historical query. Former validation metrics are not holdout scores for this model.',config=config,data_hashes=data.hashes,fit_interactions=len(data.users),query_count=len(train),folds=checks,source_file=str(source_file),source_sha256=sha256(source_file),records=records,path=str(path.resolve()),sha256=sha256(path),validation_ndcg20=None)
    dump(out/'complete.json',result);print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
