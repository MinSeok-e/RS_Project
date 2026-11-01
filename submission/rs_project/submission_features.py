"""Full-history features for the frozen validation-selected ranking models.

The serving schema is verified against historical validation caches. Auxiliary
historical quality uses events strictly inside the current observed history.
"""

import argparse

import gc

import json

import os

from pathlib import Path

import numpy as np

from scipy import sparse

import torch

from threadpoolctl import threadpool_limits

from .data import load_data, sha256

from .io_utils import dump

from .submission_bases import full_context, fit_prefix

from .submission_graph import load_scores

from .submission_rankers import is_ranker

from .rerank import context, inner_context, base_scores, make_features

from .common import weighted_history

from .linear import ease_predict, user_knn, transitions

from .ensemble import transform

from .history_features import history_model_quality

from .filter_features import views as filter_views

from .filter_sequence import FilterRec

from .filter_bag_features import SOURCES, FEATURE_NAMES, candidate_features, aggregate_candidate_features

from .stronger_sequence_features import fixed_sequence, model_for, sequence_views

from .diverse_features import fixed_als

from .higher_order import choose_pairs, constrained_predict

RAW_NAMES=['ease','recent_ease','rating_ease','slim','userknn','markov5','markov20','markov50','sasbag']

def saved_arrays(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}

def historical_quality(ctx,project,out,verify=False):
    """19 per-user features. Never use the unknown future after ctx's fit cut."""
    all_quality={m:[] for m in RAW_NAMES};all_valid={m:[] for m in RAW_NAMES}
    source=project/'results/search_81255';meta=json.loads((source/'features.json').read_text())
    for entry in meta['checks']:
        if entry['fold']=='outer':continue
        fraction=entry.get('fraction',int(entry['fold'].removeprefix('inner'))/100)
        past=inner_context(ctx.data,fraction);past=context(past.data,past.data.train_rows,[])
        if verify:arrays=saved_arrays(source/(entry['fold']+'.npz'));names=meta['feature_names']
        else:
            print('HISTORICAL '+str(fraction),flush=True)
            scores=base_scores(past);arrays,names=make_features(past,scores,False);del scores
        lengths=np.array([len(h) for h in past.histories]);rows=[]
        for model in RAW_NAMES:
            col=names.index(model);ranking=[]
            for u in range(len(ctx.histories)):
                ids=np.flatnonzero(arrays['users']==u)
                order=np.lexsort((arrays['items'][ids],-arrays['features'][ids,col]))[:20]
                ranking.append(arrays['items'][ids[order]])
            ranking=np.asarray(ranking);quality,valid=history_model_quality(ctx.histories,lengths,ranking)
            all_quality[model].append(quality);all_valid[model].append(valid)
            rows.append(ranking)
        if not verify:
            np.savez_compressed(out/(entry['fold']+'_history.npz'),rankings=np.stack(rows),lengths=lengths,
                                past_rows=past.data.train_rows)
        del arrays,past;torch.cuda.empty_cache();gc.collect()
    values=[];names=[]
    for model in RAW_NAMES:
        mean=np.zeros(len(ctx.histories));recent=mean.copy();number=mean.copy()
        for quality,valid in zip(all_quality[model],all_valid[model]):
            mean+=quality;number+=valid;recent[valid]=quality[valid]
        mean/=np.maximum(number,1);values.extend([mean,recent])
        names.extend([model+'_past_pool_ndcg_mean',model+'_past_pool_ndcg_recent'])
    values.append(number);names.append('past_quality_queries')
    return np.stack(values,axis=1).astype(np.float32),names

def auxiliary(ctx,project,out,verify=False):
    records={}
    def save(label,values,details):
        path=out/(label+'.npz');np.savez_compressed(path,**values)
        records[label]=dict(path=str(path.resolve()),sha256=sha256(path),**details)
        dump(out/'auxiliary.json',records);print('AUXILIARY '+label,flush=True)
    quality,names=historical_quality(ctx,project,out,verify)
    save('history',{'features':quality},{'feature_names':names});del quality
    for kind,job,filename in [('wide',81174,'trial_5.pt'),('past',81201,'sasrec_exclude_past_best.pt')]:
        if verify:
            model=model_for(ctx,kind);saved=torch.load(project/f'results/search_{job}'/filename,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);del saved
        else:model=fixed_sequence(ctx,kind)
        save('strong_'+kind,sequence_views(model,ctx),{'fit_epochs':80 if kind=='wide' else 50})
        if not verify:torch.save(model.state_dict(),out/('strong_'+kind+'.pt'))
        del model;torch.cuda.empty_cache()
    for job,family,seed in SOURCES:
        path=project/f'results/search_{job}'/(family+'_best.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
        c=saved['config'];epochs=saved['best_epoch']
        if verify:
            model=FilterRec(ctx.x.shape[1],c['length'],c['dim'],c['dropout'],c['layers']).cuda();model.load_state_dict(saved['state_dict'])
        else:model,_=fit_prefix(ctx,c,epochs)
        save('filter_bag_'+str(seed),filter_views(model,ctx),{'fit_epochs':epochs,'parameters':c,'source_checkpoint_sha256':sha256(path)})
        if not verify:torch.save(model.state_dict(),out/('filter_bag_'+str(seed)+'.pt'))
        del model,saved;torch.cuda.empty_cache()
    if verify:
        als=load_scores(project/'results/search_81192/implicit_als_best_predictions.npz')
        higher=load_scores(project/'results/search_81239/higher_order_ease_best_predictions.npz')
    else:
        als=fixed_als(ctx);pairs=choose_pairs(ctx.x,5000)
        higher=constrained_predict(ctx.x,[weighted_history(ctx,half_life=20)],pairs,1.,1000)[0]
    save('diverse',{'implicit_als':als,'higher_order':higher},{'als_epochs':30,'higher_order_pairs':5000})
    dump(out/'complete.json',dict(auxiliary_complete=True,verification=verify,fit_interactions=len(ctx.data.train_rows),data_hashes=ctx.data.hashes))

class FeatureBuilder:
    def __init__(self,ctx,plan,project,out,bases=None,aux=None,verify=False):
        self.ctx=ctx;self.plan=plan;self.project=Path(project);self.out=Path(out);self.bases=bases;self.aux=Path(aux);self.verify=verify
        self.eligible=~ctx.seen&ctx.active[None,:];self.paths={};self.records={}

    def leaf(self,run,family):
        keys=[k for k in self.plan['leaves'] if self.plan['nodes'][k]['run']==run and self.plan['nodes'][k]['family']==family]
        if len(keys)!=1:raise ValueError(f'Ambiguous selected leaf {run}/{family}: {keys}')
        return keys[0]

    def score(self,run,family):
        key=self.leaf(run,family)
        return load_scores(key if self.verify else self.bases[key]['path'])

    def base(self):
        if self.verify:scores=base_scores(self.ctx,self.project)
        else:
            ctx=self.ctx;x=ctx.x;q=weighted_history(ctx,half_life=20);r=weighted_history(ctx,rating_power=2)
            scores={'ease':ease_predict(x,x,300).astype(np.float32),'recent_ease':ease_predict(x,q,300).astype(np.float32),
                    'rating_ease':ease_predict(r,r,100).astype(np.float32),'slim':self.score('cpu_81056','slim').astype(np.float32),
                    'userknn':user_knn(x,q,x,20,20).astype(np.float32)}
            for horizon in [5,20,50]:
                counts=transitions(ctx.histories,x.shape[1],horizon);degrees=np.asarray(counts.sum(1)).ravel()
                weights=sparse.diags(1/np.maximum(degrees,1))@counts
                scores[f'markov{horizon}']=(sparse.csr_matrix(weighted_history(ctx,tail=10))@weights).toarray()
            bag=[]
            for key in self.plan['leaves']:
                if self.plan['nodes'][key]['family']=='predictions.npz':bag.append(transform(load_scores(self.bases[key]['path']),self.eligible,'zscore'))
            if len(bag)!=3:raise AssertionError('Expected all three SAS seeds')
            scores['sasbag']=np.mean(bag,axis=0)
        arrays,names=make_features(self.ctx,scores,False);del scores
        return arrays,names

    def read(self,job):
        path=self.build(job);arrays=saved_arrays(path);names=list(self.records[str(job)]['feature_names'])
        return arrays,names

    def append_views(self,arrays,names,views,prefix):
        columns=[arrays['features']]
        for label,score in views.items():
            for scale in ['raw','zscore','rrf']:
                value=score if scale=='raw' else transform(score,self.eligible,scale)
                columns.append(value[arrays['users'],arrays['items']][:,None]);names.append(prefix+label+'_'+scale)
        arrays['features']=np.concatenate(columns,axis=1)

    def build(self,job):
        job=int(job)
        if job in self.paths:return self.paths[job]
        print('FEATURE SCHEMA '+str(job),flush=True)
        if job in (81169,81193):arrays,names=self.base() if job==81169 else self.read(81169)
        elif job in (81293,81527,81632):arrays,names=self.read(81234)
        else:
            parent={81175:81169,81234:81193,81246:81234,81255:81246,81264:81255,81326:81234,81334:81234,81432:81234,81612:81234,81622:81234,81746:81622}[job]
            arrays,names=self.read(parent)
            functions=[]
            if job in (81175,81234):
                from .rerank_augment import augment
                functions.append(augment)
            if job==81234:
                from .rerank_rich import recent_similarity,directional_features
                functions.extend([recent_similarity,directional_features])
            if job==81246:
                from .order_features import order_features
                functions.append(lambda c,u,i:order_features(c,u,i)[:2])
            if job==81326:
                from .neighbor_order_features import neighbor_features
                functions.append(neighbor_features)
            for function in functions:
                values,columns=function(self.ctx,arrays['users'],arrays['items'])
                arrays['features']=np.concatenate((arrays['features'],values),axis=1);names+=columns
            if job==81255:
                diverse=saved_arrays(self.aux/'diverse.npz')
                values={'bpr_mf':self.score('search_81192','bpr_mf'),'implicit_als':diverse['implicit_als'],
                        'gru_ce':self.score('search_81144','gru_ce'),'higher_order':diverse['higher_order']}
                # Historical naming puts 'stack' after the model's label.
                self.append_views(arrays,names,{k+'_stack':v for k,v in values.items()},'')
            if job==81264:
                h=saved_arrays(self.aux/'history.npz')['features'];meta=json.loads((self.aux/'auxiliary.json').read_text())
                arrays['features']=np.concatenate((arrays['features'],h[arrays['users']]),axis=1);names+=meta['history']['feature_names']
            if job==81334:
                for kind in ['wide','past']:self.append_views(arrays,names,saved_arrays(self.aux/('strong_'+kind+'.npz')),'strong_sas_'+kind+'_')
            if job==81432:
                from .fold_normalization import normalize_fold
                arrays['features']=normalize_fold(arrays['features'],names,'zscore')
            if job in (81612,81622):
                key=self.leaf('search_81594' if job==81612 else 'search_81605','filter_mlp_events_past')
                path=Path(key).parent/'filter_mlp_events_past_best.pt' if self.verify else Path(self.bases[key]['checkpoint'])
                saved=torch.load(path,map_location='cpu',weights_only=False)
                c=saved['config'] if self.verify else saved['spec']['parameters']
                model=FilterRec(self.ctx.x.shape[1],c['length'],c['dim'],c['dropout'],c['layers']).cuda();model.load_state_dict(saved['state_dict'])
                self.append_views(arrays,names,filter_views(model,self.ctx),'filter_');del model,saved;torch.cuda.empty_cache()
            if job==81746:
                samples=[]
                for _,_,seed in SOURCES:
                    samples.append(candidate_features(saved_arrays(self.aux/('filter_bag_'+str(seed)+'.npz')),self.eligible,arrays['users'],arrays['items']))
                arrays['features']=np.concatenate((arrays['features'],aggregate_candidate_features(samples)),axis=1);names+=FEATURE_NAMES
        source=self.project/f'results/search_{job}';meta=json.loads((source/'features.json').read_text())
        if meta['smoke'] or meta['data_hashes']!=self.ctx.data.hashes:raise ValueError('Feature schema metadata mismatch')
        if names!=meta['feature_names']:raise ValueError(f'Feature schema differs: {job}')
        if not np.isfinite(arrays['features']).all() or 'labels' in arrays:raise ValueError('Bad serving features')
        if self.ctx.seen[arrays['users'],arrays['items']].any():raise ValueError('Seen item in serving candidate pool')
        error=None
        if self.verify:
            original=saved_arrays(source/'outer.npz')
            for k in ['users','items','groups']:np.testing.assert_array_equal(arrays[k],original[k],err_msg=str(job))
            np.testing.assert_allclose(arrays['features'],original['features'],rtol=2e-4,atol=2e-5,err_msg=str(job))
            error=float(np.abs(arrays['features']-original['features']).max());del original
        path=self.out/(str(job)+'.npz');np.savez_compressed(path,**arrays);self.paths[job]=path
        self.records[str(job)]=dict(path=str(path.resolve()),sha256=sha256(path),feature_names=names,shape=list(arrays['features'].shape),validation_max_error=error)
        dump(self.out/'features.json',self.records);print('FEATURE DONE '+json.dumps(dict(schema=job,shape=list(arrays['features'].shape),validation_max_error=error)),flush=True)
        return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['auxiliary','features'],required=True);p.add_argument('--plan',required=True);p.add_argument('--data',required=True)
    p.add_argument('--project',required=True);p.add_argument('--out',required=True);p.add_argument('--bases');p.add_argument('--aux');p.add_argument('--verify',action='store_true');args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Use a Slurm GPU allocation')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    data=load_data(args.data);ctx=context(data,data.train_rows,[]) if args.verify else full_context(data)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);project=Path(args.project)
    with threadpool_limits(limits=2):
        if args.stage=='auxiliary':auxiliary(ctx,project,out,args.verify)
        else:
            plan=json.loads(Path(args.plan).read_text());bases=None
            if not args.verify:
                complete=json.loads((Path(args.bases)/'complete.json').read_text())
                if complete['smoke'] or complete['data_hashes']!=data.hashes or complete['fit_interactions']!=len(data.users):raise ValueError('Incomplete full base refit')
                bases=json.loads((Path(args.bases)/'bases.json').read_text())
            aux_complete=json.loads((Path(args.aux)/'complete.json').read_text())
            if aux_complete['verification']!=args.verify or aux_complete['data_hashes']!=data.hashes:raise ValueError('Wrong auxiliary cache')
            builder=FeatureBuilder(ctx,plan,project,out,bases,args.aux,args.verify)
            jobs=sorted({int(Path(plan['nodes'][k]['row']['parameters']['features_directory']).name.removeprefix('search_')) for k in plan['leaves'] if is_ranker(plan['nodes'][k])})
            for job in jobs:builder.build(job);gc.collect()
            dump(out/'complete.json',dict(features_complete=True,verification=args.verify,fit_interactions=len(ctx.data.train_rows),data_hashes=data.hashes,schemas=jobs))

if __name__=='__main__':main()
