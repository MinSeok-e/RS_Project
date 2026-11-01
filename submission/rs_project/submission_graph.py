"""Replay the selected ensemble graph without further validation selection.

Graph edges retain every selected component, normalization, and float32 mixing
step. Validation replay is required before using newly fitted leaf predictions.
"""

import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

from scipy import sparse

from sklearn.preprocessing import normalize

from .data import load_data

from .evaluation import topk, ranking_metrics

from .ensemble import transform

from .fusion import calibrated

from .score_diffusion import cosine_graph, propagate

from .linear import transitions

from .models import retain_row_topk

from .io_utils import dump

def recover_steps(node):
    rows=node['steps']
    if not rows or len(rows[0]['parameters']['weights'])!=1:
        raise ValueError('Missing initial single-model ensemble state')
    initial=rows[0]['parameters']['weights']
    if list(initial.values()) != [1.]:
        raise ValueError('Unexpected initial ensemble weight')
    operations=[];old=initial
    for row in rows[1:]:
        new=row['parameters']['weights'];matches=[]
        for alpha in (.025,.05,.1,.25,.5,.75,.9):
            for candidate in new:
                wanted={k:v*(1-alpha) for k,v in old.items()}
                wanted[candidate]=wanted.get(candidate,0)+alpha
                if set(new)==set(wanted) and all(abs(new[k]-wanted[k])<1e-12 for k in new):
                    matches.append((candidate,alpha))
        if len(matches)!=1:
            raise ValueError(f'Ambiguous mixing step: {node["run"]} {row["trial"]}: {matches}')
        operations.append(matches[0]);old=new
    if old!=node['row']['parameters']['weights']:
        raise ValueError('Last ensemble state differs from selected row')
    return next(iter(initial)),operations

def load_scores(path):
    path=Path(path)
    if path.suffix=='.npy':return np.load(path,mmap_mode='r')
    with np.load(path) as f:return f['scores'].copy()

class GraphReplay:
    def __init__(self,plan,ctx,out,leaf_paths=None,validate=True):
        self.plan=plan;self.ctx=ctx;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.eligible=~ctx.seen&ctx.active[None,:]
        self.leaf_paths=leaf_paths;self.validate=validate;self.paths={};self.checks=[];self.graphs={}

    def score(self,key):
        return load_scores(self.run(key))

    def scaled(self,node,name):
        child=node['children'][name];family=node['family'];p=node['row']['parameters']
        if 'bag' in child:
            return np.mean([transform(self.score(k),self.eligible,p['method']) for k in child['bag']],axis=0)
        score=self.score(child['prediction'])
        if family.startswith('ensemble_'):return transform(score,self.eligible,p['method'])
        if family.startswith('calibrated_'):return calibrated(score,self.eligible,p['kind'],p['parameter'],p['power'])
        if family=='mixed_calibration':
            _,kind,parameter,power=name.split('|')
            value=calibrated(score,self.eligible,kind,float(parameter),float(power))
            return value/np.maximum(value.sum(1,keepdims=True),1e-20)
        raise ValueError(f'Unsupported normalization: {family}')

    def diffusion(self,node):
        p=node['row']['parameters'];child=node['children'][p['source']]['prediction']
        probability=calibrated(self.score(child),self.eligible,'probability',p['temperature'])
        graph_key=(p['graph'],p['pop_power'],p['neighbors'])
        if graph_key not in self.graphs:
            if p['graph']=='cosine':g=cosine_graph(self.ctx.x,p['neighbors'])
            else:g=retain_row_topk(transitions(self.ctx.histories,self.ctx.x.shape[1],int(p['graph'].removeprefix('markov'))),p['neighbors'])
            popularity=self.ctx.x.sum(0).clip(1)
            self.graphs[graph_key]=normalize(g@sparse.diags(popularity**(-p['pop_power'])),norm='l1',axis=1).tocsr()
        predicted=probability.copy()
        for _ in range(p['steps']):predicted=propagate(predicted,self.graphs[graph_key],self.eligible)
        return (1-p['alpha'])*probability+p['alpha']*predicted

    def run(self,key):
        if key in self.paths:return self.paths[key]
        node=self.plan['nodes'][key]
        if 'children' not in node:
            if self.leaf_paths is not None and key not in self.leaf_paths:
                raise ValueError(f'Missing refitted leaf: {key}')
            path=key if self.leaf_paths is None else self.leaf_paths[key]
            score=load_scores(path)
            if score.shape!=self.ctx.x.shape or not np.isfinite(score).all():
                raise ValueError('Invalid leaf shape or values')
            self.paths[key]=path
            return path
        if node['family'].startswith('score_diffusion_'):
            score=self.diffusion(node)
        else:
            initial,operations=recover_steps(node)
            normalized={name:self.scaled(node,name) for name in node['children']}
            score=normalized[initial].copy()
            for name,alpha in operations:score=(1-alpha)*score+alpha*normalized[name]
            del normalized
        check={'run':node['run'],'family':node['family'],'original_prediction':key}
        if self.validate:
            original=load_scores(key)
            check['max_absolute_error']=float(np.max(np.abs(score-original)))
            np.testing.assert_allclose(score,original,rtol=2e-6,atol=2e-7,err_msg=f'{node["run"]}/{node["family"]}')
            np.testing.assert_array_equal(topk(score,self.ctx.seen,self.ctx.active),topk(original,self.ctx.seen,self.ctx.active))
            del original
        path=self.out/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.npy')
        np.save(path,score);self.paths[key]=str(path);self.checks.append(check)
        dump(self.out/'checks.json',self.checks)
        print('REPLAY '+json.dumps(check),flush=True)
        return str(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    from types import SimpleNamespace
    from threadpoolctl import threadpool_limits
    data=load_data(args.data);x=data.matrix(data.train_rows,dtype=np.float32)
    histories=[data.items[data.train_rows[data.users[data.train_rows]==u]] for u in range(data.shape[0])]
    ctx=SimpleNamespace(x=x,seen=x.astype(bool),active=x.any(0),histories=histories)
    plan=json.loads(Path(args.plan).read_text());replay=GraphReplay(plan,ctx,args.out)
    with threadpool_limits(limits=2):
        path=replay.run(plan['root']);scores=load_scores(path)
        metrics,_=ranking_metrics(topk(scores,ctx.seen,ctx.active),data.targets())
    np.testing.assert_allclose(metrics['ndcg@20'],plan['selected_best']['metrics']['ndcg@20'],rtol=0,atol=1e-12)
    dump(Path(args.out)/'complete.json',dict(validation_graph_reproduced=True,metrics=metrics,root_scores=path,
         nodes=len(plan['nodes']),leaves=len(plan['leaves']),original_split_sha256=data.summary()['split_sha256']))
    print('VERIFIED '+json.dumps(metrics),flush=True)

if __name__=='__main__':main()
