"""Load selected ranking checkpoints and verify their cached validation outputs."""

import argparse

from functools import lru_cache

import json

from pathlib import Path

import sys

import numpy as np

import torch

from .data import load_data, sha256

from .evaluation import topk

from .feature_selection import select_columns

from .neural_ranker import QueryRanker, pack

from .mixture_ranker import MixtureQueryRanker

from .cross_ranker import CrossQueryRanker

from .user_deepset import UserQueryRanker, identity_tensors

from .io_utils import dump

def is_ranker(node):
    return node['family'].startswith(('neural_ranker_','rerank_','catboost_','deepset_user_','mixture_finetune_'))

def ranker_checkpoint(node):
    root=Path(node['prediction']).parent;family=node['family']
    if family.startswith('rerank_'):return root/(family.removeprefix('rerank_')+'_best.txt')
    if family.startswith('catboost_'):return root/(family+'_best.cbm')
    return root/(family+'_best.pt')

def predict_ranker(node,arrays,feature_names,shape):
    family=node['family'];c=node['row']['parameters'];path=ranker_checkpoint(node)
    score=np.full(shape,-1e6,np.float32)
    if family.startswith('rerank_'):
        sys.path.insert(0,'/home/minseok/.local/rs-search-lightgbm-4.6.0')
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(path));pred=model.predict(arrays['features'],num_iteration=c['trees_used'],num_threads=2)
    elif family.startswith('catboost_'):
        sys.path.insert(0,'/home/minseok/.local/rs-search-catboost-1.2.10')
        from catboost import CatBoostRanker
        model=CatBoostRanker();model.load_model(str(path));pred=model.predict(arrays['features'],thread_count=2)
    else:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        columns=select_columns(feature_names,c.get('feature_groups','all'))
        features=arrays['features'] if columns is None else arrays['features'][:,columns]
        mean,std=saved['mean'].numpy(),saved['std'].numpy()
        if features.shape[1]!=len(mean):raise ValueError('Checkpoint feature dimension mismatch')
        x,_,mask=pack(features,None,arrays['groups'],int(arrays['groups'].max()),mean,std)
        if family.startswith('deepset_user_'):
            model=UserQueryRanker(features.shape[1],shape[0],c['dim'],c['embedding_dropout']).cuda()
            users,items=identity_tensors(arrays['users'],arrays['items'],arrays['groups'],int(arrays['groups'].max()))
        else:
            kind=c['kind'];hidden=c['hidden']
            cls=MixtureQueryRanker if kind.startswith(('mixture','mean')) else CrossQueryRanker if kind.startswith('cross') else QueryRanker
            if c.get('expert_feature_sharing')=='none':
                from .independent_mixture import IndependentFeatureMixture
                cls=IndependentFeatureMixture
            model=cls(features.shape[1],hidden,kind).cuda()
        model.load_state_dict(saved['state_dict']);model.eval()
        with torch.no_grad():
            if family.startswith('deepset_user_'):
                pred=torch.cat([model(x[j:j+32],mask[j:j+32],users[j:j+32],items[j:j+32]) for j in range(0,len(x),32)])
            else:pred=torch.cat([model(x[j:j+32],mask[j:j+32]) for j in range(0,len(x),32)])
            pred=pred[mask].cpu().numpy()
        del model,saved,x,mask;torch.cuda.empty_cache()
    if not np.isfinite(pred).all():raise ValueError('Nonfinite ranker inference')
    score[arrays['users'],arrays['items']]=pred
    return score

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Allocate GPU')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_limits
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(args.plan).read_text());data=load_data(args.data);seen=data.matrix(data.train_rows,dtype=bool);active=seen.any(0)
    @lru_cache(maxsize=2)
    def feature_source(root):
        root=Path(root);meta=json.loads((root/'features.json').read_text())
        if meta['smoke'] or meta['data_hashes']!=data.hashes:raise ValueError('Invalid feature source')
        with np.load(root/'outer.npz') as f:
            if 'labels' in f:raise ValueError('Outer labels in inference inputs')
            arrays={k:f[k].copy() for k in f.files}
        return arrays,meta['feature_names']
    checks=[]
    with threadpool_limits(limits=2):
        for key in plan['leaves']:
            node=plan['nodes'][key]
            if not is_ranker(node):continue
            arrays,names=feature_source(node['row']['parameters']['features_directory'])
            scores=predict_ranker(node,arrays,names,data.shape)
            with np.load(key) as f:
                original=f['scores']
                np.testing.assert_allclose(scores,original,rtol=2e-5,atol=2e-5,err_msg=node['run']+'/'+node['family'])
                np.testing.assert_array_equal(topk(scores,seen,active),f['rankings'])
            check=dict(run=node['run'],family=node['family'],checkpoint=str(ranker_checkpoint(node)),
                       checkpoint_sha256=sha256(ranker_checkpoint(node)),max_absolute_error=float(np.max(np.abs(scores-original))),rankings_reproduced=True)
            checks.append(check);dump(out/'checks.json',checks);print('VERIFIED '+json.dumps(check),flush=True)
            del scores,original
    dump(out/'complete.json',dict(selected_rankers_reproduced=True,count=len(checks),data_hashes=data.hashes))

if __name__=='__main__':main()
