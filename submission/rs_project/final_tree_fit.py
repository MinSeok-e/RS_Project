"""Refit the fixed selected CatBoost ranker on supplied historical training blocks."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from rs_project.data import load_data,sha256
from rs_project.ensemble import transform
from rs_project.evaluation import topk
from rs_project.io_utils import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not os.getenv('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Use Slurm GPU')
    sys.path.insert(0,'/home/minseok/.local/rs-search-catboost-1.2.10');import catboost as cb
    project=Path(args.project);c=json.loads(Path(args.config).read_text());data=load_data(project.parent/'data');source=Path(c['features_directory']);meta=json.loads((source/'features.json').read_text())
    if meta['smoke'] or meta['data_hashes']!=data.hashes:raise ValueError('Feature data mismatch')
    fields={k:[] for k in ['features','labels','groups']};checks=[]
    truth=np.zeros(data.shape,np.float32);truth[data.users[data.valid_rows],data.items[data.valid_rows]]=1
    for entry in meta['checks']:
        name=entry['fold'];path=source/(name+'.npz')
        with np.load(path) as f:
            fields['features'].append(f['features']);fields['groups'].append(f['groups'])
            if name=='outer':
                if 'labels' in f:raise ValueError('Original holdout cache was modified')
                fields['labels'].append(truth[f['users'],f['items']])
            else:fields['labels'].append(f['labels'])
        checks.append(dict(fold=name,sha256=sha256(path)))
    x=np.asfortranarray(np.concatenate(fields['features']));y=np.concatenate(fields['labels']);groups=np.concatenate(fields['groups']);del fields,truth
    qid=np.repeat(np.arange(len(groups)),groups);pool=cb.Pool(x,label=y,group_id=qid,feature_names=meta['feature_names'])
    path=Path(c['inference_features'])
    if sha256(path)!=c['inference_features_sha256']:raise ValueError('Full features changed')
    with np.load(path) as f:features=f['features'];users=f['users'];items=f['items']
    seen=data.matrix(np.arange(len(data.users)),dtype=bool);active=seen.any(0);eligible=~seen&active[None,:]
    if seen[users,items].any():raise ValueError('Seen item in final pool')
    out=Path(args.out);out.mkdir(exist_ok=False,parents=True);records=[];predictions=[]
    for seed in c['seeds']:
        params=dict(c['parameters'],random_seed=seed);model=cb.CatBoostRanker(**params);print('FIT',seed,flush=True);model.fit(pool)
        score=np.full(data.shape,-1e6,np.float32);score[users,items]=model.predict(features,thread_count=2)
        path=out/f'seed{seed}.npy';np.save(path,score);checkpoint=out/f'seed{seed}.cbm';model.save_model(str(checkpoint))
        records.append(dict(path=str(path.resolve()),sha256=sha256(path),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha256(checkpoint),seed=seed));predictions.append(transform(score,eligible,'rrf'));del model;print('DONE',seed,flush=True)
    path=out/'bag_rrf.npy';np.save(path,np.mean(predictions,axis=0));topk(np.load(path),seen,active)
    dump(out/'complete.json',dict(status='complete',config=c,data_hashes=data.hashes,fit_interactions=len(data.users),query_count=len(groups),records=records,path=str(path.resolve()),sha256=sha256(path),source_file=str(Path(__file__).resolve()),source_sha256=sha256(__file__),folds=checks,validation_ndcg20=None,training_policy='Final refit including previous train.csv holdout block; no holdout claim for this refit. No test labels. GPU training is not bitwise deterministic.'))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
