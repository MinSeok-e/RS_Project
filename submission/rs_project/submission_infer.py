"""Apply frozen rankers and the exact selected ensemble to full-history features."""

import argparse

import ast

from datetime import datetime, timezone

import gc

import hashlib

import json

import os

from pathlib import Path

import subprocess

import numpy as np

import torch

from threadpoolctl import threadpool_limits

from .data import load_data, sha256

from .evaluation import topk, write_submission

from .io_utils import dump

from .submission_bases import full_context

from .submission_features import saved_arrays

from .submission_graph import GraphReplay, load_scores

from .submission_rankers import is_ranker, predict_ranker, ranker_checkpoint

def source_closure(root,entries):
    """Include local import dependencies; never read credentials or artifact data."""
    found=set();pending=list(entries)
    while pending:
        rel=pending.pop()
        if rel in found:continue
        path=root/rel
        if not path.is_file():raise ValueError(f'Missing local source: {rel}')
        found.add(rel)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names if a.name.startswith('rs_project.')]
            elif isinstance(node,ast.ImportFrom):
                if node.level and rel.startswith('rs_project/') and node.module:modules=['rs_project.'+node.module]
                elif node.module and node.module.startswith('rs_project.'):modules=[node.module]
            for module in modules:
                child=module.replace('.','/')+'.py'
                if child not in found:pending.append(child)
    return sorted(found)

def verify_committed_sources(source_root,project,commit,files):
    if subprocess.check_output(['git','-C',str(project),'rev-parse',commit],text=True).strip()!=commit:raise ValueError('Unknown source commit')
    prefix = "submission/" if subprocess.check_output(["git","-C",str(project),"ls-tree","--name-only",commit,"submission"],text=True).strip()=="submission" else ""
    for rel in files:
        expected=subprocess.check_output(['git','-C',str(project),'show',f'{commit}:{prefix}{rel}'])
        if hashlib.sha256(expected).hexdigest()!=sha256(source_root/rel):raise ValueError('Executed source differs from commit: '+rel)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--data',required=True);p.add_argument('--project',required=True)
    p.add_argument('--bases',required=True);p.add_argument('--features',required=True);p.add_argument('--aux',required=True);p.add_argument('--out',required=True)
    p.add_argument('--base-source',required=True);p.add_argument('--aux-source',required=True);p.add_argument('--feature-source',required=True)
    p.add_argument('--graph-verification',required=True);p.add_argument('--ranker-verification',required=True);p.add_argument('--feature-verification',required=True)
    p.add_argument("--base-commit");p.add_argument("--aux-commit");p.add_argument("--feature-commit")
    args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Allocate Slurm GPU')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    project=Path(args.project);source=Path(__file__).resolve().parents[1];commit=os.environ['RS_SOURCE_COMMIT']
    source_files=source_closure(source,['rs_project/submission_bases.py','rs_project/submission_features.py','rs_project/submission_infer.py',
        'scripts/kaggle_project.py','configs/hybrid_ensemble_submission.json','slurm/submission_bases.sbatch','slurm/submission_features.sbatch','slurm/submission_infer.sbatch'])
    verify_committed_sources(source,project,commit,source_files)
    # Earlier full refits may predate this commit. Their executed bytes must match.
    stages=[]
    for label,root,entry in [('bases',args.base_source,'rs_project/submission_bases.py'),('auxiliary',args.aux_source,'rs_project/submission_features.py'),('features',args.feature_source,'rs_project/submission_features.py')]:
        root=Path(root);files=source_closure(root,[entry,'configs/hybrid_ensemble_submission.json'])
        stage_commit={"bases":args.base_commit,"auxiliary":args.aux_commit,"features":args.feature_commit}[label] or commit
        verify_committed_sources(root,project,stage_commit,files)
        stages.append(dict(stage=label,source_commit=stage_commit,source_root=str(root),source_sha256={f:sha256(root/f) for f in files}))
    data=load_data(args.data);ctx=full_context(data);plan=json.loads(Path(args.plan).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    validations={}
    for label,path in [('graph',args.graph_verification),('rankers',args.ranker_verification),('features',args.feature_verification)]:
        payload=json.loads(Path(path).read_text());validations[label]=dict(path=str(Path(path).resolve()),sha256=sha256(path),result=payload)
    if validations['graph']['result']['metrics']!=plan['selected_best']['metrics']:raise ValueError('Verified graph differs from selected best')
    if not validations['rankers']['result']['selected_rankers_reproduced'] or not validations['features']['result']['features_complete'] or not validations['features']['result']['verification']:raise ValueError('Missing inference verification')
    for label,folder in [('bases',args.bases),('auxiliary',args.aux),('features',args.features)]:
        meta=json.loads((Path(folder)/'complete.json').read_text())
        if meta.get('smoke') or meta.get('verification') or meta['data_hashes']!=data.hashes or meta['fit_interactions']!=len(data.users):raise ValueError('Wrong full-history artifact '+label)
    verified_rankers=json.loads((Path(args.ranker_verification).parent/"checks.json").read_text())
    verified_hashes={str(Path(r["checkpoint"]).resolve()):r["checkpoint_sha256"] for r in verified_rankers}
    bases=json.loads((Path(args.bases)/'bases.json').read_text());features=json.loads((Path(args.features)/'features.json').read_text())
    leaf_paths={};rankers=[]
    manifest=dict(run_id=out.name,status='running',smoke=False,utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ['SLURM_JOB_ID'],
        source_commit=commit,source_root=str(source),source_sha256={f:sha256(source/f) for f in source_files},stage_sources=stages,data_hashes=data.hashes,
        fit_interactions=len(data.users),users=data.shape[0],items=data.shape[1],test_users=len(data.test_ids),selected_validation=plan['selected_best'],
        refit_policy='Refit base recommenders at fixed selected epochs on all supplied train rows; keep selected ranker weights and ensemble weights frozen.',
        verification=validations,base_directory=str(Path(args.bases).resolve()),feature_directory=str(Path(args.features).resolve()),submissions={})
    dump(out/'manifest.json',manifest)
    with threadpool_limits(limits=2):
        for key in plan['leaves']:
            node=plan['nodes'][key]
            if not is_ranker(node):
                details=bases[key]
                if sha256(details['path'])!=details['sha256'] or details['fit_interactions']!=len(data.users):raise ValueError('Base prediction provenance mismatch')
                leaf_paths[key]=details['path'];continue
            checkpoint=ranker_checkpoint(node)
            if sha256(checkpoint)!=verified_hashes.get(str(checkpoint.resolve())):raise ValueError("Selected ranker changed after verification")
            job=Path(node['row']['parameters']['features_directory']).name.removeprefix('search_');feature=features[job]
            if sha256(feature['path'])!=feature['sha256']:raise ValueError('Serving features changed')
            arrays=saved_arrays(feature['path']);scores=predict_ranker(node,arrays,feature['feature_names'],data.shape)
            label=hashlib.sha256(key.encode()).hexdigest()[:20];path=out/(label+'.npy');np.save(path,scores);leaf_paths[key]=str(path.resolve())
            rankers.append(dict(source_prediction=key,checkpoint=str(ranker_checkpoint(node)),checkpoint_sha256=sha256(ranker_checkpoint(node)),path=str(path.resolve()),sha256=sha256(path)))
            dump(out/'rankers.json',rankers);print('RANKER '+node['run']+'/'+node['family'],flush=True)
            del arrays,scores;gc.collect();torch.cuda.empty_cache()
        if set(leaf_paths)!=set(plan['leaves']):raise AssertionError('Missing ensemble leaf')
        dump(out/'leaves.json',leaf_paths)
        root=GraphReplay(plan,ctx,out/'graph',leaf_paths,validate=False).run(plan['root'])
        ranks=topk(load_scores(root),ctx.seen,ctx.active);path=out/'hybrid_ensemble.csv';checks=write_submission(path,data,ranks,ctx.seen)
        manifest['submissions']['hybrid_ensemble']=dict(path=str(path.resolve()),sha256=sha256(path),parameters=dict(method='frozen_stacking_and_nested_rank_fusion',selected_graph_nodes=len(plan['nodes']),selected_prediction_leaves=len(plan['leaves'])),checks=checks)
        manifest.update(status='complete',root_scores=str(root),root_scores_sha256=sha256(root),rankers=rankers)
        dump(out/'manifest.json',manifest);print('COMPLETE '+str(out/'manifest.json'),flush=True)

if __name__=='__main__':main()
