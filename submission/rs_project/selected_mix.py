"""Export a fixed submitted blend from verified full-history model predictions."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .data import load_data,sha256
from .evaluation import topk,write_submission
from .ensemble import transform
from .submission_graph import load_scores
from .io_utils import dump


def combine(config,seen,active):
    eligible=~seen&active[None,:];result=None;total=0.
    for entry in config['inputs']:
        path=Path(entry['path'])
        if sha256(path)!=entry['sha256']:raise ValueError('Selected prediction changed')
        scores=load_scores(path)
        if scores.shape!=seen.shape or not np.isfinite(scores).all():raise ValueError('Invalid full-history scores')
        weight=float(entry['weight'])
        if weight<0:raise ValueError('Negative mixture weight')
        value=scores if config['method']=='raw' else transform(scores,eligible,config['method'])
        if result is None:result=np.zeros_like(value,dtype=np.float64)
        result+=weight*value;total+=weight
    if total<=0:raise ValueError('Empty blend')
    return result/total


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--project',required=True);parser.add_argument('--data',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use a Slurm compute node')
    source=Path(__file__).resolve().parents[1];project=Path(args.project);config=json.loads(Path(args.config).read_text())
    data=load_data(args.data)
    if config['data_hashes']!=data.hashes:raise ValueError('Competition inputs changed')
    for artifact in config.get('artifacts',[]):
        if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('Selected training artifact changed')
    seen=data.matrix(np.arange(len(data.users)),dtype=bool);active=seen.any(0)
    if config['fit_interactions']!=len(data.users):raise ValueError('Expected final full-history inputs')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    scores=combine(config,seen,active);scorepath=out/'scores.npy';np.save(scorepath,scores)
    csv=out/'recommendations.csv';checks=write_submission(csv,data,topk(scores,seen,active),seen)
    files=[p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    hashes={str(p.relative_to(source)):sha256(p) for p in files}
    commit=os.environ['RS_SOURCE_COMMIT']
    for rel,digest in hashes.items():
        import hashlib
        committed=subprocess.check_output(['git','-C',str(project),'show',f'{commit}:submission/{rel}'])
        if hashlib.sha256(committed).hexdigest()!=digest:raise ValueError('Uncommitted execution source')
    manifest=dict(run_id=out.name,status='complete',smoke=False,utc=datetime.now(timezone.utc).isoformat(),source_commit=commit,source_root=str(source),source_sha256=hashes,data_hashes=data.hashes,fit_interactions=len(data.users),selected_config=config,quota_utc_date=config['quota_utc_date'],root_scores=str(scorepath.resolve()),submissions={'hybrid_ensemble':{'path':str(csv.resolve()),'sha256':sha256(csv),'parameters':{'keyword':config['keyword'],'method':config['method']},'checks':checks}})
    dump(out/'manifest.json',manifest);print('COMPLETE '+str(out/'manifest.json'),flush=True)

if __name__=='__main__':main()
