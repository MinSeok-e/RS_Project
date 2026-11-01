"""Fixed-configuration full-train refit of a selected FMLP model and seed bag."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rs_project.data import load_data,sha256
from rs_project.submission_bases import full_context,fit_prefix
from rs_project.ensemble import transform
from rs_project.evaluation import topk
from rs_project.io_utils import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--selection',required=True);p.add_argument('--seeds',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Use a Slurm GPU')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    project=Path(args.project);row=json.loads(Path(args.selection).read_text());c=dict(row['parameters']);epochs=row['best_epoch'];data=load_data(project.parent/'data');ctx=full_context(data)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);members=[];predictions=[]
    with threadpool_limits(limits=2):
        for seed in map(int,args.seeds.split(',')):
            params=dict(c,seed=seed);print('FIT',seed,epochs,flush=True);model,scores=fit_prefix(ctx,params,epochs)
            path=out/f'seed{seed}.npy';np.save(path,scores);checkpoint=out/f'seed{seed}.pt';torch.save({'state_dict':model.cpu().state_dict(),'config':params,'epochs':epochs},checkpoint)
            members.append(dict(seed=seed,path=str(path.resolve()),sha256=sha256(path),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha256(checkpoint)))
            predictions.append(transform(scores,~ctx.seen&ctx.active[None,:],'rrf'));del model,scores;torch.cuda.empty_cache();print('DONE',seed,flush=True)
    path=out/'bag_rrf.npy';np.save(path,np.mean(predictions,axis=0));topk(np.load(path),ctx.seen,ctx.active)
    dump(out/'complete.json',dict(status='complete',parameters=c,epochs=epochs,data_hashes=data.hashes,fit_interactions=len(data.users),records=members,path=str(path.resolve()),sha256=sha256(path),selected_validation=row['metrics'],source_file=str(Path(__file__).resolve()),source_sha256=sha256(__file__)))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
