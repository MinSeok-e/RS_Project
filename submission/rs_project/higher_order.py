import numpy as np

from scipy.linalg import solve

def choose_pairs(x,count,minimum_support=5):
    pop=x.sum(0);all_left=[];all_right=[];all_scores=[]
    for start in range(0,x.shape[1],256):
        end=min(start+256,x.shape[1]);overlap=x[:,start:end].T@x
        score=overlap/np.maximum(np.sqrt(pop[start:end,None]*pop[None,:]),1)
        eligible=(np.arange(start,end)[:,None]<np.arange(x.shape[1])[None,:])&(overlap>=minimum_support)
        flat=np.flatnonzero(eligible);take=min(count,len(flat))
        if not take:continue
        best=flat[np.argpartition(-score.ravel()[flat],take-1)[:take]]
        all_left.append(best//x.shape[1]+start);all_right.append(best%x.shape[1]);all_scores.append(score.ravel()[best])
    left=np.concatenate(all_left);right=np.concatenate(all_right);scores=np.concatenate(all_scores)
    order=np.lexsort((right,left,-scores))[:count]
    return left[order],right[order]

def pair_design(x,pairs,scale):
    left,right=pairs
    return np.concatenate((x,x[:,left]*x[:,right]*scale),axis=1).astype(np.float64)

def constrained_predict(x,queries,pairs,scale,reg):
    a=pair_design(x,pairs,scale);n_items=x.shape[1]
    kernel=a@a.T;kernel.flat[::len(kernel)+1]+=reg
    inverse_a=solve(kernel,a,assume_a='pos');inverse_x=inverse_a[:,:n_items]
    designs=[pair_design(q,pairs,scale) for q in queries]
    projections=[q@a.T for q in designs]
    predictions=[t@inverse_x for t in projections]
    constraints=[[j] for j in range(n_items)]
    for k,(left,right) in enumerate(zip(*pairs)):
        constraints[left].append(n_items+k);constraints[right].append(n_items+k)
    for j,columns in enumerate(constraints):
        ac=a[:,columns];rc=inverse_a[:,columns]
        pcc=np.eye(len(columns))-ac.T@rc
        b0=ac.T@inverse_x[:,j]
        correction=solve(pcc,b0,assume_a='pos')
        for q,t,pred in zip(designs,projections,predictions):
            pred[:,j]-=(q[:,columns]-t@rc)@correction
    return predictions
