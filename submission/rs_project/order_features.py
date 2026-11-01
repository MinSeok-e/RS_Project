import numpy as np

from scipy.optimize import minimize

from scipy.special import expit

def fit_order(histories,n_items,regularization):
    left=[];right=[];weights=[]
    for h in histories:
        a=[];b=[]
        for gap in [1,5,20,50]:
            if len(h)>gap:a.extend(h[:-gap]);b.extend(h[gap:])
        if a:
            left.extend(a);right.extend(b);weights.extend([1/(len(a)*len(histories))]*len(a))
    left=np.asarray(left);right=np.asarray(right);weights=np.asarray(weights)
    def objective(theta):
        delta=theta[right]-theta[left]
        loss=np.dot(weights,np.logaddexp(0,-delta))+.5*regularization*np.mean(theta*theta)
        derivative=-weights*expit(-delta)
        grad=np.bincount(right,weights=derivative,minlength=n_items)-np.bincount(left,weights=derivative,minlength=n_items)
        grad+=regularization*theta/n_items
        return loss,grad
    result=minimize(objective,np.zeros(n_items),method='L-BFGS-B',jac=True,options={'maxiter':150,'ftol':1e-9,'gtol':1e-7})
    if not np.isfinite(result.x).all():raise ValueError('Nonfinite learned ordering')
    return result.x,{'iterations':int(result.nit),'success':bool(result.success),'message':str(result.message),
                     'weighted_pair_accuracy':float(np.dot(weights,(result.x[right]>result.x[left]))/weights.sum())}

def order_features(ctx,users,items):
    values=[];names=[];checks=[]
    for reg in [.03,.3,3.]:
        theta,check=fit_order(ctx.histories,ctx.x.shape[1],reg)
        theta/=max(theta[ctx.active].std(),1e-8)
        prefix=f'order_reg{str(reg).replace(".","p")}'
        values.append(theta[items]);names.append(prefix+'_item')
        stats={key:np.zeros(len(ctx.histories)) for key in ['all_mean','all_std','last','last5_mean','last20_mean','last20_std','last20_max','last20_min']}
        for u,h in enumerate(ctx.histories):
            t=theta[h]
            stats['all_mean'][u]=t.mean();stats['all_std'][u]=t.std();stats['last'][u]=t[-1]
            stats['last5_mean'][u]=t[-5:].mean();stats['last20_mean'][u]=t[-20:].mean();stats['last20_std'][u]=t[-20:].std()
            stats['last20_max'][u]=t[-20:].max();stats['last20_min'][u]=t[-20:].min()
        for key,a in stats.items():
            values.append(a[users]);names.append(prefix+'_'+key)
            if 'std' not in key:
                values.append(theta[items]-a[users]);names.append(prefix+'_delta_'+key)
        checks.append({'regularization':reg,**check})
    return np.stack(values,axis=1).astype(np.float32),names,checks
