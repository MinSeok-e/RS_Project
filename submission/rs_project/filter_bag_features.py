import numpy as np

from .ensemble import transform

SOURCES=[(81626,'filter_mlp_events_past',2020)]+[(81648,f'filter_mlp_seed{seed}_events_past',seed) for seed in (2021,2022,2023)]

VIEW_NAMES=['prediction','profile_all','profile_recent20']

FEATURE_NAMES=[f'filter_bag_{label}_{kind}_mean' for label in VIEW_NAMES for kind in ['raw','zscore','rrf']]+['filter_bag_prediction_zscore_std']

def candidate_features(predictions,eligible,users,items):
    columns=[]
    for label in VIEW_NAMES:
        for kind in ['raw','zscore','rrf']:
            score=predictions[label]
            value=score if kind=='raw' else transform(score,eligible,kind)
            columns.append(value[users,items])
    return np.stack(columns,axis=1)

def aggregate_candidate_features(samples):
    values=np.stack(samples).astype(np.float64)
    if values.ndim!=3 or values.shape[0]<2 or values.shape[2]!=9 or not np.isfinite(values).all():raise ValueError('Invalid bag samples')
    return np.concatenate((values.mean(0),values[:,:,1].std(0)[:,None]),axis=1).astype(np.float32)
