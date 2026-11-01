import numpy as np

def group_of(name):
    if name.startswith(('svd','neighbor_')):return 'latent'
    if name.startswith(('cosine_','jaccard_','lift_')):return 'similarity'
    if name.startswith('directed_'):return 'direction'
    if name.startswith('order_reg'):return 'order'
    if '_stack_' in name:return 'diverse'
    if '_past_pool_ndcg_' in name or name=='past_quality_queries':return 'quality'
    return 'base'

def select_columns(names,selection):
    if selection=='all':return None
    wanted=set(selection.split(','));known={group_of(name) for name in names}
    if not wanted<=known:raise ValueError(f'Unknown feature groups: {wanted-known}')
    columns=np.array([i for i,name in enumerate(names) if group_of(name) in wanted])
    if not len(columns):raise ValueError('Empty feature subset')
    return columns
