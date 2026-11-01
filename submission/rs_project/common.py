import numpy as np

def fast_topk(scores, seen, active, k=20):
    scores = np.asarray(scores)
    if scores.shape != seen.shape or not np.isfinite(scores).all():
        raise ValueError('Nonfinite or incorrectly shaped scores')
    a = scores.copy()
    a[seen | ~active[None, :]] = -np.inf
    if ((~seen & active[None, :]).sum(1) < k).any():
        raise ValueError('Insufficient candidates')
    cutoff = np.partition(a, -k, axis=1)[:, -k]
    result = np.empty((len(a), k), dtype=np.int64)
    for u in range(len(a)):
        greater = np.flatnonzero(a[u] > cutoff[u])
        equal = np.flatnonzero(a[u] == cutoff[u])[:k-len(greater)]
        selected = np.concatenate((greater, equal))
        result[u] = selected[np.lexsort((selected, -a[u, selected]))]
    return result

def weighted_history(search, half_life=0, tail=0, rating_power=0):
    data = search.data
    q = np.zeros_like(search.x)
    for u, h in enumerate(search.histories):
        rows = data.train_rows[data.users[data.train_rows] == u]
        weights = np.ones(len(h), dtype=np.float32)
        if half_life:
            weights *= np.exp2(-np.arange(len(h)-1, -1, -1)/half_life)
        if tail:
            weights[:-tail] = 0
        if rating_power:
            weights *= (data.ratings[rows]/5.0) ** rating_power
        q[u, h] = weights
    return q
