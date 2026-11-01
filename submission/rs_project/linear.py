import numpy as np

from scipy.linalg import solve

from scipy import sparse

def ease_predict(z, q, regularization=300, dropout=0.0):
    """General diagonal ridge, zero diagonal, arbitrary fit-only query histories."""
    z, q = np.asarray(z, dtype=np.float64), np.asarray(q, dtype=np.float64)
    d = regularization + dropout / (1-dropout) * (z*z).sum(0)
    zd = z/d
    a = zd @ z.T
    a.flat[::len(a)+1] += 1
    az = solve(a, zd, assume_a='pos')
    diag = 1/d - (zd*az).sum(0)
    projection = (q/d) @ z.T @ az
    return q - (q/d - projection)/diag

def user_knn(z, q, targets, k, shrink=0):
    overlap = q @ z.T
    denom = np.sqrt((q*q).sum(1)[:, None] * (z*z).sum(1)[None, :])
    sim = overlap / np.maximum(denom + shrink, 1e-12)
    np.fill_diagonal(sim, 0)
    order = np.argsort(-sim, axis=1, kind='stable')
    keep = np.zeros_like(sim)
    np.put_along_axis(keep, order[:, :k], np.take_along_axis(sim, order[:, :k], 1), 1)
    return keep @ targets

def transitions(histories, n_items, horizon):
    rows, cols, weights = [], [], []
    for h in histories:
        for t in range(len(h)-1):
            future = h[t+1:t+1+horizon]
            rows.extend([int(h[t])]*len(future)); cols.extend(future.tolist())
            weights.extend([1/len(future)]*len(future))
    return sparse.csr_matrix((weights, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
