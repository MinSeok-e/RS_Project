"""EASE, positive SLIM ElasticNet, and RP3beta on binary interactions.

References: Steck (WWW 2019), Ning & Karypis (ICDM 2011),
Paudel et al. (ACM TiiS, doi:10.1145/2955101). See README for implementation references.
"""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize


class EASE:
    def __init__(self, regularization=300):
        if regularization <= 0:
            raise ValueError('regularization must be positive')
        self.regularization = regularization

    def fit(self, x):
        # Woodbury identity avoids an items x items inverse on this small-user data.
        x = np.asarray(x, dtype=np.float64)
        gram = x @ x.T
        gram.flat[::len(gram) + 1] += self.regularization
        ax = solve(gram, x, assume_a='pos')
        diagonal = (1 - np.sum(x * ax, axis=0)) / self.regularization
        if (diagonal <= 0).any():
            raise ValueError('Invalid inverse diagonal')
        self.scores = x - ax / diagonal
        self.diagnostics = {'solver': 'user-space Woodbury', 'regularization': self.regularization}
        return self

    def predict(self, x):
        # This factorization computes reconstruction for the fitted users only.
        if x.shape != self.scores.shape:
            raise ValueError('EASE currently predicts fitted users only')
        return self.scores


def retain_row_topk(matrix, k):
    """Keep positive top-k weights per source row, tie-breaking on target index."""
    matrix = matrix.tocsr(copy=True)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row + 2]
        weights = matrix.data[start:end]
        indices = matrix.indices[start:end]
        if len(weights) > k:
            keep = np.lexsort((indices, -weights))[:k]
            mask = np.ones(len(weights), dtype=bool)
            mask[keep] = False
            weights[mask] = 0
    matrix.eliminate_zeros()
    return matrix


class RP3Beta:
    def __init__(self, alpha=1.0, beta=0.6, topk=100, normalize_similarity=True):
        if alpha <= 0 or beta < 0 or topk < 1:
            raise ValueError('Invalid RP3beta hyperparameters')
        self.alpha, self.beta, self.topk = alpha, beta, topk
        self.normalize_similarity = normalize_similarity

    def fit(self, x):
        x = sparse.csr_matrix(x, dtype=np.float64)
        pui = normalize(x, norm='l1', axis=1)
        piu = normalize(x.T.tocsr(), norm='l1', axis=1)
        pui.data **= self.alpha
        piu.data **= self.alpha
        degree = np.asarray(x.sum(axis=0)).ravel()
        penalty = np.zeros_like(degree)
        penalty[degree > 0] = degree[degree > 0] ** (-self.beta)
        blocks = []
        # Row pruning is applied to outgoing item-item edges; no extra column pruning.
        for start in range(0, x.shape[1], 256):
            end = min(start + 256, x.shape[1])
            block = (piu[start:end] @ pui).multiply(penalty).tocsr()
            for offset, item in enumerate(range(start, end)):
                a, b = block.indptr[offset:offset + 2]
                block.data[a:b][block.indices[a:b] == item] = 0
            block.eliminate_zeros()
            blocks.append(retain_row_topk(block, self.topk))
        self.weights = sparse.vstack(blocks, format='csr')
        if self.normalize_similarity:
            self.weights = normalize(self.weights, norm='l1', axis=1)
        self.diagnostics = {'nnz': self.weights.nnz, 'pruning': 'row top-k',
                            'normalize_similarity': self.normalize_similarity}
        return self

    def predict(self, x):
        return (sparse.csr_matrix(x) @ self.weights).toarray()


class SLIM:
    """Positive ElasticNet per target item; diagonal constrained to zero.

    sklearn objective is ||y-Xw||^2/(2*n_users) + alpha*l1_ratio*||w||_1
    + alpha*(1-l1_ratio)*||w||^2/2. topk is per TARGET column after fitting.
    """
    def __init__(self, alpha=0.001, l1_ratio=0.1, topk=100,
                 max_iter=2000, tol=1e-4, workers=4):
        self.alpha, self.l1_ratio, self.topk = alpha, l1_ratio, topk
        self.max_iter, self.tol, self.workers = max_iter, tol, workers

    def fit(self, x):
        x = sparse.csc_matrix(x, dtype=np.float64)
        active = np.flatnonzero(np.asarray(x.sum(axis=0)).ravel() > 0)
        started = perf_counter()

        def fit_chunk(targets):
            design = x.copy()  # Private to this worker; no mutation shared between threads.
            rows, cols, values, gaps, iterations = [], [], [], [], []
            retried = 0
            for item in targets:
                a, b = design.indptr[item:item + 2]
                original = design.data[a:b].copy()
                y = design[:, item].toarray().ravel()
                design.data[a:b] = 0
                model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                   positive=True, fit_intercept=False, copy_X=False,
                                   max_iter=self.max_iter, tol=self.tol, selection='cyclic')
                # Numeric convergence checks below avoid process-wide warning filters
                # inside worker threads. sklearn can also emit a warning to the log.
                model.fit(design, y)
                threshold = self.tol * np.dot(y, y) / len(y)
                if model.dual_gap_ > threshold:
                    retried += 1
                    model.set_params(max_iter=self.max_iter * 10, warm_start=True)
                    model.fit(design, y)
                design.data[a:b] = original
                coef = model.coef_.copy()
                coef[item] = 0
                nz = np.flatnonzero(coef > 0)
                nz = nz[np.lexsort((nz, -coef[nz]))[:self.topk]]
                rows.extend(nz.tolist())
                cols.extend([int(item)] * len(nz))
                values.extend(coef[nz].tolist())
                gaps.append(float(model.dual_gap_ / max(threshold, 1e-30)))
                iterations.append(int(model.n_iter_))
            return rows, cols, values, gaps, iterations, retried

        chunks = np.array_split(active, min(self.workers, len(active)))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            parts = list(pool.map(fit_chunk, chunks))
        rows = [v for p in parts for v in p[0]]
        cols = [v for p in parts for v in p[1]]
        values = [v for p in parts for v in p[2]]
        gaps = [v for p in parts for v in p[3]]
        iterations = [v for p in parts for v in p[4]]
        unconverged = sum(g > 1.00001 for g in gaps)
        self.weights = sparse.csr_matrix((values, (rows, cols)), shape=(x.shape[1], x.shape[1]))
        self.diagnostics = {'nnz': self.weights.nnz, 'fit_targets': len(active),
                            'retried_targets': sum(p[5] for p in parts),
                            'unconverged_targets': unconverged,
                            'max_dual_gap_over_tolerance': max(gaps, default=0),
                            'max_iterations': max(iterations, default=0),
                            'fit_seconds': perf_counter() - started}
        if unconverged:
            raise RuntimeError(f'SLIM did not converge for {unconverged} targets: {self.diagnostics}')
        return self

    def predict(self, x):
        return (sparse.csr_matrix(x) @ self.weights).toarray()
