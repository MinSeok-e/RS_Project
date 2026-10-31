import math
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from rs_project.data import load_data
from rs_project.evaluation import ranking_metrics, topk
from rs_project.models import EASE, SLIM, RP3Beta
from rs_project.sasrec import SASRec, training_arrays, inference_array, negative_pools, sample_negatives


class ClassicalTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0],
                           [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float)

    def test_ease_matches_item_space_closed_form(self):
        lam = 3.0
        inverse = np.linalg.inv(self.x.T @ self.x + lam * np.eye(4))
        weights = -inverse / np.diag(inverse)
        np.fill_diagonal(weights, 0)
        model = EASE(lam).fit(self.x)
        np.testing.assert_allclose(model.predict(self.x), self.x @ weights, atol=1e-12)

    def test_rp3_matches_dense_walk_with_target_degree_penalty(self):
        alpha, beta = 0.7, 0.4
        pui = (self.x / self.x.sum(axis=1)[:, None]) ** alpha
        piu = (self.x.T / self.x.sum(axis=0)[:, None]) ** alpha
        expected = (piu @ pui) * self.x.sum(axis=0) ** (-beta)
        np.fill_diagonal(expected, 0)
        model = RP3Beta(alpha, beta, topk=4, normalize_similarity=False).fit(self.x)
        np.testing.assert_allclose(model.weights.toarray(), expected)
        np.testing.assert_allclose(model.predict(self.x), self.x @ expected)

    def test_rp3_prunes_and_handles_cold_columns(self):
        x = np.column_stack((self.x, np.zeros(len(self.x))))
        model = RP3Beta(topk=1).fit(x)
        self.assertTrue((np.diff(model.weights.indptr) <= 1).all())
        self.assertTrue((model.weights.diagonal() == 0).all())
        self.assertEqual(model.weights[:, -1].nnz, 0)
        self.assertTrue(np.isfinite(model.predict(x)).all())

    def test_slim_diagonal_positivity_convergence_and_threads(self):
        x = np.tile(self.x, (5, 1))
        params = dict(alpha=0.01, l1_ratio=0.2, topk=4, tol=1e-7)
        serial = SLIM(workers=1, **params).fit(x)
        threaded = SLIM(workers=2, **params).fit(x)
        w = serial.weights.toarray()
        self.assertTrue((w >= 0).all())
        self.assertTrue((np.diag(w) == 0).all())
        self.assertGreater(w[0, 1], w[3, 1])
        self.assertEqual(serial.diagnostics['unconverged_targets'], 0)
        np.testing.assert_allclose(threaded.weights.toarray(), w, atol=1e-12)
        np.testing.assert_array_equal(x, np.tile(self.x, (5, 1)))


class ProtocolTests(unittest.TestCase):
    def test_chronological_split_rounding_and_input_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rows = ['userId,movieId,rating,time_idx']
            for u in range(2):
                rows += [f'{u},{u*10+t},0.5,{t}' for t in reversed(range(6))]
            (path / 'train.csv').write_text('\n'.join(rows) + '\n')
            (path / 'test.csv').write_text('id,userId\n0,1\n')
            (path / 'sample_submission.csv').write_text('id,recommendations\n0,1 2\n')
            data = load_data(path)
            for u in range(2):
                fit = data.train_rows[data.users[data.train_rows] == u]
                val = data.valid_rows[data.users[data.valid_rows] == u]
                self.assertEqual(len(fit), 4)
                self.assertEqual(len(val), 2)
                self.assertLess(data.times[fit].max(), data.times[val].min())
            self.assertTrue((data.matrix(data.train_rows)[data.users[data.train_rows], data.items[data.train_rows]] == 1).all())

    def test_cold_target_counts_in_idcg(self):
        metrics, _ = ranking_metrics(np.array([[1, 2]]), [np.array([2, 3, 4])], k=2)
        expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics['ndcg@2'], expected)

    def test_seen_cold_exclusion_and_stable_ties(self):
        result = topk(np.array([[100., 2, 2, 90]]), np.array([[1, 0, 0, 0]]),
                      np.array([1, 1, 1, 0]), k=2)
        np.testing.assert_array_equal(result, [[1, 2]])


class SequentialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2020)
        torch.set_num_threads(1)
        self.model = SASRec(8, maxlen=5, dim=8, blocks=2, heads=2, dropout=0)
        self.model.eval()

    def test_causality_and_right_padding(self):
        with torch.no_grad():
            a = self.model.encode(torch.tensor([[1, 2, 3, 4]]))
            b = self.model.encode(torch.tensor([[1, 2, 7, 8]]))
            torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-6)
            short = self.model.score(torch.tensor([[1, 2]]))
            padded = self.model.score(torch.tensor([[1, 2, 0, 0]]))
            torch.testing.assert_close(short, padded, atol=1e-6, rtol=1e-6)

    def test_sequence_alignment_negative_exclusion_and_update(self):
        histories = [np.array([0, 1, 2, 3]), np.array([4, 5])]
        tokens, pos = training_arrays(histories, 3)
        np.testing.assert_array_equal(tokens, [[1, 2, 3], [5, 0, 0]])
        np.testing.assert_array_equal(pos, [[2, 3, 4], [6, 0, 0]])
        np.testing.assert_array_equal(inference_array(histories, 3), [[2, 3, 4], [5, 6, 0]])
        pools = negative_pools(histories, np.arange(8))
        neg = sample_negatives(np.array([0, 1]), pos, pools, np.random.default_rng(1))
        for u in range(2):
            self.assertFalse(np.isin(neg[u, pos[u] > 0] - 1, histories[u]).any())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        before = self.model.item_embedding.weight.detach().clone()
        loss = self.model.loss(torch.tensor(tokens), torch.tensor(pos), torch.tensor(neg))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.item_embedding.weight))
        self.assertTrue((self.model.item_embedding.weight[0] == 0).all())


if __name__ == '__main__':
    unittest.main()
