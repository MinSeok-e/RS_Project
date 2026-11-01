import unittest
import numpy as np
import torch
from torch.nn import functional as F
from rs_project.sasrec import SASRec, training_arrays, inference_array
from rs_project.sequence_training import (make_windows, shuffled_tail_histories,
                                          full_catalog_loss, update_coverage)


class SequenceWindowTests(unittest.TestCase):
    def setUp(self):
        self.histories = [np.arange(12), np.arange(20, 24)]

    def test_tail_matches_existing_training_inputs_and_labels(self):
        windows = make_windows(self.histories, 5, stride=2)
        tokens, positive = training_arrays(self.histories, 5)
        np.testing.assert_array_equal(windows.tokens, tokens)
        np.testing.assert_array_equal(windows.targets[..., 0], positive)

    def test_coverage_trains_each_fit_transition_once(self):
        windows = make_windows(self.histories, 5, mode='coverage', stride=2)
        observed = [[] for _ in self.histories]
        for user, positions, tokens, labels in zip(windows.users, windows.positions, windows.tokens, windows.targets):
            for offset, position in enumerate(positions):
                if position < 0:
                    continue
                observed[user].append(int(position))
                self.assertEqual(labels[offset, 0], self.histories[user][position] + 1)
                # Every visible token at this causal position comes before its label.
                self.assertTrue((tokens[:offset+1] - 1 < self.histories[user][position]).all())
        for user, history in enumerate(self.histories):
            self.assertEqual(sorted(observed[user]), list(range(1, len(history))))
        seen = [np.zeros(len(h), bool) for h in self.histories]
        update_coverage(seen, windows)
        self.assertEqual(sum(p.sum() for p in seen), sum(len(h)-1 for h in self.histories))

    def test_order_control_preserves_items_lengths_and_last_item(self):
        shuffled = shuffled_tail_histories(self.histories, 5, 10)
        original_windows = make_windows(self.histories, 5, stride=2)
        shuffled_windows = make_windows(shuffled, 5, stride=2)
        for a, b in ((original_windows.tokens, shuffled_windows.tokens),
                     (original_windows.targets[..., 0], shuffled_windows.targets[..., 0]),
                     (inference_array(self.histories, 5), inference_array(shuffled, 5))):
            np.testing.assert_array_equal(np.sort(a, axis=1), np.sort(b, axis=1))
        for original, altered in zip(self.histories, shuffled):
            self.assertEqual(original[-1], altered[-1])
        self.assertFalse(np.array_equal(original_windows.tokens[0], shuffled_windows.tokens[0]))

    def test_prefix_and_multi_future_targets_use_only_fit_history(self):
        windows = make_windows(self.histories, 5, 'prefix', 1, 2, ends=[4, 2])
        np.testing.assert_array_equal(windows.targets[0, :3, 0], [2, 3, 4])
        self.assertEqual(windows.targets[1, 0, 0], 22)
        multi = make_windows(self.histories, 5, 'coverage', 3, 2)
        for user, positions, labels in zip(multi.users, multi.positions, multi.targets):
            for offset, position in enumerate(positions):
                if position < 0:
                    self.assertTrue((labels[offset] == 0).all())
                    continue
                expected = self.histories[user][position:position+3] + 1
                np.testing.assert_array_equal(labels[offset, :len(expected)], expected)
                self.assertTrue((labels[offset, len(expected):] == 0).all())


class CatalogLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)
        self.model = SASRec(8, maxlen=3, dim=8, blocks=1, heads=2, dropout=0)
        self.model.eval()
        self.tokens = torch.tensor([[1, 3, 0]])
        self.active = torch.tensor([1, 3, 5, 7])

    def test_full_catalog_next_item_ce_matches_manual_loss(self):
        targets = torch.tensor([[[3], [5], [0]]])
        loss = full_catalog_loss(self.model, self.tokens, targets, self.active)
        hidden = self.model.encode(self.tokens)[0, :2]
        logits = hidden @ self.model.item_embedding(self.active).T
        expected = F.cross_entropy(logits, torch.tensor([1, 2]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.item_embedding.weight.grad).all())
        self.assertTrue((self.model.item_embedding.weight.grad[[0, 2, 4, 6, 8]] == 0).all())

    def test_future_target_distribution_and_cold_class_rejection(self):
        targets = torch.tensor([[[3, 5], [5, 0], [0, 0]]])
        hidden = self.model.encode(self.tokens)[0, :2]
        log_probs = F.log_softmax(hidden @ self.model.item_embedding(self.active).T, dim=-1)
        expected = -((log_probs[0, 1] + log_probs[0, 2])/2 + log_probs[1, 2])/2
        torch.testing.assert_close(full_catalog_loss(self.model, self.tokens, targets, self.active), expected)
        with self.assertRaises(ValueError):
            full_catalog_loss(self.model, self.tokens, torch.tensor([[[8], [5], [0]]]), self.active)


if __name__ == '__main__':
    unittest.main()
