"""Fit-only sequence windows and losses for controlled SASRec development."""

from dataclasses import dataclass

import numpy as np

import torch

from torch.nn import functional as F

@dataclass
class Windows:
    users: np.ndarray
    tokens: np.ndarray
    targets: np.ndarray
    positions: np.ndarray

def shuffled_tail_histories(histories, maxlen, seed):
    """Shuffle the interior of the retained L+1 items; keep both endpoints.

    This preserves training input/target item sets, inference input item sets,
    the last item, and each sequence length. Excluded older fit items remain
    excluded from the negative pools, which are built from original histories.
    This ablates interior order, NOT the identity of the most recent item.
    """
    rng = np.random.default_rng(seed)
    shuffled = []
    for history in histories:
        seq = history[-(maxlen + 1):].copy()
        seq[1:-1] = rng.permutation(seq[1:-1])
        shuffled.append(seq)
    return shuffled

def make_windows(histories, maxlen, mode='tail', horizon=1, stride=50, ends=None):
    if mode not in ('tail', 'coverage', 'prefix') or horizon < 1 or not 1 <= stride <= maxlen:
        raise ValueError('Invalid window configuration')
    specs = []
    for user, history in enumerate(histories):
        if len(history) < 2:
            raise ValueError('Training requires at least two fit interactions')
        if mode == 'coverage':
            for first_target in range(1, len(history), stride):
                end = min(first_target + stride, len(history))
                start = max(0, end - 1 - maxlen)
                specs.append((user, start, end, first_target))
        else:
            end = int(ends[user]) if mode == 'prefix' else len(history)
            if not 2 <= end <= len(history):
                raise ValueError('Prefix endpoint outside fit history')
            start = max(0, end - 1 - maxlen)
            specs.append((user, start, end, start + 1))
    tokens = np.zeros((len(specs), maxlen), dtype=np.int64)
    targets = np.zeros((len(specs), maxlen, horizon), dtype=np.int64)
    positions = np.full((len(specs), maxlen), -1, dtype=np.int64)
    users = np.empty(len(specs), dtype=np.int64)
    for row, (user, start, end, first_target) in enumerate(specs):
        history = histories[user]
        users[row] = user
        source = history[start:end-1]
        tokens[row, :len(source)] = source + 1
        for offset, position in enumerate(range(start + 1, end)):
            if position < first_target:
                continue  # Overlap is input context, not another weighted target.
            future = history[position:min(position + horizon, len(history))]
            targets[row, offset, :len(future)] = future + 1
            positions[row, offset] = position
    return Windows(users, tokens, targets, positions)

def prefix_endpoints(histories, rng, tail_probability=0.5):
    ends = []
    for history in histories:
        if rng.random() < tail_probability:
            ends.append(len(history))
        else:
            ends.append(int(rng.integers(2, len(history) + 1)))
    return ends

def full_catalog_loss(model, tokens, targets, active_tokens, temperature=1.0,
                      label_smoothing=0.0):
    """Full-softmax CE. With H>1, use uniform mass on the next H FIT items.

    Validation items/labels never enter this function. Cold-fit items are not
    softmax classes. H=1 is ordinary next-item cross entropy.
    """
    if temperature <= 0 or not 0 <= label_smoothing < 1:
        raise ValueError('Invalid loss parameters')
    mask = targets[..., 0] > 0
    hidden = model.encode(tokens)[mask]
    logits = hidden @ model.item_embedding(active_tokens).T / temperature
    wanted = targets[mask]
    valid = wanted > 0
    # active_tokens are sorted; padding positions are replaced before lookup.
    lookup_tokens = torch.where(valid, wanted, active_tokens[0])
    indices = torch.searchsorted(active_tokens, lookup_tokens.contiguous())
    if (indices >= len(active_tokens)).any() or not torch.equal(active_tokens[indices], lookup_tokens):
        raise ValueError('A training label is absent from the fit-active catalog')
    if wanted.shape[1] == 1:
        return F.cross_entropy(logits, indices[:, 0], label_smoothing=label_smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -(log_probs.gather(1, indices) * valid).sum(1) / valid.sum(1)
    return ((1 - label_smoothing) * nll - label_smoothing * log_probs.mean(1)).mean()

def update_coverage(seen_positions, windows):
    for user, positions in zip(windows.users, windows.positions):
        valid = positions >= 0
        seen_positions[user][positions[valid]] = True
