"""Read only the competition CSVs and build a fixed chronological holdout."""
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path, columns):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f'{path}: expected columns {columns}, got {reader.fieldnames}')
        rows = list(reader)
    if not rows or any(any(value is None or value == '' for value in row.values()) for row in rows):
        raise ValueError(f'{path}: empty file or missing values')
    return rows


@dataclass
class Data:
    user_ids: np.ndarray
    item_ids: np.ndarray
    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    times: np.ndarray
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_ids: np.ndarray
    test_users: np.ndarray
    hashes: dict

    @property
    def shape(self):
        return len(self.user_ids), len(self.item_ids)

    def matrix(self, rows, dtype=np.float64):
        x = np.zeros(self.shape, dtype=dtype)
        x[self.users[rows], self.items[rows]] = 1
        return x

    def targets(self):
        return [self.items[self.valid_rows[self.users[self.valid_rows] == u]] for u in range(self.shape[0])]

    def summary(self):
        x = self.matrix(self.train_rows, dtype=bool)
        active = x.any(axis=0)
        targets = self.targets()
        cold = ~active[self.items[self.valid_rows]]
        return {
            'file_sha256': self.hashes,
            'interactions': len(self.users), 'users': self.shape[0], 'items': self.shape[1],
            'test_users': len(self.test_ids), 'fit_interactions': len(self.train_rows),
            'validation_interactions': len(self.valid_rows),
            'validation_users': sum(len(t) > 0 for t in targets),
            'fit_active_items': int(active.sum()),
            'validation_only_items': int((~active).sum()),
            'cold_validation_interactions': int(cold.sum()),
            'validation_users_with_only_cold_targets': sum(len(t) > 0 and not active[t].any() for t in targets),
            'rating_values': np.unique(self.ratings).tolist(),
            'test_id_userId_mismatches': int(np.sum(self.test_ids != self.user_ids[self.test_users])),
            'split_sha256': hashlib.sha256(self.train_rows.astype('<i8').tobytes() + b'|' + self.valid_rows.astype('<i8').tobytes()).hexdigest(),
        }


def load_data(directory, valid_fraction=0.2):
    directory = Path(directory)
    if not 0 < valid_fraction < 1:
        raise ValueError('valid_fraction must be strictly between 0 and 1')
    rows = read_csv(directory / 'train.csv', ['userId', 'movieId', 'rating', 'time_idx'])
    raw_u = np.array([int(r['userId']) for r in rows], dtype=np.int64)
    raw_i = np.array([int(r['movieId']) for r in rows], dtype=np.int64)
    ratings = np.array([float(r['rating']) for r in rows], dtype=np.float64)
    times = np.array([int(r['time_idx']) for r in rows], dtype=np.int64)
    if not np.isfinite(ratings).all():
        raise ValueError('Non-finite ratings')
    if len(set(zip(raw_u, raw_i))) != len(rows):
        raise ValueError('Duplicate user/movie pairs require an explicit policy')
    user_ids, users = np.unique(raw_u, return_inverse=True)
    item_ids, items = np.unique(raw_i, return_inverse=True)
    train, valid = [], []
    for u in range(len(user_ids)):
        ix = np.flatnonzero(users == u)
        ix = ix[np.argsort(times[ix], kind='stable')]
        if len(ix) < 2 or len(np.unique(times[ix])) != len(ix):
            raise ValueError(f'User {user_ids[u]} needs >=2 interactions and unique time_idx')
        n_valid = min(len(ix) - 1, max(1, int(np.ceil(len(ix) * valid_fraction))))
        train.extend(ix[:-n_valid])
        valid.extend(ix[-n_valid:])
    test = read_csv(directory / 'test.csv', ['id', 'userId'])
    sample = read_csv(directory / 'sample_submission.csv', ['id', 'recommendations'])
    test_ids = np.array([int(r['id']) for r in test], dtype=np.int64)
    test_raw_u = [int(r['userId']) for r in test]
    if len(set(test_ids)) != len(test_ids) or len(set(test_raw_u)) != len(test_raw_u):
        raise ValueError('Duplicate test id or userId')
    if test_ids.tolist() != [int(r['id']) for r in sample]:
        raise ValueError('Sample submission IDs/order differ from test.csv')
    u_map = {int(u): n for n, u in enumerate(user_ids)}
    if any(u not in u_map for u in test_raw_u):
        raise ValueError('Cold-start test user: define a fallback before proceeding')
    return Data(user_ids, item_ids, users, items, ratings, times,
                np.asarray(train, dtype=np.int64), np.asarray(valid, dtype=np.int64),
                test_ids, np.asarray([u_map[u] for u in test_raw_u]),
                {name: sha256(directory / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv')})
