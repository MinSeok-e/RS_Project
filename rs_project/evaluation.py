"""Full-catalog binary NDCG; score ties break by ascending original movie ID."""
import csv
from pathlib import Path
import numpy as np


def topk(scores, seen, active_items=None, k=20):
    scores = np.asarray(scores).copy()
    seen = np.asarray(seen, dtype=bool)
    if scores.shape != seen.shape or scores.ndim != 2:
        raise ValueError('Scores and seen matrix must have identical 2D shapes')
    if not np.isfinite(scores).all():
        raise ValueError('Model produced non-finite scores')
    eligible = ~seen
    if active_items is not None:
        active_items = np.asarray(active_items, dtype=bool)
        if active_items.shape != (scores.shape[1],):
            raise ValueError('Invalid active item mask')
        eligible &= active_items[None, :]
    if (eligible.sum(axis=1) < k).any():
        raise ValueError('Fewer than k eligible items for at least one user')
    scores[~eligible] = -np.inf
    # Item columns are sorted by original movieId; stable sort fixes tied scores.
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def ranking_metrics(recommendations, targets, k=20):
    recommendations = np.asarray(recommendations)
    if recommendations.shape != (len(targets), k):
        raise ValueError('Recommendation/target shapes do not agree')
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    per_user = []
    for u, (ranking, target) in enumerate(zip(recommendations, targets)):
        target = np.unique(target)
        if not len(target):
            continue
        if len(np.unique(ranking)) != k:
            raise ValueError('Duplicate recommendation')
        hits = np.isin(ranking, target)
        idcg = discounts[:min(len(target), k)].sum()
        per_user.append({'user_index': u, 'ndcg': float(discounts[hits].sum() / idcg),
                         'recall': float(hits.sum() / len(target)), 'hit': float(hits.any()),
                         'targets': int(len(target))})
    if not per_user:
        raise ValueError('No validation users')
    metrics = {f'{metric}@{k}': float(np.mean([r[metric] for r in per_user])) for metric in ('ndcg', 'recall', 'hit')}
    metrics['users'] = len(per_user)
    return metrics, per_user


def evaluate(scores, fit_matrix, targets, k=20):
    return ranking_metrics(topk(scores, fit_matrix, fit_matrix.any(axis=0), k), targets, k)


def write_submission(path, data, rankings, all_seen, k=20):
    """Rankings are indexed by internal user; output rows follow test.csv IDs."""
    if rankings.shape != (len(data.user_ids), k):
        raise ValueError('Expected rankings for every mapped user')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'recommendations'])
        for row_id, user in zip(data.test_ids, data.test_users):
            ranking = rankings[user]
            if len(np.unique(ranking)) != k or (ranking < 0).any() or (ranking >= len(data.item_ids)).any():
                raise ValueError('Invalid ranked item indices')
            if all_seen[user, ranking].any():
                raise ValueError('Submission contains an observed item')
            writer.writerow([int(row_id), ' '.join(map(str, data.item_ids[ranking]))])
    return validate_submission(path, data, all_seen, k)


def validate_submission(path, data, all_seen, k=20):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'recommendations']:
            raise ValueError('Invalid submission header')
        rows = list(reader)
    if [int(r['id']) for r in rows] != data.test_ids.tolist():
        raise ValueError('Submission IDs/order do not match test.csv')
    i_map = {int(raw): i for i, raw in enumerate(data.item_ids)}
    for row, u in zip(rows, data.test_users):
        movies = [int(s) for s in row['recommendations'].split()]
        if len(movies) != k or len(set(movies)) != k or any(m not in i_map for m in movies):
            raise ValueError('Invalid submission movies')
        if any(all_seen[u, i_map[m]] for m in movies):
            raise ValueError('Submission contains previously observed movies')
    return {'rows': len(rows), 'items_per_user': k, 'id_order_matches': True,
            'duplicates': 0, 'unknown_items': 0, 'seen_items': 0}
