"""Inspect only the competition CSVs; do not load the raw MovieLens dataset."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

if not os.getenv('SLURM_JOB_ID'):
    raise RuntimeError('Run on a Slurm compute node')
root = Path(__file__).resolve().parents[1]
data = root.parent / 'data'

def read(name):
    with (data / name).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)

train_columns, train = read('train.csv')
test_columns, test = read('test.csv')
sample_columns, sample = read('sample_submission.csv')
by_user = defaultdict(list)
for row in train:
    by_user[int(row['userId'])].append((int(row['movieId']), float(row['rating']), int(row['time_idx'])))
users = set(by_user)
items = {int(row['movieId']) for row in train}
test_users = {int(row['userId']) for row in test}
lengths = [len(rows) for rows in by_user.values()]
nonprefix = []
for user, rows in sorted(by_user.items()):
    times = sorted(row[2] for row in rows)
    if times != list(range(len(times))):
        nonprefix.append({'userId': user, 'count': len(times), 'min': min(times), 'max': max(times),
                          'gaps': max(times) - min(times) + 1 - len(set(times))})
sample_items = [int(movie) for row in sample for movie in row['recommendations'].split()]
pairs = [(int(row['userId']), int(row['movieId'])) for row in train]
order = [(int(row['userId']), int(row['time_idx'])) for row in train]
result = {
    'job_id': os.environ['SLURM_JOB_ID'],
    'data_source': str(data),
    'raw_movielens_loaded': False,
    'sha256': {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
               for name in ('train.csv', 'test.csv', 'sample_submission.csv')},
    'train': {'columns': train_columns, 'rows': len(train), 'users': len(users), 'items': len(items),
              'user_id_min': min(users), 'user_id_max': max(users),
              'user_ids_are_zero_based_contiguous': sorted(users) == list(range(len(users))),
              'movie_id_min': min(items), 'movie_id_max': max(items),
              'missing_movie_ids_within_observed_range': max(items) - min(items) + 1 - len(items),
              'rating_histogram': dict(sorted(Counter(float(row['rating']) for row in train).items())),
              'rows_per_user_min': min(lengths), 'rows_per_user_median': statistics.median(lengths),
              'rows_per_user_max': max(lengths),
              'users_whose_time_idx_is_exactly_0_to_n_minus_1': len(users) - len(nonprefix),
              'time_idx_nonprefix_examples': nonprefix[:10],
              'time_idx_min': min(row[2] for rows in by_user.values() for row in rows),
              'time_idx_max': max(row[2] for rows in by_user.values() for row in rows),
              'rows_sorted_by_user_and_time_idx': order == sorted(order),
              'duplicate_user_movie_pairs': len(pairs) - len(set(pairs)),
              'missing_cells': sum(value in (None, '') for row in train for value in row.values())},
    'test': {'columns': test_columns, 'rows': len(test), 'users': len(test_users),
             'unknown_users': sorted(test_users - users),
             'train_users_without_test_row': sorted(users - test_users),
             'id_is_row_number': [int(row['id']) for row in test] == list(range(len(test))),
             'test_users_sorted': [int(row['userId']) for row in test] == sorted(test_users),
             'id_userId_mismatch_rows': sum(row['id'] != row['userId'] for row in test)},
    'sample_submission': {'columns': sample_columns, 'rows': len(sample),
                          'items_per_row': sorted({len(row['recommendations'].split()) for row in sample}),
                          'movie_id_min': min(sample_items), 'movie_id_max': max(sample_items),
                          'distinct_movies_absent_from_train': len(set(sample_items) - items),
                          'note': 'Format example only; these lists are not ground-truth labels.'},
    'comparison_to_published_ml_latest_small_metadata': {
        'published_users': 610, 'published_ratings': 100836, 'published_movies_including_tagged_only': 9742,
        'train_fraction_if_this_is_the_source': len(train) / 100836,
        'unprovided_ratings_if_this_is_the_source': 100836 - len(train),
        'note': 'Metadata comparison is evidence for a hypothesis, not proof of source identity or split code.'}
}
output = root / 'results' / f'data_audit_job{os.environ["SLURM_JOB_ID"]}.json'
output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, indent=2, ensure_ascii=False))

assignment = root / 'RS Project.docx'
with zipfile.ZipFile(assignment) as archive:
    document = ET.fromstring(archive.read('word/document.xml'))
namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', namespace))
              for p in document.findall('.//w:p', namespace)]
(root / 'report/project_assignment.txt').write_text('\n'.join(paragraphs) + '\n')
print('Assignment text saved under ignored report/ directory.')
