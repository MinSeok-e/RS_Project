"""Upload one full-data prediction per model, with versioning and duplicate protection."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

COMPETITION = 'skku-esw-4024-recommender-system'
MODELS = ('ease', 'slim', 'rp3beta', 'sasrec')


class SubmissionError(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def all_submissions(api):
    rows = {}
    page = 1
    while True:
        batch = api.competition_submissions(COMPETITION, page_number=page, page_size=100) or []
        if any(row.ref in rows for row in batch):
            raise SubmissionError('Repeated history page; version assignment stopped')
        rows.update((row.ref, row) for row in batch)
        if len(batch) < 100:
            return list(rows.values())
        page += 1


def next_number(existing, ledger):
    names = [row.file_name for row in existing] + [row['file_name'] for row in ledger.values()]
    versions = [int(match.group(1)) for name in names
                if (match := re.match(r'^MinseokJeong_v(\d+)_', name))]
    return max([len(existing), *versions]) + 1


def filename(number, model, params):
    if model == 'ease':
        keyword = f'EASE_lambda{params["regularization"]:g}'
    elif model == 'slim':
        keyword = f'SLIM_alpha{params["alpha"]:g}_l1ratio{params["l1_ratio"]:g}_k{params["topk"]}'
    elif model == 'rp3beta':
        keyword = f'RP3beta_alpha{params["alpha"]:g}_beta{params["beta"]:g}_k{params["topk"]}'
    elif model == 'sasrec':
        keyword = f'SASRec_d{params["dim"]}_len{params["maxlen"]}_ep{params["epochs"]}'
    else:
        raise SubmissionError('Unsupported model')
    return f'MinseokJeong_v{number}_{keyword}.csv'


def describe(row):
    return {'submission_id': row.ref, 'file_name': row.file_name, 'description': row.description,
            'date': str(row.date), 'status': getattr(row.status, 'name', str(row.status)),
            'public_score': row.public_score, 'error_description': row.error_description,
            'team_name': row.team_name}


def verify_csv(path, data_dir, expected_hash):
    if digest(path) != expected_hash:
        raise SubmissionError('Submission hash changed since model fitting')
    with (data_dir / 'train.csv').open() as handle:
        known, seen = set(), {}
        for row in csv.DictReader(handle):
            known.add(row['movieId'])
            seen.setdefault(row['userId'], set()).add(row['movieId'])
    with (data_dir / 'test.csv').open() as handle:
        test = list(csv.DictReader(handle))
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'recommendations']:
            raise SubmissionError('Incorrect CSV header')
        rows = list(reader)
    if len(rows) != len(test):
        raise SubmissionError('Incorrect submission row count')
    for row, target in zip(rows, test):
        movies = row['recommendations'].split()
        if row['id'] != target['id'] or len(movies) != 20 or len(set(movies)) != 20:
            raise SubmissionError('Incorrect IDs or ranked item count')
        if not set(movies).issubset(known) or set(movies) & seen[target['userId']]:
            raise SubmissionError('Unknown or previously observed movie')


def verify_manifest(manifest, data_dir):
    if manifest['status'] != 'complete' or manifest['smoke']:
        raise SubmissionError('Only a completed full-data run can be uploaded')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['source_commit']):
        raise SubmissionError('Submission source must have a recorded Git commit')
    for name, expected in manifest['data_hashes'].items():
        if digest(data_dir / name) != expected:
            raise SubmissionError('Competition data changed after fitting')
    for name, expected in manifest['source_sha256'].items():
        if digest(Path(manifest['source_root']) / name) != expected:
            raise SubmissionError('Executed source snapshot changed after fitting')
    if set(manifest['submissions']) != set(MODELS):
        raise SubmissionError('Expected exactly four model files')
    # Validate every file before performing any external write.
    for model in MODELS:
        details = manifest['submissions'][model]
        verify_csv(Path(details['path']), data_dir, details['sha256'])


def publish(api, root, manifest, action):
    ledger_path = root / 'submissions/kaggle_records.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for model in MODELS:
        key = f'{manifest["run_id"]}:{model}'
        details = manifest['submissions'][model]
        existing = all_submissions(api)
        record = ledger.get(key)
        if record:
            previous = next((row for row in existing if row.ref == record.get('submission_id')
                             or row.file_name == record['file_name']), None)
            if previous:
                record.update(describe(previous))
                ledger[key] = record
                write(ledger_path, ledger)
                print(json.dumps(record), flush=True)
                continue
            if record.get('attempted_utc'):
                raise SubmissionError('Prior upload attempt is not visible; inspect before any retry')
        if action == 'status':
            print(json.dumps({'model': model, 'status': 'NOT_SUBMITTED'}), flush=True)
            continue
        number = next_number(existing, ledger)
        name = filename(number, model, details['parameters'])
        upload = root / 'submissions' / name
        if upload.exists():
            if digest(upload) != details['sha256']:
                raise SubmissionError('Versioned filename already contains another prediction')
        else:
            with upload.open('xb') as handle:
                handle.write(Path(details['path']).read_bytes())
        record = {'model': model, 'run_id': manifest['run_id'], 'source_commit': manifest['source_commit'],
                  'sha256': details['sha256'], 'file_name': name, 'path': str(upload),
                  'description': upload.stem, 'submission_number': number,
                  'attempted_utc': datetime.now(timezone.utc).isoformat(), 'status': 'SUBMITTING'}
        ledger[key] = record
        write(ledger_path, ledger)
        response = api.competition_submit(str(upload), upload.stem, COMPETITION, quiet=True)
        record.update(submission_id=response.ref, status='SUBMITTED')
        write(ledger_path, ledger)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'submit', 'status'])
    parser.add_argument('--root', type=Path, default=Path('/home/minseok/data/semester/1/RS/project'))
    parser.add_argument('--data', type=Path, default=Path('/home/minseok/data/semester/1/RS/data'))
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise SubmissionError('Use a Slurm compute node for Kaggle API execution')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if args.action == 'check':
        existing = all_submissions(api)
        print(json.dumps({'count': len(existing), 'next_number': next_number(existing, {}),
                          'submissions': [describe(row) for row in existing]}, indent=2))
        return
    if args.manifest is None:
        raise SubmissionError('--manifest is required')
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest, args.data)
    (args.root / 'submissions').mkdir(parents=True, exist_ok=True)
    with (args.root / 'submissions/.kaggle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(api, args.root, manifest, args.action)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        print(f'Kaggle operation failed: {type(error).__name__}; HTTP status={status}', file=sys.stderr)
        if isinstance(error, SubmissionError):
            print(str(error), file=sys.stderr)
        raise SystemExit(1)
