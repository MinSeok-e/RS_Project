"""Verify saved validation predictions and write a compact local comparison."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .data import load_data, sha256
from .evaluation import evaluate, topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', nargs='+', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    data = load_data(args.data)
    x, targets = data.matrix(data.train_rows), data.targets()
    candidates, verified_sources = [], {}
    experiments = [(Path(folder), ['ease', 'slim', 'rp3beta']) for folder in args.cpu]
    experiments.append((Path(args.gpu), ['sasrec']))
    for directory, names in experiments:
        if not (directory / 'complete.json').is_file():
            raise RuntimeError(f'Incomplete experiment: {directory}')
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['data']['split_sha256'] != data.summary()['split_sha256']:
            raise AssertionError('Experiments used different holdouts')
        if manifest['data']['file_sha256'] != data.hashes:
            raise AssertionError('Data changed since the experiment')
        source = Path(manifest['source_root'])
        for file, expected in manifest['source_sha256'].items():
            if sha256(source / file) != expected:
                raise AssertionError(f'Run source snapshot changed: {file}')
        verified_sources[str(directory)] = {'job_id': manifest['job_id'], 'source_root': str(source)}
        for name in names:
            best_path = directory / f'{name}_best.json'
            if not best_path.is_file():
                continue
            record = json.loads(best_path.read_text())
            with np.load(directory / f'{name}_best_predictions.npz') as saved:
                metrics, per_user = evaluate(saved['scores'], x, targets)
                np.testing.assert_array_equal(saved['rankings'], topk(saved['scores'], x, x.any(axis=0)))
                np.testing.assert_allclose(saved['ndcg'], [row['ndcg'] for row in per_user], atol=0, rtol=0)
            for key, value in record['metrics'].items():
                if abs(metrics[key] - value) > 1e-12:
                    raise AssertionError(f'Metric mismatch for {name}: {key}')
            candidates.append({'model': name, 'ndcg@20': metrics['ndcg@20'], 'recall@20': metrics['recall@20'],
                               'fit_seconds': record['fit_seconds'], 'evaluate_seconds': record['evaluate_seconds'],
                               'best_epoch': record.get('best_epoch'),
                               'parameters': record['parameters'], 'run_directory': str(directory)})
    best = {}
    for row in candidates:
        if row['model'] not in best or row['ndcg@20'] > best[row['model']]['ndcg@20']:
            best[row['model']] = row
    if set(best) != {'ease', 'slim', 'rp3beta', 'sasrec'}:
        raise AssertionError('All four models are required for the comparison')
    rows = sorted(best.values(), key=lambda row: -row['ndcg@20'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    selected_config = {"valid_fraction": 0.2, "seed": manifest["config"]["seed"]}
    selected_config.update({row["model"]: [row["parameters"]] for row in rows})
    (out / "best_config.json").write_text(json.dumps(selected_config, indent=2) + "\n")
    result = {'comparison': rows, 'run_bests': candidates, 'verified_sources': verified_sources,
              'validation': data.summary(), 'verification': 'source hashes, split/data hashes, saved ranks and metrics match',
              'limitations': 'one chronological holdout, one seed; limited hyperparameter search; not Kaggle scores'}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'parameters': json.dumps(row['parameters'], sort_keys=True)})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
