"""Refit selected models on every provided training interaction; export the configured CSVs."""

import argparse

from datetime import datetime, timezone

import importlib.metadata

import json

import os

from pathlib import Path

import random

import socket

from time import perf_counter

import numpy as np

from scipy import sparse

from threadpoolctl import threadpool_limits

from .data import load_data, sha256

from .evaluation import topk, write_submission

from .models import EASE, SLIM, RP3Beta

from .io_utils import dump

def full_histories(data):
    histories = []
    for user in range(data.shape[0]):
        rows = np.flatnonzero(data.users == user)
        rows = rows[np.argsort(data.times[rows], kind='stable')]
        histories.append(data.items[rows])
    return histories

def fit_sasrec(data, x, params, seed, out):
    import torch
    from .sasrec import SASRec, training_arrays, inference_array, negative_pools, sample_negatives, predict
    if not torch.cuda.is_available():
        raise RuntimeError('SASRec refit requires an allocated GPU')
    device = torch.device('cuda')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    histories = full_histories(data)
    if sum(map(len, histories)) != len(data.users):
        raise AssertionError('Full histories must include all training rows')
    objective = params.get('objective', 'bce')
    if objective == 'ce':
        from .sequence_training import make_windows, full_catalog_loss
        if params['order'] != 'chronological' or params['window'] != 'coverage':
            raise ValueError('CE refitting requires chronological coverage windows')
        windows = make_windows(histories, params['maxlen'], params['window'],
                               params['horizon'], params['stride'])
        inputs, positive = windows.tokens, windows.targets
        if int((positive[..., 0] > 0).sum()) != sum(len(h) - 1 for h in histories):
            raise AssertionError('Coverage windows must supervise every training transition')
        active_tokens = torch.as_tensor(np.flatnonzero(x.any(axis=0)) + 1, device=device)
    elif objective == 'bce':
        pools = negative_pools(histories, np.flatnonzero(x.any(axis=0)))
        inputs, positive = training_arrays(histories, params['maxlen'])
    else:
        raise ValueError('Unknown SASRec objective')
    infer = inference_array(histories, params['maxlen'])
    arch = {key: params[key] for key in ('dim', 'maxlen', 'blocks', 'heads', 'dropout')}
    model = SASRec(data.shape[1], **arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.98),
                                 weight_decay=params.get('weight_decay', 0.0))
    history = []
    torch.cuda.synchronize()
    start = perf_counter()
    for epoch in range(1, params['epochs'] + 1):
        model.train()
        order = rng.permutation(len(inputs))
        losses, count = 0.0, 0
        for offset in range(0, len(order), params['batch_size']):
            users = order[offset:offset + params['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            if objective == 'ce':
                loss = full_catalog_loss(model, torch.as_tensor(inputs[users], device=device),
                                         torch.as_tensor(positive[users], device=device), active_tokens,
                                         params['temperature'], params['label_smoothing'])
            else:
                neg = sample_negatives(users, positive[users], pools, rng)
                tensors = [torch.as_tensor(arr, device=device)
                           for arr in (inputs[users], positive[users], neg)]
                loss = model.loss(*tensors)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            labels = positive[users, ..., 0] if objective == 'ce' else positive[users]
            positions = int((labels > 0).sum())
            losses += loss.item() * positions
            count += positions
        if epoch == 1 or epoch % 25 == 0 or epoch == params['epochs']:
            record = {'epoch': epoch, 'loss': losses / count}
            history.append(record)
            print(json.dumps({'model': 'sasrec', **record}), flush=True)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    checkpoint = {'state_dict': {key: val.detach().cpu() for key, val in model.state_dict().items()},
                  'architecture': arch, 'n_items': data.shape[1], 'item_ids': data.item_ids.tolist(),
                  'epochs': params['epochs'], 'seed': seed, 'fit_interactions': len(data.users),
                  'parameters': params}
    torch.save(checkpoint, out / 'sasrec_full.pt')
    dump(out / 'sasrec_training_history.json', history)
    scores = predict(model, infer, device)
    # Check the serialized model, not just the in-memory object, before publication.
    restored = torch.load(out / 'sasrec_full.pt', map_location=device, weights_only=True)
    model.load_state_dict(restored['state_dict'])
    np.testing.assert_array_equal(topk(predict(model, infer, device), x, x.any(axis=0)),
                                  topk(scores, x, x.any(axis=0)))
    return scores, {'fit_seconds': seconds, 'epochs': params['epochs'],
                    'training_positions': count, 'training_windows': len(inputs), 'objective': objective,
                    'gpu': torch.cuda.get_device_name(0), 'checkpoint_rankings_verified': True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='configs/submission.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run refitting on a Slurm compute node')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    if not config['models'] or not set(config['models']).issubset({'ease', 'slim', 'rp3beta', 'sasrec'}):
        raise ValueError('Configure at least one supported model')
    data = load_data(args.data)
    # The validation partition is used only for the earlier model selection.
    # Final fitting, masking, and negative pools use ALL rows from train.csv.
    x = data.matrix(np.arange(len(data.users)))
    if int(x.sum()) != len(data.users):
        raise AssertionError('Full interaction matrix lost training rows')
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(source_root)): sha256(path)
                     for folder in ('rs_project', 'configs', 'scripts', 'slurm', 'tests')
                     for path in sorted((source_root / folder).glob('**/*'))
                     if path.is_file() and '__pycache__' not in path.parts}
    manifest = {'run_id': out.name, 'status': 'running', 'smoke': args.smoke,
                'utc': datetime.now(timezone.utc).isoformat(), 'job_id': os.environ['SLURM_JOB_ID'],
                'hostname': socket.gethostname(), 'source_commit': os.getenv('RS_SOURCE_COMMIT', 'UNCOMMITTED'),
                'source_root': str(source_root), 'source_sha256': source_hashes,
                'data_hashes': data.hashes, 'fit_interactions': len(data.users),
                'users': data.shape[0], 'items': data.shape[1], 'test_users': len(data.test_ids),
                'config': config, 'submissions': {},
                'versions': {name: importlib.metadata.version(name)
                             for name in ('numpy', 'scipy', 'scikit-learn', 'torch')}}
    dump(out / 'manifest.json', manifest)
    with threadpool_limits(limits=1):
        for name, cls in (('ease', EASE), ('slim', SLIM), ('rp3beta', RP3Beta), ('sasrec', None)):
            if name not in config['models']:
                continue
            params = dict(config['models'][name])
            if args.smoke and name == 'sasrec':
                params['epochs'] = 2
            print(f'FULL REFIT {name}: {params}', flush=True)
            if cls is None:
                scores, diagnostics = fit_sasrec(data, x, params, config['seed'], out)
            else:
                start = perf_counter()
                model = cls(**params).fit(x)
                scores = model.predict(x)
                diagnostics = {'fit_seconds': perf_counter() - start, **model.diagnostics}
                if hasattr(model, 'weights'):
                    sparse.save_npz(out / f'{name}_full_weights.npz', model.weights)
                del model
            rankings = topk(scores, x, x.any(axis=0))
            path = out / f'{name}.csv'
            checks = write_submission(path, data, rankings, x)
            manifest['submissions'][name] = {'path': str(path), 'sha256': sha256(path),
                                            'parameters': params, 'diagnostics': diagnostics, 'checks': checks}
            dump(out / 'manifest.json', manifest)
            print(json.dumps({'model': name, **manifest['submissions'][name]}), flush=True)
            del scores
    manifest['status'] = 'complete'
    dump(out / 'manifest.json', manifest)
    print(f'COMPLETE {out / "manifest.json"}', flush=True)

if __name__ == '__main__':
    main()
