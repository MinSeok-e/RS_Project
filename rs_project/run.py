"""Run validation experiments; never submit to Kaggle or commit automatically."""
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
from .evaluation import evaluate, topk
from .models import EASE, SLIM, RP3Beta


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_predictions(path, scores, x, targets):
    metrics, per_user = evaluate(scores, x, targets)
    np.savez_compressed(path, scores=scores, rankings=topk(scores, x, x.any(axis=0)),
                        ndcg=np.array([p['ndcg'] for p in per_user]))
    return metrics


def provenance(out, data, args, config):
    root = Path(__file__).resolve().parents[1]
    source = {}
    for folder in ('rs_project', 'configs', 'tests', 'slurm'):
        for path in sorted((root / folder).glob('**/*')):
            if path.is_file() and '__pycache__' not in path.parts:
                source[str(path.relative_to(root))] = sha256(path)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
              'job_id': os.getenv('SLURM_JOB_ID'), 'source_root': str(root),
              'source_sha256': source, 'args': vars(args), 'config': config,
              'data': data.summary(),
              'versions': {name: importlib.metadata.version(name)
                           for name in ('numpy', 'scipy', 'scikit-learn', 'torch')},
              'protocol': {'feedback': 'all observed ratings = 1',
                           'split': 'per-user chronological; last ceil(20%) for validation',
                           'candidate_items': 'fit-active items minus personal fit history',
                           'metric': 'macro binary full-catalog NDCG@20; cold targets included',
                           'sequential_validation': 'one recommendation from fit history; no validation teacher forcing'},
              'git_commit_policy': 'uncommitted source snapshot; commit model code at actual submission'}
    dump(out / 'manifest.json', result)


def run_cpu(data, x, targets, config, out, smoke=False):
    results = []
    classes = {'ease': EASE, 'slim': SLIM, 'rp3beta': RP3Beta}
    for name, cls in classes.items():
        best = -1
        candidates = config[name]
        if smoke:
            candidates = candidates[:1] if name != 'ease' else [{'regularization': 300}]
            if name == 'slim':
                continue  # Small SLIM fits and convergence are covered by the unit tests.
        for idx, params in enumerate(candidates):
            print(f'START {name} {idx}: {params}', flush=True)
            start = perf_counter()
            model = cls(**params).fit(x)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            scores = model.predict(x)
            metrics, _ = evaluate(scores, x, targets)
            record = {'model': name, 'trial': idx, 'parameters': params, 'metrics': metrics,
                      'fit_seconds': fit_seconds, 'evaluate_seconds': perf_counter() - start,
                      'diagnostics': model.diagnostics}
            results.append(record)
            dump(out / 'trials.json', results)
            print(json.dumps(record), flush=True)
            if smoke and name == 'ease':
                if abs(metrics['ndcg@20'] - 0.15500816004052462) > 1e-12:
                    raise AssertionError('EASE did not reproduce the HW2 split/metric reference')
            if metrics['ndcg@20'] > best:
                best = metrics['ndcg@20']
                save_predictions(out / f'{name}_best_predictions.npz', scores, x, targets)
                if hasattr(model, 'weights'):
                    sparse.save_npz(out / f'{name}_best_weights.npz', model.weights)
                dump(out / f'{name}_best.json', record)
    return results


def run_gpu(data, x, targets, config, out, smoke=False):
    import torch
    from .sasrec import SASRec, training_arrays, inference_array, negative_pools, sample_negatives, predict
    if not torch.cuda.is_available():
        raise RuntimeError('GPU job requires CUDA')
    device = torch.device('cuda')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    histories = [data.items[data.train_rows[data.users[data.train_rows] == u]] for u in range(data.shape[0])]
    pools = negative_pools(histories, np.flatnonzero(x.any(axis=0)))
    candidates = config['sasrec'][:1] if smoke else config['sasrec']
    results, global_best = [], -1
    for idx, original in enumerate(candidates):
        params = dict(original)
        if smoke:
            params.update(epochs=2, eval_every=1, patience=2)
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        arch = {key: params[key] for key in ('dim', 'maxlen', 'blocks', 'heads', 'dropout')}
        model = SASRec(data.shape[1], **arch).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.98))
        inputs, positive = training_arrays(histories, params['maxlen'])
        infer = inference_array(histories, params['maxlen'])
        best, best_epoch, history, best_state = -1, 0, [], None
        training_seconds, evaluation_seconds = 0.0, 0.0
        print(f'START sasrec {idx}: {params}', flush=True)
        for epoch in range(1, params['epochs'] + 1):
            model.train()
            torch.cuda.synchronize()
            start = perf_counter()
            order = rng.permutation(len(histories))
            losses, count = 0.0, 0
            for offset in range(0, len(order), params['batch_size']):
                users = order[offset:offset + params['batch_size']]
                neg = sample_negatives(users, positive[users], pools, rng)
                tensors = [torch.as_tensor(arr, device=device) for arr in (inputs[users], positive[users], neg)]
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(*tensors)
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                positions = int((positive[users] > 0).sum())
                losses += loss.item() * positions
                count += positions
            torch.cuda.synchronize()
            training_seconds += perf_counter() - start
            if epoch % params['eval_every'] and epoch != params['epochs']:
                continue
            start = perf_counter()
            scores = predict(model, infer, device)
            metrics, _ = evaluate(scores, x, targets)
            evaluation_seconds += perf_counter() - start
            entry = {'epoch': epoch, 'loss': losses / count, **metrics}
            history.append(entry)
            print(json.dumps({'model': 'sasrec', 'trial': idx, **entry}), flush=True)
            if metrics['ndcg@20'] > best:
                best, best_epoch = metrics['ndcg@20'], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            dump(out / f'sasrec_trial{idx}_history.json', history)
            if epoch - best_epoch >= params['patience']:
                break
        model.load_state_dict(best_state)
        scores = predict(model, infer, device)
        metrics, _ = evaluate(scores, x, targets)
        if abs(metrics['ndcg@20'] - best) > 1e-12:
            raise AssertionError('Reloaded checkpoint did not reproduce validation score')
        checkpoint = {'state_dict': best_state, 'architecture': arch, 'n_items': data.shape[1],
                      'item_ids': data.item_ids.tolist(), 'best_epoch': best_epoch, 'seed': seed}
        torch.save(checkpoint, out / f'sasrec_trial{idx}.pt')
        record = {'model': 'sasrec', 'trial': idx, 'parameters': params, 'seed': seed,
                  'metrics': metrics, 'fit_seconds': training_seconds, 'evaluate_seconds': evaluation_seconds,
                  'best_epoch': best_epoch, 'stopped_epoch': epoch,
                  'diagnostics': {'gpu': torch.cuda.get_device_name(0), 'objective': 'sampled BCE, 1 negative per positive',
                                  'training_positions': int((positive > 0).sum()),
                                  'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}}
        results.append(record)
        dump(out / 'trials.json', results)
        print('FINISH ' + json.dumps(record), flush=True)
        if best > global_best:
            global_best = best
            save_predictions(out / 'sasrec_best_predictions.npz', scores, x, targets)
            torch.save(checkpoint, out / 'sasrec_best.pt')
            dump(out / 'sasrec_best.json', record)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['cpu', 'gpu', 'smoke'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='configs/best.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    data = load_data(args.data, config['valid_fraction'])
    x, targets = data.matrix(data.train_rows), data.targets()
    provenance(out, data, args, config)
    print(json.dumps(data.summary()), flush=True)
    with threadpool_limits(limits=1):
        if args.suite in ('cpu', 'smoke'):
            run_cpu(data, x, targets, config, out, args.suite == 'smoke')
        if args.suite in ('gpu', 'smoke'):
            run_gpu(data, x, targets, config, out, args.suite == 'smoke')
    dump(out / 'complete.json', {'suite': args.suite, 'completed_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
