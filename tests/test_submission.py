import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from rs_project.refit import full_histories
from scripts.kaggle_project import (MODELS, SubmissionError, digest, filename, next_number,
                                    publish, verify_csv)


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.rows = [self.row(1, 'old_ease.csv'), self.row(2, 'old_neumf.csv')]

    def row(self, ref, name):
        return SimpleNamespace(ref=ref, file_name=name, description=Path(name).stem,
                               date='2026-10-31', status=SimpleNamespace(name='COMPLETE'),
                               public_score='0.1', error_description='', team_name='fake')

    def competition_submissions(self, competition, page_number=1, page_size=100):
        return self.rows[(page_number-1)*page_size:page_number*page_size]

    def competition_submit(self, path, description, competition, quiet=True):
        self.calls.append((path, description))
        row = self.row(len(self.rows) + 1, Path(path).name)
        self.rows.append(row)
        return SimpleNamespace(ref=row.ref)


class SubmissionTests(unittest.TestCase):
    def test_full_histories_include_every_row_in_time_order(self):
        data = SimpleNamespace(shape=(2, 4), users=np.array([1, 0, 0, 1]),
                               items=np.array([3, 1, 0, 2]), times=np.array([1, 1, 0, 0]))
        histories = full_histories(data)
        self.assertEqual([h.tolist() for h in histories], [[0, 1], [2, 3]])

    def test_versioning_follows_count_and_existing_versions(self):
        api = FakeAPI()
        self.assertEqual(next_number(api.rows, {}), 3)
        self.assertEqual(next_number(api.rows, {'x': {'file_name': 'MinseokJeong_v8_EASE.csv'}}), 9)
        self.assertEqual(filename(6, 'sasrec', {'dim':64, 'maxlen':100, 'epochs':265}),
                         'MinseokJeong_v6_SASRec_d64_len100_ep265.csv')

    def test_uploads_four_models_once_even_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'submissions').mkdir()
            params = json.loads(Path('configs/submission.json').read_text())['models']
            manifest = {'run_id': 'test', 'source_commit': 'a'*40, 'submissions': {}}
            for model in MODELS:
                path = root / f'{model}.csv'
                path.write_text('dummy fixture: no real upload')
                manifest['submissions'][model] = {'path': str(path), 'sha256': digest(path),
                                                  'parameters': params[model]}
            api = FakeAPI()
            with contextlib.redirect_stdout(io.StringIO()):
                publish(api, root, manifest, 'submit')
                publish(api, root, manifest, 'submit')
            self.assertEqual(len(api.calls), 4)
            self.assertEqual([Path(path).name.split('_')[1] for path, _ in api.calls], ['v3','v4','v5','v6'])
            for path, description in api.calls:
                self.assertEqual(description, Path(path).stem)
            ledger = json.loads((root / 'submissions/kaggle_records.json').read_text())
            self.assertTrue(all(row['status'] == 'COMPLETE' for row in ledger.values()))
            # A recorded attempt that is absent remotely must never auto-upload again.
            ledger['test:ease']['file_name'] = 'missing.csv'
            ledger['test:ease']['submission_id'] = -1
            (root / 'submissions/kaggle_records.json').write_text(json.dumps(ledger))
            with self.assertRaises(SubmissionError), contextlib.redirect_stdout(io.StringIO()):
                publish(api, root, manifest, 'submit')
            self.assertEqual(len(api.calls), 4)

    def test_csv_uses_test_id_and_user_history_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'train.csv').write_text('userId,movieId,rating,time_idx\n99,0,1,0\n' +
                ''.join(f'10,{i},1,{i}\n' for i in range(1,21)))
            (root / 'test.csv').write_text('id,userId\n0,99\n')
            path = root / 'pred.csv'
            path.write_text('id,recommendations\n0,' + ' '.join(map(str, range(1,21))) + '\n')
            verify_csv(path, root, digest(path))
            original_hash = digest(path)
            path.write_text('id,recommendations\n0,' + ' '.join(map(str, range(20))) + '\n')
            with self.assertRaises(SubmissionError):
                verify_csv(path, root, original_hash)
            with self.assertRaises(SubmissionError):
                verify_csv(path, root, digest(path))


if __name__ == '__main__':
    unittest.main()
