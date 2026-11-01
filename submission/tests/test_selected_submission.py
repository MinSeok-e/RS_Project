import contextlib
from datetime import datetime,timezone
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from tests.test_submission import FakeAPI
from scripts.kaggle_project import SubmissionError,digest,filename,publish

class SelectedDailySubmissionTests(unittest.TestCase):
    def test_submission_keyword_is_explicit_and_safe(self):
        self.assertEqual(filename(9,'hybrid_ensemble',{'keyword':'MoE16_CE_Frozen'}),'MinseokJeong_v9_MoE16_CE_Frozen.csv')
        with self.assertRaises(SubmissionError):filename(9,'hybrid_ensemble',{'keyword':'../invalid'})

    def test_daily_guard_blocks_upload_and_expired_authorization(self):
        for day,remaining in [(str(datetime.now(timezone.utc).date()),0),('2000-01-01',9)]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'submissions').mkdir();path=root/'scores.csv';path.write_text('fixture')
                api=FakeAPI();api.competition_get_submission_limits=lambda _:SimpleNamespace(num_allowed_now=remaining)
                m={'run_id':'fixed','source_commit':'a'*40,'quota_utc_date':day,'submissions':{'hybrid_ensemble':{'path':str(path),'sha256':digest(path),'parameters':{'keyword':'MoE16_CE_Frozen'}}}}
                with self.assertRaises(SubmissionError),contextlib.redirect_stdout(io.StringIO()):publish(api,root,m,'submit')
                self.assertEqual(api.calls,[])

if __name__=='__main__':unittest.main()
