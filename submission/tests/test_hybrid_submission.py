import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from tests.test_submission import FakeAPI
from scripts.kaggle_project import SubmissionError,digest,filename,publish,selected_models
from rs_project.submission_graph import recover_steps


class HybridSubmissionTests(unittest.TestCase):
    def test_single_ensemble_uses_user_naming_and_never_uploads_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'submissions').mkdir();path=root/'ensemble.csv';path.write_text('fake offline fixture')
            api=FakeAPI();manifest={'run_id':'ensemble','source_commit':'a'*40,'submissions':{'hybrid_ensemble':{'path':str(path),'sha256':digest(path),'parameters':{}}}}
            with contextlib.redirect_stdout(io.StringIO()):
                publish(api,root,manifest,'submit');publish(api,root,manifest,'submit')
            self.assertEqual(len(api.calls),1)
            self.assertEqual(Path(api.calls[0][0]).name,'MinseokJeong_v3_HybridEnsemble_Rerank.csv')
            self.assertEqual(api.calls[0][1],'MinseokJeong_v3_HybridEnsemble_Rerank')
            self.assertEqual(selected_models(manifest),['hybrid_ensemble'])
            ledger_path=root/'submissions/kaggle_records.json';ledger=json.loads(ledger_path.read_text())
            ledger['ensemble:hybrid_ensemble'].update(submission_id=-1,file_name='not_visible.csv')
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaises(SubmissionError),contextlib.redirect_stdout(io.StringIO()):publish(api,root,manifest,'submit')
            self.assertEqual(len(api.calls),1)

    def test_serving_schemas_do_not_mutate_the_parent_cache(self):
        import numpy as np
        from rs_project.submission_features import FeatureBuilder
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"features.npz";np.savez(path,features=np.ones((2,1)))
            builder=FeatureBuilder.__new__(FeatureBuilder)
            builder.records={"1":{"feature_names":["base"]}}
            builder.build=lambda job:path
            _,names=builder.read(1);names.append("additional")
            self.assertEqual(builder.records["1"]["feature_names"],["base"])

    def test_repeated_component_preserves_selected_mixing_order(self):
        weights=[{'a':1.},{'a':.75,'b':.25},{'a':.775,'b':.225}]
        rows=[{'trial':i,'parameters':{'weights':w}} for i,w in enumerate(weights)]
        node={'run':'fixture','steps':rows,'row':rows[-1]}
        self.assertEqual(recover_steps(node),('a',[('b',.25),('a',.1)]))
        node['row']={'parameters':{'weights':{'a':.8,'b':.2}}}
        with self.assertRaises(ValueError):recover_steps(node)


if __name__=='__main__':unittest.main()
