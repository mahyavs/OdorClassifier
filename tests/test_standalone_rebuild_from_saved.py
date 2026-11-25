"""Safety and functional tests of script using synthetic saved artifacts."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from odor_classifier.applicability_domain import ApplicabilityDomain
from odor_classifier.config import Config
from odor_classifier.experiment_setups import _raw_feature_hash
from odor_classifier.multilabel import SingleClassSafeClassifier

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / 'rebuild_heldout_from_saved.py'


class SavedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.outputs = pathlib.Path(self.folder.name) / 'outputs'
        self.setup_id = 'setup_test_001'
        self.setup = self.outputs / 'experiment_setups' / self.setup_id
        self.setup.mkdir(parents=True)
        (self.outputs / 'reports').mkdir(parents=True)
        (self.outputs / 'models').mkdir(parents=True)
        (self.outputs / 'cache').mkdir(parents=True)
        rng = np.random.default_rng(202)
        n = 38
        self.x = pd.DataFrame(rng.normal(size=(n, 5)), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        self.classes = ['fruity', 'floral']
        self.y = np.column_stack([
            (self.x['f1'] + self.x['f3'] > 0).astype(int),
            (self.x['f2'] + self.x['f4'] > 0).astype(int),
        ])
        self.smiles = ['C' * (i + 1) for i in range(n)]
        labels = pd.DataFrame({'No.': np.arange(1, n + 1), 'SMILES': self.smiles,
                               'label__fruity': self.y[:,0], 'label__floral': self.y[:,1]})
        labels.to_csv(self.outputs / 'reports' / 'training_ground_truth_binary.csv', index=False)
        joblib.dump(self.classes, self.outputs / 'models' / 'label_encoder.joblib')
        joblib.dump(np.array([.5, .5]), self.outputs / 'models' / 'multilabel_thresholds.joblib')
        idx_train_full = np.arange(28)
        idx_test = np.arange(28,n)
        pos_train = np.arange(21)
        pos_val = np.arange(21,28)
        split = ['train']*21 + ['validation']*7 + ['test']*10
        pd.DataFrame({'id': np.arange(1, n+1), 'SMILES': self.smiles, 'split': split}).to_csv(
            self.setup / 'data_split.csv', index=False)
        joblib.dump({'placeholder': True}, self.setup / 'feature_pipeline.joblib')
        (self.setup / 'indices.json').write_text(json.dumps({
            'idx_train_full':idx_train_full.tolist(), 'idx_test':idx_test.tolist(),
            'pos_train':pos_train.tolist(), 'pos_val':pos_val.tolist(),
        }))
        if importlib.util.find_spec('pyarrow') or importlib.util.find_spec('fastparquet'):
            self.x.to_parquet(self.outputs / 'cache' / 'raw_features.parquet')
        (self.setup / 'manifest.json').write_text(json.dumps({
            'id':self.setup_id, 'n_compounds':n,
            'raw_feature_sha256':_raw_feature_hash(self.x),
            'model_input_features':list(self.x.columns),
        }))
        # Prior production AD and model are fitted on all n, but will only
        # be read as a FEATURE SOURCE and unfitted hyperparameter template.
        ad = ApplicabilityDomain(config=Config(ad_k_neighbors=3))
        ad.fit(self.x)
        ad.training_smiles_sha256_ = hashlib.sha256('\n'.join(self.smiles).encode()).hexdigest()
        ad.save(str(self.outputs / 'models' / 'applicability_domain.joblib'))
        model = MultiOutputClassifier(SingleClassSafeClassifier(LogisticRegression(max_iter=100)))
        model.fit(self.x, self.y)
        joblib.dump(model, self.outputs / 'models' / 'production_model.joblib')

    def tearDown(self):
        self.folder.cleanup()

    def call(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--outputs-dir', str(self.outputs),
                               '--setup-id', self.setup_id, '--features-source', 'production-ad',
                               *extra], capture_output=True, text=True)

    def test_check_only(self):
        out = self.call('--check-only')
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('CHECK PASSED:', out.stdout)

    def test_rebuild_model_and_snapshot(self):
        out = self.call('--skip-plots', '--audit', 'none', '--ad-k', '3')
        self.assertEqual(out.returncode, 0, out.stderr)
        report_dir = self.outputs / 'standalone_heldout_rebuild' / 'reports'
        pred = pd.read_csv(report_dir / 'heldout_label_audit_input' / 'heldout_test_predictions.csv')
        ref = pd.read_csv(report_dir / 'heldout_label_audit_input' / 'training_only_reference.csv')
        self.assertEqual(len(pred), 10)
        self.assertEqual(len(ref), 28)
        self.assertTrue((self.outputs / 'standalone_heldout_rebuild' / 'models' / 'heldout_model.joblib').is_file())

    def test_rebuild_with_both_audits(self):
        out = self.call('--skip-plots', '--audit', 'both', '--ad-k', '3')
        self.assertEqual(out.returncode, 0, out.stderr)
        root = self.outputs / 'standalone_heldout_rebuild'
        self.assertFalse((root / 'AUDIT_ERROR.txt').exists(), out.stderr)
        self.assertTrue((root / 'reports' / 'heldout_label_audit' / 'morgan' /
                         'reference_agreement_summary.csv').exists())
        self.assertTrue((root / 'reports' / 'heldout_label_audit' / 'descriptor' /
                         'reference_agreement_summary.csv').exists())

    def test_bad_input_checksum_rejected(self):
        ad_path = self.outputs / 'models' / 'applicability_domain.joblib'
        ad = joblib.load(ad_path)
        ad.training_smiles_sha256_ = 'invalid'
        joblib.dump(ad, ad_path)
        out = self.call('--check-only')
        self.assertNotEqual(out.returncode, 0)
        self.assertIn('checksum', out.stderr)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow') or importlib.util.find_spec('fastparquet'),
                         'Requires parquet engine')
    def test_raw_cache_hash_mismatch_rejected(self):
        x = self.x.copy()
        x.iloc[0,0] += .00001
        x.to_parquet(self.outputs / 'cache' / 'raw_features.parquet')
        out = self.call('--check-only', '--features-source', 'cache')
        self.assertNotEqual(out.returncode, 0)
        self.assertIn('SHA-256 mismatch', out.stderr)


if __name__ == '__main__':
    unittest.main()
