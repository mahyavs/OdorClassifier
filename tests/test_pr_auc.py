"""Synthetic tests for probability-based PR-AUC and saved evaluation outputs.

Run from the package root: python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from odor_classifier.config import Config
from odor_classifier.evaluation import compute_pr_auc, full_evaluation, full_evaluation_multilabel


class PRAUCTests(unittest.TestCase):
    def test_multiclass_matches_sklearn(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        scores = np.array([
            [.8, .1, .1], [.1, .7, .2], [.2, .1, .7],
            [.5, .3, .2], [.3, .5, .2], [.3, .4, .3],
        ])
        names = ["fruity", "floral", "woody"]
        result = compute_pr_auc(y, scores, names)
        indicator = np.eye(3)[y]
        for i, name in enumerate(names):
            self.assertAlmostEqual(result[name], average_precision_score(indicator[:, i], scores[:, i]))
        self.assertAlmostEqual(result["micro"], average_precision_score(indicator.ravel(), scores.ravel()))
        self.assertAlmostEqual(result["macro"], np.mean([result[name] for name in names]))

    def test_multilabel_matches_sklearn(self):
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])
        scores = np.array([
            [.8, .3, .9], [.1, .6, .3], [.6, .8, .1],
            [.2, .1, .7], [.7, .3, .3], [.3, .7, .6],
        ])
        result = compute_pr_auc(y, scores, ["fruity", "floral", "woody"], multilabel=True)
        self.assertAlmostEqual(result["micro"], average_precision_score(y.ravel(), scores.ravel()))
        self.assertAlmostEqual(result["macro"], average_precision_score(y, scores, average="macro"))

    def test_undefined_labels_are_skipped(self):
        y = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
        scores = np.array([[.9, .2, .5], [.3, .1, .4], [.8, .2, .4], [.2, .1, .3]])
        result = compute_pr_auc(y, scores, ["fruity", "floral", "woody"], multilabel=True)
        self.assertIsNone(result["floral"])
        self.assertIsNone(result["woody"])
        self.assertAlmostEqual(result["macro"], result["fruity"])

    def test_full_evaluation_reports(self):
        names = ["fruity", "floral", "woody"]
        y_multi = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])
        proba_multi = np.array([
            [.8, .3, .9], [.1, .6, .3], [.6, .8, .1],
            [.2, .1, .7], [.7, .3, .3], [.3, .7, .6],
        ])
        y_single = np.array([0, 1, 2, 0, 1, 2])
        proba_single = np.array([
            [.8, .1, .1], [.1, .7, .2], [.2, .1, .7],
            [.5, .3, .2], [.3, .5, .2], [.3, .4, .3],
        ])
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(reports_dir=Path(tmp) / "reports", plots_dir=Path(tmp) / "plots")
            config.reports_dir.mkdir(parents=True)
            config.plots_dir.mkdir(parents=True)
            for prefix, func, truth, probabilities, predictions in [
                ("test", full_evaluation_multilabel, y_multi, proba_multi, (proba_multi > .5).astype(int)),
                ("train", full_evaluation, y_single, proba_single, np.argmax(proba_single, axis=1)),
            ]:
                metrics = func(truth, predictions, probabilities, names, config, prefix)
                self.assertIn("pr_auc", metrics)
                self.assertIn("macro", metrics["pr_auc"])
                self.assertIn("micro", metrics["pr_auc"])
                saved = json.loads((config.reports_dir / f"{prefix}_metrics.json").read_text())
                self.assertEqual(saved["pr_auc"], metrics["pr_auc"])
                per_class = pd.read_csv(config.reports_dir / f"{prefix}_per_class_stats.csv")
                self.assertIn("pr_auc", per_class.columns)
                for filename in ("pr_curves.png", "pr_curves_per_class.png"):
                    self.assertTrue((config.plots_dir / f"{prefix}_{filename}").is_file())


if __name__ == "__main__":
    unittest.main()
