"""End-to-end smoke test of Single model without Optuna (using synthetic descriptors).

The chemical-space calculation and descriptor generation are mocked so the
training/evaluation/split logic runs without the user's private compounds.
"""
from pathlib import Path
from importlib.util import find_spec
import sys
import types
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
import pandas as pd
from odor_classifier.config import Config
# The optional mol2vec package is imported eagerly by the existing feature
# module, even though this smoke test mocks molecular descriptor generation.
if find_spec("mol2vec") is None:
    fake_mol2vec = types.ModuleType("mol2vec")
    fake_mol2vec.__path__ = []
    fake_features = types.ModuleType("mol2vec.features")
    fake_features.mol2alt_sentence = lambda mol, radius: []
    sys.modules["mol2vec"] = fake_mol2vec
    sys.modules["mol2vec.features"] = fake_features
from odor_classifier.pipeline import run_full_pipeline


def test_multilabel_validation_reuses_threshold_calibration_predictions(tmp_path):
    """The new validation report must not alter per-label calibration."""
    from unittest.mock import patch
    from odor_classifier.pipeline import run_full_pipeline
    n = 60
    rng = np.random.default_rng(19)
    names = ["floral", "fruity", "woody"]
    compounds = pd.DataFrame({
        "No.": np.arange(n), "Name": [f"mol{i}" for i in range(n)],
        "SMILES": ["CC"] * n,
        "odor1": [names[i % 3] for i in range(n)],
        "odor2": [names[(i + 1) % 3] if i % 2 == 0 else None for i in range(n)],
        "class1": [names[i % 3] for i in range(n)],
    })
    dictionary = pd.DataFrame({"class": names, "type1": names})
    raw = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"rdkit_{i}" for i in range(6)])
    cfg = Config(
        outputs_dir=tmp_path / "out", reports_dir=tmp_path / "out" / "reports",
        plots_dir=tmp_path / "out" / "plots", models_dir=tmp_path / "out" / "models",
        selected_model="naive_bayes", use_optuna=False, cv_folds=0,
        selection_method="none", save_experiment_setup=False,
        calibrate_multilabel_thresholds=True, resume_from_checkpoint=False,
        ad_k_neighbors=3,
    )
    for path in [cfg.outputs_dir, cfg.reports_dir, cfg.plots_dir, cfg.models_dir]:
        path.mkdir(parents=True, exist_ok=True)
    with (patch('odor_classifier.pipeline.load_compounds', return_value=compounds),
          patch('odor_classifier.pipeline.load_class_dictionary', return_value=dictionary),
          patch('odor_classifier.io_utils.load_class_dictionary', return_value=dictionary),
          patch('odor_classifier.pipeline.get_or_compute_raw_features', return_value=raw),
          patch('odor_classifier.pipeline.generate_chemical_space_plots')):
        results = run_full_pipeline(cfg)
    val = json.loads((cfg.reports_dir / 'validation_metrics.json').read_text())
    meta = json.loads((cfg.reports_dir / 'validation_evaluation_info.json').read_text())
    self_thresh = json.loads((cfg.reports_dir / 'multilabel_thresholds.json').read_text())
    assert results['validation_stats']['macro_f1'] == val['macro_f1']
    assert meta['thresholds_optimized_on_validation'] is True
    assert len(self_thresh) == 3
    assert (cfg.reports_dir / 'test_metrics.json').exists()
