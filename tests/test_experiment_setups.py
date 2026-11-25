"""Tests for exact split/feature-pipeline archival and mismatch rejection."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odor_classifier.config import Config
from odor_classifier.experiment_setups import (
    list_experiment_setups, load_experiment_setup, save_experiment_setup,
)


class ExampleFittedPipeline:
    """Simple pickleable stand-in for the real fitted FeaturePipeline."""
    selected_features_ = ["rdkit_a", "rdkit_c"]
    kept_after_variance_ = ["rdkit_a", "rdkit_b", "rdkit_c"]
    pca_ = None

    def transform(self, raw):
        return raw[self.selected_features_].copy()


def sample(tmp_path):
    cfg = Config(outputs_dir=tmp_path)
    compounds = pd.DataFrame({"No.": [1, 2, 3, 4, 5, 6, 7, 8],
                              "SMILES": ["CC", "CO", "CCC", "N", "C=C", "O", "CN", "CCO"]})
    labels = pd.Series(["sweet", "fruity", "sweet", "floral", "floral", "fruity", "sweet", "fruity"])
    Y = pd.DataFrame({"sweet": [1, 0, 1, 0, 0, 0, 1, 0],
                      "fruity": [0, 1, 0, 0, 0, 1, 0, 1]})
    raw = pd.DataFrame({"rdkit_a": range(8), "rdkit_b": range(8, 16),
                        "rdkit_c": np.arange(8) * 1.5})
    index = ([7, 0, 1, 2, 4, 5], [6, 3], [2, 1, 0, 5], [4, 3])
    return cfg, compounds, labels, Y, raw, index


def test_archived_setup_reuses_exact_row_order_and_features(tmp_path):
    cfg, compounds, labels, Y, raw, indices = sample(tmp_path)
    manifest = save_experiment_setup(cfg, compounds, labels, Y, raw, ExampleFittedPipeline(), *indices)
    saved = list_experiment_setups(cfg)
    assert [x["id"] for x in saved] == [manifest["id"]]
    new_cfg = replace(cfg, selected_model="svm", n_optuna_trials=150,
                      selection_method="rfecv", rfecv_step=0.05)
    loaded_indices, fp, loaded_meta = load_experiment_setup(
        new_cfg, manifest["id"], compounds, labels, Y, raw)
    assert [x.tolist() for x in loaded_indices] == [list(x) for x in indices]
    assert fp.selected_features_ == ["rdkit_a", "rdkit_c"]
    assert fp.transform(raw).columns.tolist() == ["rdkit_a", "rdkit_c"]
    assert loaded_meta["split_counts"] == {"train": 4, "validation": 2, "test": 2}


def test_reject_changed_dataset_or_targets(tmp_path):
    cfg, compounds, labels, Y, raw, indices = sample(tmp_path)
    manifest = save_experiment_setup(cfg, compounds, labels, Y, raw, ExampleFittedPipeline(), *indices)
    with pytest.raises(ValueError, match="different compounds"):
        altered = compounds.copy()
        altered.loc[2, "SMILES"] = "CCl"
        load_experiment_setup(cfg, manifest["id"], altered, labels, Y, raw)
    with pytest.raises(ValueError, match="different compounds"):
        altered_y = Y.copy()
        altered_y.loc[0, "sweet"] = 0
        load_experiment_setup(cfg, manifest["id"], compounds, labels, altered_y, raw)


def test_reject_descriptor_changes_or_numeric_values(tmp_path):
    cfg, compounds, labels, Y, raw, indices = sample(tmp_path)
    manifest = save_experiment_setup(cfg, compounds, labels, Y, raw, ExampleFittedPipeline(), *indices)
    with pytest.raises(ValueError, match="Raw descriptor settings"):
        load_experiment_setup(replace(cfg, morgan_n_bits=1024),
                              manifest["id"], compounds, labels, Y, raw)
    with pytest.raises(ValueError, match="raw molecular descriptor values"):
        altered = raw.copy()
        altered.loc[1, "rdkit_c"] += 0.1
        load_experiment_setup(cfg, manifest["id"], compounds, labels, Y, altered)


def test_reject_tampered_split_indices(tmp_path):
    cfg, compounds, labels, Y, raw, indices = sample(tmp_path)
    manifest = save_experiment_setup(cfg, compounds, labels, Y, raw, ExampleFittedPipeline(), *indices)
    import json
    file = tmp_path / "experiment_setups" / manifest["id"] / "indices.json"
    data = json.loads(file.read_text())
    data["idx_test"][0] = data["idx_train_full"][0]
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid saved experiment split"):
        load_experiment_setup(cfg, manifest["id"], compounds, labels, Y, raw)


def test_comparison_history_preserves_scores_when_classifier_changes(tmp_path):
    from odor_classifier.experiment_setups import record_classifier_result
    cfg, compounds, labels, Y, raw, indices = sample(tmp_path)
    manifest = save_experiment_setup(cfg, compounds, labels, Y, raw, ExampleFittedPipeline(), *indices)
    score = {"macro_f1": 0.7, "pr_auc": {"macro": 0.83, "micro": 0.9},
             "roc_auc": {"macro": 0.91}, "subset_accuracy": 0.42}
    record_classifier_result(replace(cfg, selected_model="svm"), manifest["id"], "svm", score, score)
    score2 = {**score, "macro_f1": 0.73}
    path = record_classifier_result(replace(cfg, selected_model="random_forest"),
                                    manifest["id"], "random_forest", score2, score2)
    result = pd.read_csv(path)
    assert result["classifier"].tolist() == ["svm", "random_forest"]
    assert result["test_macro_f1"].tolist() == [0.7, 0.73]
    assert result["test_macro_pr_auc"].tolist() == [0.83, 0.83]


def test_launcher_recovers_archived_setup_id_and_seed(tmp_path):
    import importlib.util
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("training_launcher", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args([
        "--selection-method", "rfecv", "--selected-model", "svm",
        "--reuse-experiment-setup", "20261104T153000_12ab34cd",
        "--random-state", "123", "--no-optuna",
    ])
    cfg = module.make_config(args)
    assert cfg.selection_method == "rfecv"
    assert cfg.selected_model == "svm"
    assert cfg.reuse_experiment_setup == "20261104T153000_12ab34cd"
    assert cfg.random_state == 123
    assert cfg.multilabel and not cfg.use_optuna
