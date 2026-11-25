"""CV-fold reporting is independent of hyperparameter search in Single model mode."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

# Previous package imports an optional mol2vec library at package import time.
import sys
import types
from importlib.machinery import ModuleSpec
if "mol2vec" not in sys.modules:
    try:
        import mol2vec  # noqa: F401
    except ImportError:
        dummy = types.ModuleType("mol2vec")
        dummy.__path__ = []
        dummy.__spec__ = ModuleSpec("mol2vec", loader=None, is_package=True)
        features = types.ModuleType("mol2vec.features")
        features.__spec__ = ModuleSpec("mol2vec.features", loader=None)
        features.mol2alt_sentence = lambda m, r: []
        sys.modules["mol2vec"] = dummy
        sys.modules["mol2vec.features"] = features

from odor_classifier.config import Config
from odor_classifier.pipeline import run_full_pipeline
from odor_classifier.cv_reporting import evaluate_single_model_cv
from odor_classifier.multilabel import wrap_models_for_multilabel


def _run(tmp_path, *, folds, multilabel=True, optuna=False, save_setup=False, selected_model="naive_bayes"):
    rng = np.random.default_rng(45)
    names = ["floral", "fruity", "woody"]
    n = 72
    df = pd.DataFrame({
        "No.": np.arange(n), "Name": [f"m{i}" for i in range(n)],
        "SMILES": ["CC"] * n,
        "odor1": [names[i % 3] for i in range(n)],
        "odor2": [names[(i+1) % 3] if i % 2 == 0 else None for i in range(n)],
        "class1": [names[i % 3] for i in range(n)],
    })
    dic = pd.DataFrame({"class": names, "type1": names})
    desc = pd.DataFrame(rng.normal(size=(n, 7)), columns=[f"rdkit_{i}" for i in range(7)])
    config = Config(
        outputs_dir=tmp_path / "out", reports_dir=tmp_path / "out/reports",
        plots_dir=tmp_path / "out/plots", models_dir=tmp_path / "out/models",
        selected_model=selected_model, use_optuna=optuna, cv_folds=folds,
        selection_method="none", save_experiment_setup=save_setup,
        calibrate_multilabel_thresholds=multilabel, resume_from_checkpoint=False,
        ad_k_neighbors=3,
    )
    for path in (config.outputs_dir, config.reports_dir, config.plots_dir, config.models_dir):
        path.mkdir(parents=True, exist_ok=True)
    with (patch("odor_classifier.pipeline.build_candidate_models", return_value={"naive_bayes": GaussianNB()}),
          patch("odor_classifier.pipeline.load_compounds", return_value=df),
          patch("odor_classifier.pipeline.load_class_dictionary", return_value=dic),
          patch("odor_classifier.io_utils.load_class_dictionary", return_value=dic),
          patch("odor_classifier.pipeline.get_or_compute_raw_features", return_value=desc),
          patch("odor_classifier.pipeline.generate_chemical_space_plots")):
        if optuna:
            with patch("odor_classifier.pipeline.tune_best_model", return_value=(
                    wrap_models_for_multilabel({"nb": GaussianNB()}, config)["nb"], SimpleNamespace(best_value=0.7, best_params={}))):
                result = run_full_pipeline(config)
        else:
            result = run_full_pipeline(config)
    return config, result


@pytest.mark.parametrize("folds", [0, 1])
def test_without_optuna_zero_or_one_fold_remains_validation_only(tmp_path, folds):
    cfg, result = _run(tmp_path, folds=folds)
    assert result["validation_stats"] is not None
    assert result["single_model_cv_stats"] is None
    assert not (cfg.reports_dir / "single_model_cv_metrics.json").exists()
    assert (cfg.reports_dir / "test_metrics.json").exists()


@pytest.mark.parametrize("multilabel", [True])
def test_without_optuna_three_folds_produces_cv_on_train_plus_validation_and_test(tmp_path, multilabel):
    cfg, result = _run(tmp_path, folds=3, multilabel=multilabel)
    cv = json.loads((cfg.reports_dir / "single_model_cv_metrics.json").read_text())
    folds = pd.read_csv(cfg.reports_dir / "single_model_cv_folds.csv")
    assert cv["cv_folds"] == 3
    assert len(folds) == 3
    assert result["single_model_cv_stats"]["mean"] == cv["mean"]
    assert cv["optuna_used"] is False
    assert cv["evaluated_on"] == "train_plus_validation"
    split = pd.read_csv(cfg.reports_dir / "data_split.csv")
    n_non_test = int((split["split"] != "test").sum())
    assert cv["cv_total_samples"] == n_non_test
    assert cv["cv_total_scored"] == n_non_test
    assert folds["n_scored"].sum() == n_non_test
    assert folds["n_fit"].add(folds["n_scored"]).eq(n_non_test).all()
    assert cv["fold_thresholds"] == ("fixed_global_threshold" if multilabel else "argmax")
    for metric in ("macro_f1", "micro_f1", "macro_pr_auc", "micro_pr_auc", "macro_roc_auc"):
        assert metric in folds.columns
        assert np.isclose(folds[metric].mean(), cv["mean"][metric], equal_nan=True)
        assert np.isclose(folds[metric].std(), cv["std"][metric], equal_nan=True)
    assert result["validation_stats"] is None
    assert not (cfg.reports_dir / "validation_metrics.json").exists()
    assert (cfg.reports_dir / "test_metrics.json").exists()
    if multilabel:
        # Thresholds are still calibrated on the internal validation partition
        # even though only CV scores are displayed to the user.
        assert (cfg.reports_dir / "multilabel_thresholds.json").exists()


def test_optuna_enabled_produces_independent_cv_reporting_without_changing_tuning(tmp_path):
    cfg, result = _run(tmp_path, folds=3, optuna=True)
    cv = result["single_model_cv_stats"]
    assert cv is not None
    assert cv["optuna_used"] is True
    assert cv["evaluated_on"] == "train_plus_validation"
    split = pd.read_csv(cfg.reports_dir / "data_split.csv")
    assert cv["cv_total_samples"] == (split["split"] != "test").sum()
    assert (cfg.reports_dir / "single_model_cv_metrics.json").exists()
    assert result["validation_stats"] is None
    assert (cfg.reports_dir / "test_metrics.json").exists()


def test_optuna_enabled_cv_zero_reports_validation_and_test(tmp_path):
    cfg, result = _run(tmp_path, folds=0, optuna=True)
    assert result["validation_stats"] is not None
    assert result["single_model_cv_stats"] is None
    assert not (cfg.reports_dir / "single_model_cv_metrics.json").exists()
    info = json.loads((cfg.reports_dir / "validation_evaluation_info.json").read_text())
    assert info["hyperparameters_selected_on_validation"] is True
    assert info["optuna_used"] is True


def test_cv_scores_are_saved_to_comparison_history(tmp_path):
    cfg, result = _run(tmp_path, folds=3, save_setup=True)
    setup_id = json.loads((cfg.reports_dir / "experiment_setup.json").read_text())["setup_id"]
    scorecard = pd.read_csv(cfg.outputs_dir / "experiment_setups" / setup_id / "classifier_comparison.csv")
    assert len(scorecard) == 1
    assert scorecard.loc[0, "reported_cv_folds"] == 3
    assert np.isclose(scorecard.loc[0, "cv_macro_f1_mean"],
                      result["single_model_cv_stats"]["mean"]["macro_f1"])
    assert pd.isna(scorecard.loc[0, "validation_macro_f1"])
    assert "cv_macro_pr_auc_std" in scorecard.columns


@pytest.mark.parametrize("multilabel", [True])
def test_cv_reporting_does_not_change_final_test_results_or_calibration(tmp_path, multilabel):
    no_cv_cfg, no_cv = _run(tmp_path / "without_cv", folds=0, multilabel=multilabel)
    cv_cfg, with_cv = _run(tmp_path / "with_cv", folds=3, multilabel=multilabel)
    for metric in (("macro_f1", "micro_f1") if multilabel else ("macro_f1",)):
        assert np.isclose(no_cv["test_stats"][metric], with_cv["test_stats"][metric])
    for metric in ("macro", "micro"):
        assert np.isclose(no_cv["test_stats"]["pr_auc"][metric],
                          with_cv["test_stats"]["pr_auc"][metric])
    assert no_cv["validation_stats"] is not None
    assert with_cv["validation_stats"] is None
    if multilabel:
        before = json.loads((no_cv_cfg.reports_dir / "multilabel_thresholds.json").read_text())
        after = json.loads((cv_cfg.reports_dir / "multilabel_thresholds.json").read_text())
        assert before == after


def test_automatic_tabular_selection(tmp_path):
    cfg, result = _run(tmp_path, folds=0, selected_model=None)
    assert result["final_model_name"] == "tuned_naive_bayes"
    assert (cfg.reports_dir / "cv_model_comparison.csv").is_file()
    assert not (cfg.reports_dir / "final_model_comparison.csv").exists()
    assert (cfg.reports_dir / "test_metrics.json").is_file()
