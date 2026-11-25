"""OdorClassifier multi-label training command."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from odor_classifier.config import Config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-method", nargs="+", default=["mutual_info"])
    p.add_argument("--pre-reduce-to", type=int, default=800)
    p.add_argument("--pca-components", type=float, default=0.95)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--validation-size", type=float, default=0.15)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--multilabel-threshold", type=float, default=0.5)
    p.add_argument("--no-calibrate-thresholds", action="store_true")
    p.add_argument("--no-heldout-label-audit", action="store_true",
                   help="Save held-out audit inputs but skip running the audit automatically.")
    p.add_argument("--heldout-audit-knn", choices=("both", "morgan", "descriptor"), default="both",
                   help="KNN representations for the automatic held-out audit (default: both).")
    p.add_argument("--per-label-feature-selection", action="store_true")
    p.add_argument("--per-label-max-features", type=int, default=30)
    p.add_argument("--selected-model", choices=("svm", "naive_bayes", "extra_trees", "catboost", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm", "mlp"), default=None)
    p.add_argument("--use-pca", action="store_true")
    for name in ("svm", "naive-bayes", "extra-trees", "catboost", "random-forest",
                 "hist-gradient-boosting", "xgboost", "lightgbm", "mlp", "optuna",
                 "rdkit-descriptors", "morgan-fingerprints", "maccs", "mordred", "mol2vec"):
        p.add_argument("--no-" + name, action="store_true")
    p.add_argument("--tune-all-candidates", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--fresh-start", action="store_true")
    p.add_argument("--use-field-descriptors", action="store_true")
    p.add_argument("--use-grind-descriptors", action="store_true")
    p.add_argument("--reuse-experiment-setup", default=None,
                   help="ID of archived train/validation/test split and fitted feature pipeline")
    return p.parse_args(argv)


def make_config(a) -> Config:
    cfg = Config()
    fields = {
        "selection_method": ",".join(a.selection_method),
        "selection_pre_reduce_to": a.pre_reduce_to,
        "pca_n_components": a.pca_components,
        "n_optuna_trials": a.trials,
        "test_size": a.test_size,
        "validation_size": a.validation_size,
        "cv_folds": a.cv_folds,
        "random_state": a.random_state,
        "multilabel_threshold": a.multilabel_threshold,
        "run_heldout_label_audit": not a.no_heldout_label_audit,
        "heldout_audit_knn_modes": a.heldout_audit_knn,
        "calibrate_multilabel_thresholds": not a.no_calibrate_thresholds,
        "per_label_feature_selection": a.per_label_feature_selection,
        "per_label_max_features": a.per_label_max_features,
        "selected_model": a.selected_model,
        "use_pca": a.use_pca,
        "tune_all_candidates": a.tune_all_candidates,
        "resume_from_checkpoint": not a.fresh_start,
        "use_field_descriptors": a.use_field_descriptors,
        "use_grind_descriptors": a.use_grind_descriptors,
        "reuse_experiment_setup": a.reuse_experiment_setup,
    }
    for name in ("svm", "naive_bayes", "extra_trees", "catboost", "random_forest",
                 "hist_gradient_boosting", "xgboost", "lightgbm", "mlp", "optuna",
                 "rdkit_descriptors", "morgan_fingerprints", "maccs", "mordred", "mol2vec"):
        fields["use_" + name] = not getattr(a, "no_" + name)
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def main(argv=None):
    a = parse_args(argv)
    from odor_classifier.pipeline import PIPELINE_STEPS, run_full_pipeline, update_status
    config = make_config(a)
    now = lambda: datetime.now().isoformat(timespec="seconds")
    update_status(config, status="running", started_at=now(), pid=os.getpid(),
                  current_step=0, total_steps=len(PIPELINE_STEPS),
                  step_name="Preparing training", run_config={
                      "selected_model": config.selected_model,
                      "reuse_experiment_setup": config.reuse_experiment_setup,
                      "selection_method": config.selection_method,
                      "multilabel": config.multilabel,
                      "test_size": config.test_size, "validation_size": config.validation_size,
                      "random_state": config.random_state,
                  }, error="")
    try:
        result = run_full_pipeline(config, force_recompute_features=a.force_recompute)
        update_status(config, status="done", finished_at=now(),
                      final_model_name=result["final_model_name"])
        print("Completed. Best model:", result["final_model_name"], flush=True)
    except Exception:
        err = traceback.format_exc()
        update_status(config, status="failed", finished_at=now(), error=err)
        print(err, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
