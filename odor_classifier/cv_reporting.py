"""Independent, descriptive CV reporting for a single classifier.

CV operates on the full train+validation partition (the non-test portion). It never selects hyperparameters,
calibrates thresholds, or refits the production/test model. The existing
feature pipeline is already fitted upstream, so these fold scores are
*diagnostic* and are not a fully nested validation of feature selection.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import auc, f1_score, roc_curve

from .evaluation import compute_pr_auc
from .optuna_search import make_cv_folds

logger = logging.getLogger(__name__)

SCORE_NAMES = (
    "macro_f1", "micro_f1", "macro_pr_auc", "micro_pr_auc", "macro_roc_auc",
)


def _fold_scores(y_true, proba, class_names, multilabel, threshold):
    """Compute fixed-threshold F1 and probability-based AUCs without plotting."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n_classes = len(class_names)
    if proba.shape != (len(y_true), n_classes):
        raise ValueError(
            f"CV predict_proba returned shape {proba.shape}, expected "
            f"({len(y_true)}, {n_classes}). Check class coverage in CV training folds."
        )
    if multilabel:
        prediction = (proba >= threshold).astype(int)
        y_binary = y_true
        f1_kwargs = {"zero_division": 0}
    else:
        prediction = proba.argmax(axis=1)
        y_binary = np.eye(n_classes, dtype=int)[y_true]
        f1_kwargs = {"labels": list(range(n_classes)), "zero_division": 0}

    precision_recall = compute_pr_auc(y_true, proba, class_names, multilabel=multilabel)
    valid = [i for i in range(n_classes) if np.unique(y_binary[:, i]).size == 2]
    # Match evaluation.py: area under the *macro-average ROC curve*, rather
    # than the arithmetic mean of the one-vs-rest AUC values. When a rare
    # class has only one outcome in a fold, omit only that undefined curve.
    macro_roc_auc = None
    if valid:
        curves = [roc_curve(y_binary[:, i], proba[:, i])[:2] for i in valid]
        all_fpr = np.unique(np.concatenate([fpr for fpr, _ in curves]))
        mean_tpr = np.mean([np.interp(all_fpr, fpr, tpr) for fpr, tpr in curves], axis=0)
        macro_roc_auc = float(auc(all_fpr, mean_tpr))
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", **f1_kwargs)),
        "micro_f1": float(f1_score(y_true, prediction, average="micro", **f1_kwargs)),
        "macro_pr_auc": precision_recall["macro"],
        "micro_pr_auc": precision_recall["micro"],
        "macro_roc_auc": macro_roc_auc,
    }


def evaluate_single_model_cv(model, X_train, y_train, class_names, config, *,
                             cv_strat=None):
    """Fit clones on folds of the supplied non-test (train+validation) data.

    The reported CV labels use the fixed global multilabel threshold, not
    validation-optimized per-label thresholds. CV settings have no effect on
    the final fitted estimator, Optuna, or threshold-calibration behavior.
    """
    folds_count = int(config.cv_folds)
    if folds_count < 2:
        raise ValueError("Cross-validation reporting requires at least 2 folds")
    y = np.asarray(y_train)
    if len(y) < folds_count:
        raise ValueError(f"Cannot use {folds_count} CV folds with only {len(y)} training samples")
    if y.ndim == 1:
        unique, counts = np.unique(cv_strat if cv_strat is not None else y, return_counts=True)
        if len(unique) < 2 or min(counts) < folds_count:
            raise ValueError(
                f"Cannot use {folds_count}-fold stratified CV: the least represented "
                f"training class has {min(counts)} samples. Reduce CV folds."
            )

    folds = make_cv_folds(folds_count, y, cv_strat, config.random_state)
    records = []
    for fold_num, (tr, va) in enumerate(folds, start=1):
        logger.info("Single-model descriptive CV: fold %d/%d, fitting %d samples; scoring %d.",
                    fold_num, folds_count, len(tr), len(va))
        from sklearn.multioutput import MultiOutputClassifier
        from .multilabel import multioutput_predict_proba
        fitted = clone(model).fit(X_train.iloc[tr], y[tr])
        probabilities = (multioutput_predict_proba(fitted, X_train.iloc[va])
                         if isinstance(fitted, MultiOutputClassifier)
                         else fitted.predict_proba(X_train.iloc[va]))
        scores = _fold_scores(y[va], probabilities, class_names,
                              config.multilabel, config.multilabel_threshold)
        records.append({"fold": fold_num, "n_fit": len(tr), "n_scored": len(va), **scores})
    frame = pd.DataFrame.from_records(records)
    mean = {name: _json_number(frame[name].mean(skipna=True)) for name in SCORE_NAMES}
    std = {name: _json_number(frame[name].std(ddof=1, skipna=True)) for name in SCORE_NAMES}
    summary = {
        "cv_folds": folds_count,
        "optuna_used": bool(config.use_optuna),
        "evaluated_on": "train_plus_validation",
        "cv_total_samples": len(y),
        "cv_total_scored": int(frame["n_scored"].sum()),
        "feature_pipeline_fitted_on": "train_plus_validation",
        "fold_thresholds": "fixed_global_threshold" if config.multilabel else "argmax",
        "threshold_value": float(config.multilabel_threshold) if config.multilabel else None,
        "std_definition": "sample_standard_deviation_ddof_1",
        "roc_auc_definition": "area_under_macro_averaged_one_vs_rest_roc_curve_over_defined_classes",
        "pr_auc_definition": "average_precision_over_defined_classes",
        "mean": mean,
        "std": std,
        "folds": frame.where(pd.notna(frame), None).to_dict(orient="records"),
    }
    logger.info("Single-model %d-fold CV: macro-F1=%.4f ± %.4f, macro PR-AUC=%s.",
                folds_count, mean["macro_f1"], std["macro_f1"], mean["macro_pr_auc"])
    return frame, summary


def _json_number(value):
    return float(value) if value is not None and np.isfinite(value) else None
