"""Multi-label target construction and per-label classifier wrappers."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from .config import Config, DEFAULT_CONFIG, UC_LABEL

logger = logging.getLogger(__name__)


class SingleClassSafeClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a base estimator for MultiOutputClassifier's per-label
    sub-problems.

    Splitting one dictionary-derived label into a binary "present /
    absent" column per class, then further splitting that into train and
    validation, means any rare label can end up with only one class present
    in a given fold purely by chance. Some estimators handle that fine
    (RandomForest, HistGradientBoosting, LightGBM, GaussianNB, MLP all just
    predict the constant class); others don't -- CatBoost raises during
    fit, XGBoost fits without error but returns a malformed prediction
    shape, and the calibrated SVM's internal cross-fitting can hit a
    single-class fold even when the overall column has two classes. Rather
    than special-case each one, this wrapper catches any fit-time failure
    (or a garbled post-fit sanity check) and falls back to predicting the
    training majority class -- exactly what a well-behaved estimator would
    already do in this situation.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y)
        vals, counts = np.unique(y, return_counts=True)
        self.constant_ = vals[np.argmax(counts)]
        self.classes_ = vals if len(vals) > 1 else np.array([0, 1])
        self.fitted_ = None
        try:
            if len(vals) < 2:
                raise ValueError("single class present in this fold")
            model = clone(self.estimator).fit(X, y)
            # One-row sanity check in the same form the model was fit on: a
            # DataFrame row keeps its column names (a plain array would make
            # scikit-learn warn "X does not have valid feature names").
            first_row = X.iloc[:1] if hasattr(X, "iloc") else np.asarray(X)[:1]
            proba = model.predict_proba(first_row)
            if proba.shape[1] < 2:
                raise ValueError("degenerate predict_proba shape")
            self.fitted_ = model
        except Exception as exc:
            logger.debug("SingleClassSafeClassifier falling back to constant %r: %s", self.constant_, exc)
        return self

    def predict(self, X):
        if self.fitted_ is None:
            return np.full(len(X), self.constant_)
        # CatBoostClassifier (and possibly others) return (n, 1) rather than
        # sklearn's conventional (n,) -- MultiOutputClassifier stacks every
        # label's prediction together and breaks if the shapes don't match.
        return np.asarray(self.fitted_.predict(X)).ravel()

    def predict_proba(self, X):
        if self.fitted_ is None:
            n = len(X)
            proba = np.zeros((n, len(self.classes_)))
            proba[:, list(self.classes_).index(self.constant_)] = 1.0
            return proba
        return self.fitted_.predict_proba(X)


def extract_multilabel_targets(compounds: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Reconstruct ALL dictionary-defined binary classes from odor1..odorN.

    Stored class1..class10 annotations are not used to construct targets.
    The dictionary defines the stable target-column order even when a
    class has zero positive compounds in a particular input subset.
    """
    from .io_utils import load_class_dictionary
    from .target_labels import build_binary_targets

    dictionary = load_class_dictionary(config=config)
    Y, _, _ = build_binary_targets(compounds, dictionary, strict_unmapped=True)
    n_no_label = int(Y.sum(axis=1).eq(0).sum())
    if n_no_label:
        logger.warning("%d/%d compounds have no mapped odor labels.", n_no_label, len(Y))
    return Y


def jaccard_cooccurrence(Y: pd.DataFrame) -> pd.DataFrame:
    """Symmetric Jaccard co-occurrence matrix for binary odor labels."""
    arr = Y.to_numpy(dtype=bool)
    classes = list(Y.columns)
    n = len(classes)
    jac = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = (arr[:, i] & arr[:, j]).sum()
            union = (arr[:, i] | arr[:, j]).sum()
            score = inter / union if union else 0.0
            jac[i, j] = jac[j, i] = score
    return pd.DataFrame(jac, index=classes, columns=classes)


def wrap_models_for_multilabel(
    models: Dict[str, object], config: Config = DEFAULT_CONFIG
) -> Dict[str, MultiOutputClassifier]:
    """One independent binary classifier per label, using the base estimators
    from ``build_candidate_models``. Each
    base estimator is wrapped in ``SingleClassSafeClassifier`` first so a
    rare label degenerating to one class in some fold can't crash the run.

    When ``config.per_label_feature_selection`` is on, each label's
    estimator is preceded by its own ``SelectKBest`` (mutual information
    against that one label only), narrowing the shared, already-selected
    feature matrix down further per label instead of every label seeing
    the exact same columns -- see ``config.per_label_feature_selection``
    for the reasoning. This sits *inside* ``SingleClassSafeClassifier``
    (not outside it), so a degenerate single-class fold -- which would
    otherwise also break ``mutual_info_classif`` -- still falls back
    exactly the way it already does for the base estimator alone.
    """
    def _base_estimator(model):
        if not config.per_label_feature_selection:
            return model
        return Pipeline([
            ("select", SelectKBest(mutual_info_classif, k=config.per_label_max_features)),
            ("clf", model),
        ])

    return {
        name: MultiOutputClassifier(SingleClassSafeClassifier(_base_estimator(model)), n_jobs=1)
        for name, model in models.items()
    }


def multioutput_predict_proba(model: MultiOutputClassifier, X) -> np.ndarray:
    """MultiOutputClassifier.predict_proba returns a list of (n, 2) arrays
    (one per label: [P(absent), P(present)]) -- stack into one (n, n_labels)
    matrix of P(label present), which is what the rest of the pipeline
    (evaluation, UC assignment) expects.
    """
    per_label = model.predict_proba(X)
    return np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in per_label])


def calibrate_per_label_thresholds(
    y_true: np.ndarray, proba: np.ndarray, class_names: List[str],
    min_positives: int = 3, grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Per-label decision threshold that maximizes that label's own F1 on
    the given (validation) split, instead of one global cutoff.

    Multi-label columns are independently imbalanced -- a rare label
    systematically gets lower average predicted probability than a common
    one, so a single global threshold (e.g. 0.5) quietly under-predicts
    the rare labels no matter how good the underlying model is. This is a
    standard, "free" multi-label calibration step: it doesn't change the
    model at all, just where each label's decision boundary sits.

    Falls back to 0.5 for a label with too few positive validation
    examples (``min_positives``) to calibrate reliably.
    """
    from sklearn.metrics import f1_score

    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)

    thresholds: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        y_col = y_true[:, i]
        if y_col.sum() < min_positives:
            thresholds[name] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (proba[:, i] >= t).astype(int)
            f1 = f1_score(y_col, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[name] = best_t

    logger.info(
        "Calibrated per-label thresholds (median %.2f, range %.2f-%.2f).",
        float(np.median(list(thresholds.values()))), min(thresholds.values()), max(thresholds.values()),
    )
    return thresholds


def thresholds_to_array(thresholds: Dict[str, float], class_names: List[str], default: float = 0.5) -> np.ndarray:
    return np.array([thresholds.get(name, default) for name in class_names])
