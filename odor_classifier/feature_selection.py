"""Pluggable "variable selection" strategies used by FeaturePipeline.

Five strategies are available (``config.selection_method``):

- ``mutual_info``: fast, model-free ranking, keep the top K by mutual
  information with the target (the original default).
- ``boruta``: all-relevant wrapper selection -- iteratively compares each
  real feature's RandomForest importance against random "shadow" copies
  of itself, keeping only features that consistently beat their shadow.
  Considerably slower, but is the more rigorous / commonly cited approach
  for QSAR-style variable selection.
- ``rfecv``: recursive feature elimination with cross-validation, using a
  fast gradient-boosted tree as the ranking estimator.
- ``lasso``: SelectFromModel around an L1-regularized logistic regression.
  Cheap and convex; catches linear per-feature signal Boruta/RFECV might
  bury under interaction effects.
- ``shap``: top-K by mean |SHAP value| from a quick LightGBM model --
  accounts for interactions and the actual decision surface, not just
  marginal association with the target.

``selection_method`` may also be ``"none"`` (skip variable selection and
keep every feature that survived the variance/correlation filters), or a
comma-separated chain such as ``"boruta,shap"`` -- the methods then run in
that order, each on the previous one's selected columns.

Boruta and RFECV both scale badly with a few thousand raw descriptor
columns, so they (and lasso/shap) first coarsely pre-filter to
``config.selection_pre_reduce_to`` columns (default 800; 0 = no
pre-filter) by mutual information before running the slower selector.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.model_selection import StratifiedKFold

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _mutual_info_top_k(df: pd.DataFrame, y: pd.Series, k: int, config: Config, Y=None) -> list[str]:
    """Top-k columns by mutual information with ``y`` (class1) -- or, when a
    multi-label target matrix ``Y`` is given, by the AVERAGE mutual
    information over all its labels (each label a binary present/absent
    target), so features that matter for secondary labels rank too."""
    if df.shape[1] <= k:
        return list(df.columns)
    if Y is not None:
        Y = np.asarray(Y)
        mi = np.mean([
            mutual_info_classif(df.values, Y[:, j], random_state=config.random_state, discrete_features=False,
                                n_jobs=-1)
            for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)  # a constant column carries no information
        ], axis=0)
    else:
        mi = mutual_info_classif(df.values, y, random_state=config.random_state, discrete_features=False,
                                 n_jobs=-1)  # parallel over columns; same result, faster
    order = np.argsort(mi)[::-1][:k]
    return [df.columns[i] for i in sorted(order)]


# Selectors below that run behind the mutual-information pre-filter
# (column cap: config.selection_pre_reduce_to).
_PRE_REDUCED_METHODS = ("boruta", "rfecv", "lasso", "shap")


def _pre_reduce(df: pd.DataFrame, y: pd.Series, config: Config, Y=None) -> pd.DataFrame:
    cap = config.selection_pre_reduce_to
    if cap <= 0 or df.shape[1] <= cap:
        return df
    logger.info("Pre-reducing %d columns to %d by mutual information before running %s.",
                df.shape[1], cap, config.selection_method)
    cols = _mutual_info_top_k(df, y, cap, config, Y)
    return df[cols]


def select_mutual_info(df: pd.DataFrame, y: pd.Series, config: Config = DEFAULT_CONFIG, Y=None) -> list[str]:
    return _mutual_info_top_k(df, y, config.max_features_after_selection, config, Y)


def select_boruta(df: pd.DataFrame, y: pd.Series, config: Config = DEFAULT_CONFIG, Y=None) -> list[str]:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    reduced = _pre_reduce(df, y, config, Y)
    rf = RandomForestClassifier(n_jobs=-1, max_depth=7, class_weight="balanced", random_state=config.random_state)
    selector = BorutaPy(
        rf, n_estimators="auto", max_iter=config.boruta_max_iter,
        random_state=config.random_state, verbose=0,
    )
    y_arr = np.asarray(y)
    selector.fit(np.asarray(reduced.values, dtype=np.float64), y_arr)
    selected = list(reduced.columns[selector.support_])

    if not selected:
        logger.warning("Boruta confirmed 0 features; falling back to its tentative set / mutual-info top-K.")
        tentative = list(reduced.columns[selector.support_weak_])
        selected = tentative if tentative else select_mutual_info(df, y, config, Y)

    logger.info("Boruta confirmed %d/%d features as relevant.", len(selected), reduced.shape[1])
    return selected


def select_rfecv(df: pd.DataFrame, y: pd.Series, config: Config = DEFAULT_CONFIG, Y=None) -> list[str]:
    from lightgbm import LGBMClassifier

    reduced = _pre_reduce(df, y, config, Y)
    estimator = LGBMClassifier(
        n_estimators=150, learning_rate=0.1, random_state=config.random_state,
        n_jobs=-1, verbosity=-1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.random_state)
    selector = RFECV(
        estimator, step=config.rfecv_step, min_features_to_select=config.rfecv_min_features,
        cv=cv, scoring="f1_macro", n_jobs=1,
    )
    selector.fit(reduced.values, y)
    selected = list(reduced.columns[selector.support_])
    logger.info("RFECV selected %d/%d features (optimal by CV macro-F1).", len(selected), reduced.shape[1])
    return selected


def select_lasso(df: pd.DataFrame, y: pd.Series, config: Config = DEFAULT_CONFIG, Y=None) -> list[str]:
    """L1-regularized multinomial logistic regression, via SelectFromModel.

    Fast, convex, and (unlike the tree-based selectors above) sensitive to
    *linear* signal that a single feature carries on its own -- a useful,
    cheap complement to Boruta/RFECV rather than a replacement.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    reduced = _pre_reduce(df, y, config, Y)
    # saga's convergence is sensitive to feature scale; FeaturePipeline scales
    # *after* selection, so standardize locally just for this fit.
    scaled = StandardScaler().fit_transform(reduced.values)
    estimator = LogisticRegression(
        l1_ratio=1.0, solver="saga", C=0.1, max_iter=5000,
        class_weight="balanced", random_state=config.random_state,
    )
    selector = SelectFromModel(estimator, max_features=config.max_features_after_selection)
    with warnings.catch_warnings():
        # Coefficient ranking for selection doesn't need full convergence.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        selector.fit(scaled, y)
    selected = list(reduced.columns[selector.get_support()])
    if not selected:
        logger.warning("L1-logistic selection kept 0 features; falling back to mutual-info top-K.")
        return select_mutual_info(df, y, config, Y)
    logger.info("Lasso (L1-logistic) selected %d/%d features.", len(selected), reduced.shape[1])
    return selected


def select_shap(df: pd.DataFrame, y: pd.Series, config: Config = DEFAULT_CONFIG, Y=None) -> list[str]:
    """Rank features by mean |SHAP value| from a quick LightGBM model, keep top K.

    More expensive than mutual information but accounts for interactions
    and the specific model class's decision surface, not just marginal
    per-feature association with the target.
    """
    import shap
    from lightgbm import LGBMClassifier

    reduced = _pre_reduce(df, y, config, Y)
    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.1, random_state=config.random_state,
        n_jobs=-1, verbosity=-1,
    )
    model.fit(reduced.values, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(reduced.values)

    if isinstance(shap_values, list):  # one (n_samples, n_features) array per class
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif np.asarray(shap_values).ndim == 3:  # (n_samples, n_features, n_classes)
        importance = np.abs(shap_values).mean(axis=(0, 2))
    else:  # binary case: (n_samples, n_features)
        importance = np.abs(shap_values).mean(axis=0)

    k = min(config.max_features_after_selection, reduced.shape[1])
    top_idx = np.argsort(importance)[::-1][:k]
    selected = [reduced.columns[i] for i in sorted(top_idx)]
    logger.info("SHAP-based selection kept top %d/%d features by mean |SHAP|.", len(selected), reduced.shape[1])
    return selected


SELECTORS = {
    "mutual_info": select_mutual_info,
    "boruta": select_boruta,
    "rfecv": select_rfecv,
    "lasso": select_lasso,
    "shap": select_shap,
}


NO_SELECTION = "none"


def parse_selection_methods(method) -> list[str]:
    """Normalize a selection-method setting into an ordered chain of methods.

    ``None`` / ``""`` / ``"none"`` -> ``[]`` (no variable selection).
    ``"boruta"`` -> ``["boruta"]``.
    ``"boruta,shap"`` or ``["boruta", "shap"]`` -> ``["boruta", "shap"]``
    (applied in that order, each one on the previous one's output).
    """
    if method is None:
        return []
    if isinstance(method, str):
        parts = method.split(",")
    else:
        parts = list(method)
    methods = [str(m).strip() for m in parts if str(m).strip()]
    return [m for m in methods if m.lower() != NO_SELECTION]


def select_features(df: pd.DataFrame, y: pd.Series, config: Config = DEFAULT_CONFIG,
                    stage_counts: list | None = None, Y=None) -> list[str]:
    """``stage_counts``, if given, gets one (stage name, #features) entry
    appended per MI pre-filter / selection step, for the feature funnel report.

    ``Y`` (multi-label mode): the per-label target matrix, rows aligned with
    ``df``. Mutual information (the mutual_info method and the pre-filter in
    front of boruta/rfecv/lasso/shap) is then averaged over all labels; the
    one-target methods themselves still fit against ``y`` (class1).
    """
    methods = parse_selection_methods(config.selection_method)
    for method in methods:
        if method not in SELECTORS:
            raise ValueError(f"Unknown selection_method '{method}'. Choose one of "
                             f"{[NO_SELECTION] + list(SELECTORS)} (or a comma-separated chain).")
    if not methods:
        logger.info("Variable selection disabled -- keeping all %d filtered features.", df.shape[1])
        return list(df.columns)
    selected = list(df.columns)
    for method in methods:
        if len(selected) <= 1:
            break
        cap = config.selection_pre_reduce_to
        if stage_counts is not None and method in _PRE_REDUCED_METHODS and 0 < cap < len(selected):
            stage_counts.append((f"MI pre-filter before {method}", cap))
        selected = SELECTORS[method](df[selected], y, config, Y=Y)
        if stage_counts is not None:
            stage_counts.append((f"After {method}", len(selected)))
        if len(methods) > 1:
            logger.info("Variable selection step '%s' kept %d features.", method, len(selected))
    return selected
