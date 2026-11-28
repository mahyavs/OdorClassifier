"""Generic Optuna hyperparameter-search engine, used by classification
(modeling.tune_best_model).

The caller supplies the per-model-family search space (``make_model``) and
scoring function; everything else -- study creation, cross-validation
folds, pruning, per-trial progress logging, checkpointing, refitting the
winning trial's params into a real model -- lives here once, so adding a
new model family never means re-deriving this machinery again.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def make_cv_folds(n_splits: int, y, strat=None, random_state: int = 42) -> list:
    """(train_idx, val_idx) pairs for k-fold cross-validation.

    Multi-label ``y`` (2-D indicator matrix): iterative stratification
    (Sechidis et al., 2011; ``iterative-stratification`` package), which
    balances every label across folds -- rarest labels first -- not just
    one. Falls back to stratifying on ``strat`` (class1) with a warning if
    the package isn't installed.

    Single-label ``y``: ordinary StratifiedKFold on ``strat`` (or ``y``).
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    if y.ndim == 2:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

            splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(splitter.split(np.zeros(len(y)), y))
        except ImportError:
            logger.warning(
                "Package 'iterative-stratification' not installed (pip install iterative-stratification) -- "
                "multi-label CV folds are stratified on class1 only instead of on every label."
            )
    key = np.asarray(strat if strat is not None else y)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                .split(np.zeros(len(key)), key))


def run_optuna_search(
    model_name: str,
    make_model: Callable[["optuna.trial.BaseTrial"], object],
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    config: Config,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str = "score",
    wrap_fn: Optional[Callable[[object], object]] = None,
    direction: str = "maximize",
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    cv_folds: int = 0,
    cv_strat: Optional[np.ndarray] = None,
) -> Tuple[object, "optuna.Study"]:
    """Runs ``config.n_optuna_trials`` trials: each builds a model from
    ``make_model(trial)`` (optionally wrapped via ``wrap_fn`` -- multi-
    label classification passes a MultiOutputClassifier wrapper here so
    the exact same search space/estimators work unchanged) and scores it
    with ``score_fn``. Returns the best-scoring model -- refit fresh from
    its trial's params, not yet fit on any data -- plus the finished
    Optuna study.

    Scoring: with ``cv_folds >= 2``, stratified k-fold cross-validation
    inside ``X_train`` (see make_cv_folds: every label for a multi-label
    ``y_train``, else ``cv_strat``/``y_train``), trial score = mean over
    folds; ``X_val``/``y_val`` are not used.
    A trial clearly worse than the median of earlier trials after a fold
    is pruned (stopped early). Otherwise (``cv_folds < 2``): fit on the
    training split, score on the held-out validation split.

    ``storage``/``study_name`` (checkpoint.Checkpointer), if given, persist
    every trial to disk: rerunning after an interruption continues the same
    study with only the trials still missing.
    """
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    use_cv = cv_folds is not None and cv_folds >= 2
    if use_cv:
        from sklearn.base import clone

        folds = make_cv_folds(cv_folds, y_train, cv_strat, config.random_state)

    def _n_finished(study: "optuna.Study") -> int:
        # Pruned trials count toward n_optuna_trials (they were evaluated,
        # just stopped early), so pruning shortens the search, not lengthens it.
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))

    def objective(trial: "optuna.trial.Trial") -> float:
        model = make_model(trial)
        if wrap_fn is not None:
            model = wrap_fn(model)
        if not use_cv:
            model.fit(X_train, y_train)
            return score_fn(y_val, model.predict(X_val))
        scores = []
        for i, (tr, va) in enumerate(folds):
            fold_model = clone(model).fit(X_train.iloc[tr], y_train[tr])
            scores.append(score_fn(y_train[va], fold_model.predict(X_train.iloc[va])))
            trial.report(float(np.mean(scores)), step=i)
            if i < len(folds) - 1 and trial.should_prune():
                trial.set_user_attr("pruned_after_fold", i + 1)
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    # optuna's own per-trial logging is silenced above (set_verbosity(WARNING)),
    # so without this callback a slow-to-fit model (CatBoost/SVM/GPR in
    # particular can take minutes per trial) leaves this whole step
    # completely silent for its entire duration -- indistinguishable from
    # actually being stuck. This gives one log line per trial instead, with
    # a running best-so-far score.
    score_label = f"cv{cv_folds}_{metric_name}" if use_cv else f"val_{metric_name}"

    def _log_trial(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        if trial.state == TrialState.PRUNED:
            logger.info(
                "  %s: trial %d/%d pruned after fold %s/%d (below the median of earlier trials)",
                model_name, _n_finished(study), config.n_optuna_trials,
                trial.user_attrs.get("pruned_after_fold", "?"), cv_folds,
            )
            return
        logger.info(
            "  %s: trial %d/%d %s=%.4f (best so far: %.4f)",
            model_name, _n_finished(study), config.n_optuna_trials, score_label,
            trial.value if trial.value is not None else float("nan"), study.best_value,
        )

    study = optuna.create_study(
        direction=direction, storage=storage, study_name=study_name, load_if_exists=storage is not None,
        # Pruning only makes sense with per-fold intermediate scores; the
        # first 5 trials always run fully to establish a baseline.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5) if use_cv else optuna.pruners.NopPruner(),
    )
    if use_cv:
        logger.info("  %s: scoring each trial by %d-fold stratified cross-validation on the train split.",
                    model_name, cv_folds)
    # Resumed study: only run the trials still missing (a trial that was cut
    # off mid-fit stays incomplete and is simply redone).
    n_done = _n_finished(study)
    if n_done:
        logger.info("  %s: resuming Optuna search -- %d/%d trials already completed.",
                    model_name, n_done, config.n_optuna_trials)
    n_remaining = max(0, config.n_optuna_trials - n_done)
    if n_remaining:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False, callbacks=[_log_trial])

    best_trial = optuna.trial.FixedTrial(study.best_params)
    best_model = make_model(best_trial)
    if wrap_fn is not None:
        best_model = wrap_fn(best_model)
    return best_model, study
