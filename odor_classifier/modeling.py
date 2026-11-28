"""Advanced multi-class classifiers, validation-set-based model selection
and Optuna hyperparameter tuning.

Model comparison and hyperparameter tuning use a single held-out
**validation** split carved out of the training data (not k-fold
cross-validation) -- fit on train, score on validation. The final chosen
model is then refit on train+validation before being scored once on the
untouched held-out **test** set. This keeps three genuinely separate
roles (train / validation / test) instead of repeatedly refitting on
rotating folds.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _calibrated_svm(**svc_kwargs) -> CalibratedClassifierCV:
    """SVC wrapped for predict_proba via cross-validated calibration.

    ``SVC(probability=True)`` is deprecated (removal planned in sklearn
    1.11); ``CalibratedClassifierCV`` is the replacement it recommends and
    gives the same predict_proba interface the rest of the pipeline needs.
    """
    return CalibratedClassifierCV(SVC(**svc_kwargs), method="sigmoid", cv=3, ensemble=False)


def build_candidate_models(config: Config, n_classes: int) -> Dict[str, object]:
    rs = config.random_state
    return {
        **(
            {
                "random_forest": RandomForestClassifier(
                    n_estimators=500, class_weight="balanced", random_state=rs, n_jobs=-1
                )
            }
            if config.use_random_forest
            else {}
        ),
        **(
            {
                "hist_gradient_boosting": HistGradientBoostingClassifier(
                    max_iter=300, random_state=rs
                )
            }
            if config.use_hist_gradient_boosting
            else {}
        ),
        **(
            {
                "extra_trees": ExtraTreesClassifier(
                    n_estimators=500, class_weight="balanced", random_state=rs, n_jobs=-1
                )
            }
            if config.use_extra_trees
            else {}
        ),
        **(
            {
                "catboost": CatBoostClassifier(
                    iterations=689,
                    depth=5,
                    learning_rate=0.04,
                    l2_leaf_reg=0.77,
                    loss_function="MultiClass",
                    auto_class_weights="Balanced",
                    random_state=rs,
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                )
            }
            if config.use_catboost
            else {}
        ),
        **({"naive_bayes": GaussianNB()} if config.use_naive_bayes else {}),
        **(
            {
                "xgboost": XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    # objective/num_class/eval_metric deliberately left unset --
                    # XGBoost auto-detects binary vs multiclass (and the right
                    # metric) from y at fit time, which matters here because the
                    # exact same candidate is reused for multilabel mode's
                    # per-label *binary* sub-classifiers (see multilabel.py).
                    # Hardcoding multi:softprob/num_class would break those.
                    random_state=rs,
                    n_jobs=-1,
                )
            }
            if config.use_xgboost
            else {}
        ),
        **(
            {
                "lightgbm": LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.08,
                    class_weight="balanced",
                    random_state=rs,
                    n_jobs=-1,
                    verbosity=-1,
                )
            }
            if config.use_lightgbm
            else {}
        ),
        **(
            {
                "mlp": MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    max_iter=500,
                    early_stopping=True,
                    random_state=rs,
                )
            }
            if config.use_mlp
            else {}
        ),
        **(
            {
                "svm": _calibrated_svm(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=rs,
                )
            }
            if config.use_svm
            else {}
        ),
    }


def evaluate_candidates_on_validation(
    models: Dict[str, object],
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    checkpoint=None,
    cv_folds: int = 0,
    cv_strat: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score each candidate's macro-F1, with default hyperparameters.

    With ``cv_folds >= 2``: stratified k-fold cross-validation inside the
    training split (optuna_search.make_cv_folds: every label for a multi-
    label ``y_train``, else ``cv_strat``/``y_train``), score = mean over
    folds; the validation split isn't used.
    Otherwise: fit on the training split, score on the held-out validation
    split (one fit per model).

    ``checkpoint`` (checkpoint.Checkpointer), if given, saves each
    candidate's score as soon as it's computed, so a resumed run skips the
    candidates that already finished.
    """
    use_cv = cv_folds is not None and cv_folds >= 2
    if use_cv:
        from .optuna_search import make_cv_folds

        folds = make_cv_folds(cv_folds, y_train, cv_strat, random_state)
    rows = []
    for name, model in models.items():
        key = f"round1_score_{name}"
        if checkpoint is not None and checkpoint.has(key):
            score = checkpoint.load(key)
            rows.append({"model": name, "val_f1_macro": score})
            logger.info(
                "%s: validation f1_macro = %.4f (from checkpoint)", name, score)
            continue
        if use_cv:
            logger.info(
                "Scoring %s by %d-fold cross-validation on train ...", name, cv_folds)
            fold_scores = []
            for tr, va in folds:
                fold_model = clone(model).fit(X_train.iloc[tr], y_train[tr])
                fold_scores.append(f1_score(y_train[va], fold_model.predict(
                    X_train.iloc[va]), average="macro"))
            score = float(np.mean(fold_scores))
            logger.info("%s: per-fold f1_macro = %s", name,
                        ", ".join(f"{s:.4f}" for s in fold_scores))
        else:
            logger.info("Fitting %s on train, scoring on validation ...", name)
            model.fit(X_train, y_train)
            score = f1_score(y_val, model.predict(X_val), average="macro")
        if checkpoint is not None:
            checkpoint.save(key, score)
        rows.append({"model": name, "val_f1_macro": score})
        logger.info("%s: validation f1_macro = %.4f", name, score)
    return pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)


def tune_best_model(
    model_name: str,
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    config: Config,
    wrap_fn=None,
    storage=None,
    study_name=None,
    cv_strat=None,
):
    """Optuna hyperparameter search for the winning model family: each
    trial is scored (macro-F1) by ``config.cv_folds``-fold stratified
    cross-validation on the training split (stratified on ``cv_strat``), or,
    with cv_folds < 2, by fitting on train and scoring on the held-out
    validation split. Thin wrapper around
    optuna_search.run_optuna_search -- this function's only job is to
    define the classification-specific search space below; the actual
    search engine lives in optuna_search.

    ``wrap_fn``, if given, is applied to each trial's raw estimator before
    fitting -- multi-label mode passes a MultiOutputClassifier wrapper here
    so the exact same search space/estimators work unchanged.
    """

    def make_model(trial: "optuna.trial.BaseTrial"):
        rs = config.random_state
        if model_name == "xgboost":
            return XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float(
                    "reg_lambda", 1e-3, 10.0, log=True),
                # objective/eval_metric left unset -- see build_candidate_models.
                random_state=rs,
                n_jobs=-1,
            )
        if model_name == "lightgbm":
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 700),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0),
                class_weight="balanced",
                random_state=rs,
                n_jobs=-1,
                verbosity=-1,
            )
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 900),
                max_depth=trial.suggest_int("max_depth", 5, 40),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                max_features=trial.suggest_float("max_features", 0.1, 1.0),
                class_weight="balanced",
                random_state=rs,
                n_jobs=-1,
            )
        if model_name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=trial.suggest_int("max_iter", 150, 600),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True),
                l2_regularization=trial.suggest_float(
                    "l2_regularization", 1e-3, 10.0, log=True),
                random_state=rs,
            )
        if model_name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 900),
                max_depth=trial.suggest_int("max_depth", 5, 40),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                max_features=trial.suggest_float("max_features", 0.1, 1.0),
                class_weight="balanced",
                random_state=rs,
                n_jobs=-1,
            )
        if model_name == "catboost":
            return CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 200, 700),
                depth=trial.suggest_int("depth", 4, 10),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float(
                    "l2_leaf_reg", 1e-2, 10.0, log=True),
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_state=rs,
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
            )
        if model_name == "naive_bayes":
            return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-3, log=True))
        if model_name == "svm":
            return _calibrated_svm(
                kernel=trial.suggest_categorical("kernel", ["rbf", "poly"]),
                C=trial.suggest_float("C", 0.1, 100.0, log=True),
                gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                class_weight="balanced",
                random_state=rs,
            )
        # fall back: MLP
        return MLPClassifier(
            hidden_layer_sizes=(trial.suggest_categorical(
                "hidden", [128, 256, 512]),),
            alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            max_iter=400,
            early_stopping=True,
            random_state=rs,
        )

    from .optuna_search import run_optuna_search

    return run_optuna_search(
        model_name, make_model, X_train, y_train, X_val, y_val, config,
        score_fn=lambda y_true, y_pred: f1_score(
            y_true, y_pred, average="macro"),
        metric_name="f1_macro", wrap_fn=wrap_fn, direction="maximize",
        storage=storage, study_name=study_name, cv_folds=config.cv_folds, cv_strat=cv_strat,
    )

