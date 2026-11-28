"""End-to-end orchestration: load data -> features -> train -> evaluate ->
apply UC logic -> save all artifacts (model, feature pipeline, AD, plots,
metrics, labeled compound sheet).
"""
from __future__ import annotations

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from .applicability_domain import ApplicabilityDomain
from .checkpoint import Checkpointer
from .chemical_space import generate_chemical_space_plots
from .config import Config, DEFAULT_CONFIG, UC_LABEL
from .evaluation import full_evaluation_multilabel, plot_model_comparison
from .descriptors import smiles_to_mols
from .features import DESCRIPTOR_PREFIXES, FeaturePipeline, compute_raw_features, save_stage_counts
from .fingerprints import compute_morgan_fingerprints
from .io_utils import build_odor_to_class_map, load_class_dictionary, load_compounds
from .labeling import extract_labels, validate_against_dictionary
from .modeling import (
    build_candidate_models,
    evaluate_candidates_on_validation,
    tune_best_model,
)
from .multilabel import (
    calibrate_per_label_thresholds,
    extract_multilabel_targets,
    multioutput_predict_proba,
    thresholds_to_array,
    wrap_models_for_multilabel,
)
from .predict import assign_multilabel_with_uc
from .splitting import stratified_split
from .experiment_setups import (load_experiment_setup, save_experiment_setup,
                                record_classifier_result, validate_setup_descriptor_settings)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Pipeline stages in execution order.
PIPELINE_STEPS = [
    "Loading data & labels",
    "Computing molecular descriptors",
    "Building feature matrix",
    "Evaluating candidate model(s) on validation split",
    "Hyperparameter tuning (Optuna)",
    "Finalizing selected tabular classifier",
    "Evaluating on test & training sets",
    "Generating chemical-space plots",
    "Refitting production model & saving artifacts",
]


def update_status(config: Config, **fields) -> None:
    """Merge-safe write to outputs/training_status.json.

    Records run status and step-by-step progress.
    """
    path = config.outputs_dir / "training_status.json"
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(fields)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _cache_path(config: Config) -> Path:
    return config.outputs_dir / "cache" / "raw_features.parquet"


def _cache_matches_descriptor_config(columns, config: Config) -> bool:
    """True if the cached columns' descriptor families (by prefix, see
    features.DESCRIPTOR_PREFIXES) exactly match which families ``config``
    currently has enabled. Without this, toggling a descriptor checkbox
    off/on between runs and forgetting "force recompute" would silently
    reuse a cache built under the *previous* selection -- same row count,
    wrong columns -- since row count alone can't tell the two apart.
    """
    cols = list(columns)
    return all(
        getattr(config, field_name) == any(c.startswith(prefix) for c in cols)
        for field_name, prefix in DESCRIPTOR_PREFIXES.items()
    )


def _morgan_settings(config: Config) -> dict:
    """How the Morgan block was/would be generated -- stored next to the cache,
    since the column names alone can't reveal a radius/size/chirality change."""
    return {"morgan_radius": config.morgan_radius, "morgan_n_bits": config.morgan_n_bits,
            "morgan_use_chirality": config.morgan_use_chirality}


def _raw_feature_cache_signature(smiles: list[str], config: Config) -> dict:
    """Fingerprint the ordered molecule inputs plus ALL feature settings.

    Prevents a corrected SMILES/different row order/more feature parameters
    from silently reusing incompatible descriptors with the same row count.
    """
    feature_names = (
        "use_rdkit_descriptors", "use_morgan_fingerprints", "morgan_radius",
        "morgan_n_bits", "morgan_use_chirality", "use_maccs",
        "use_mordred", "mordred_n_proc", "use_mol2vec", "mol2vec_radius",
        "use_field_descriptors", "use_grind_descriptors", "grind_grid_margin",
        "grind_grid_spacing", "grind_max_nodes_per_probe",
        "grind_min_node_distance", "grind_bin_width", "grind_n_bins",
        "random_state",
    )
    encoded = json.dumps(list(smiles), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    model_path = config.mol2vec_model_path
    model_signature = (str(model_path.resolve()), model_path.stat().st_size,
                       model_path.stat().st_mtime_ns) if model_path.is_file() else (str(model_path), None)
    return {
        "smiles_sha256": hashlib.sha256(encoded).hexdigest(),
        "descriptor_settings": {key: getattr(config, key) for key in feature_names},
        "mol2vec_model_signature": model_signature if config.use_mol2vec else None,
    }


def _write_cache(raw: pd.DataFrame, config: Config, smiles: list[str]) -> None:
    cache_path = _cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_path)
    signature = _raw_feature_cache_signature(smiles, config)
    cache_path.with_suffix(".meta.json").write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )


def get_or_compute_raw_features(smiles: list[str], config: Config, force: bool = False) -> pd.DataFrame:
    cache_path = _cache_path(config)
    signature = _raw_feature_cache_signature(smiles, config)
    if cache_path.exists() and not force:
        meta_path = cache_path.with_suffix(".meta.json")
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            saved = None
        if saved == signature:
            cached = pd.read_parquet(cache_path)
            if len(cached) == len(smiles) and _cache_matches_descriptor_config(cached.columns, config):
                logger.info("Using molecular-feature cache with verified SMILES hash (%d rows)", len(cached))
                return cached
        logger.warning("Input SMILES/order or descriptor parameters changed (or cache lacks metadata): "
                       "rebuilding molecular features instead of reusing stale cache.")

    logger.info("Computing raw features for %d compounds...", len(smiles))
    t0 = time.time()
    raw = compute_raw_features(smiles, config)
    logger.info("Feature computation finished in %.1fs (%d columns)", time.time() - t0, raw.shape[1])
    _write_cache(raw, config, smiles)
    return raw


def _proba(model, X) -> np.ndarray:
    """predict_proba that works uniformly for tabular candidates, a
    MultiOutputClassifier (multilabel).
    MultiOutputClassifier.predict_proba returns a *list* of per-label
    (n, 2) arrays, not a single 2D array -- everything else already
    returns the (n, n_labels)/(n, n_classes) shape the rest of the
    pipeline expects.
    """
    from sklearn.multioutput import MultiOutputClassifier

    if isinstance(model, MultiOutputClassifier):
        return multioutput_predict_proba(model, X)
    return model.predict_proba(X)


def _pred_from_proba(proba: np.ndarray, config: Config, thresholds: Optional[np.ndarray] = None) -> np.ndarray:
    cutoffs = thresholds if thresholds is not None else config.multilabel_threshold
    return (proba >= cutoffs).astype(int)


def _labels_to_string(Y: pd.DataFrame, class_names: list[str]) -> pd.Series:
    """Comma-joined ground-truth label string per row, for a multilabel
    binary indicator matrix -- the multilabel analog of the single-label
    ``class1`` string column.
    """
    return Y[class_names].apply(lambda row: ", ".join(c for c in class_names if row[c] == 1), axis=1)


def run_full_pipeline(config: Config = DEFAULT_CONFIG, force_recompute_features: bool = False) -> dict:
    """Run the complete train + evaluate + label pipeline. Returns a dict of
    artifacts and summary results.
    """
    if not config.multilabel:
        raise ValueError("Only multi-label classification is supported.")

    def progress(step_idx: int) -> None:
        update_status(
            config, current_step=step_idx + 1, total_steps=len(PIPELINE_STEPS),
            step_name=PIPELINE_STEPS[step_idx],
        )
        logger.info("Step %d/%d: %s", step_idx + 1, len(PIPELINE_STEPS), PIPELINE_STEPS[step_idx])

    # Checkpoints (see checkpoint.py): every expensive stage below saves its
    # result, so rerunning an interrupted run with the same settings resumes
    # where it stopped. Recomputing descriptors invalidates them too.
    ckpt = Checkpointer(config, fresh=not config.resume_from_checkpoint or force_recompute_features)

    def cached(key: str, compute):
        """Load ``key`` from the checkpoint if present, else compute + save it."""
        if ckpt.has(key):
            return ckpt.load(key)
        obj = compute()
        ckpt.save(key, obj)
        return obj

    # 1. Load data ---------------------------------------------------------
    progress(0)
    compounds = load_compounds(config=config)
    class_dict = load_class_dictionary(config=config)
    odor_to_class = build_odor_to_class_map(class_dict)

    agreement = validate_against_dictionary(compounds, odor_to_class, config)
    logger.info("Sanity check: stored class1 agrees with re-derived odor1->class mapping in %.1f%% of rows",
                agreement * 100 if agreement == agreement else float("nan"))

    # The first mapped odor is retained for one-target selectors and plot colors.
    # The actual training target contains all mapped odor categories.
    labels = extract_labels(compounds, config)
    valid_mask = labels.notna()
    compounds = compounds.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].reset_index(drop=True)

    # Fail fast if an archived setup requires different descriptor settings;
    # molecular descriptor computation can otherwise take hours before the
    # mismatch would be noticed during setup restoration.
    if config.reuse_experiment_setup:
        validate_setup_descriptor_settings(config, config.reuse_experiment_setup)

    # 2. Features ------------------------------------------------------------
    # Compute features for the full valid compound set.
    progress(1)
    raw_features = get_or_compute_raw_features(
        compounds[config.smiles_column].tolist(), config, force=force_recompute_features,
    )

    y_full_target = extract_multilabel_targets(compounds, config)
    multilabel_class_names = list(y_full_target.columns)

    # Verify that every trainable dictionary class has its own target column.
    from .target_labels import dictionary_index, dictionary_target_audit
    expected, _ = dictionary_index(class_dict)
    missing = sorted(set(expected) - set(y_full_target.columns))
    if missing:
        raise ValueError(f"Trainable classes missing from binary targets: {missing}. "
                         "Check dictionary parsing and target generation.")
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    dictionary_target_audit(class_dict, y_full_target).to_csv(
        config.reports_dir / "dictionary_target_audit.csv", index=False)

    # Auditable binary class matrix actually supplied to the models.
    # Names and SMILES remain in their own columns; the only new target
    # columns are prefixed label__ to avoid collisions with source data.
    id_cols = [c for c in (config.id_column, config.name_column, config.smiles_column)
               if c in compounds.columns]
    export_y = y_full_target.add_prefix("label__").reset_index(drop=True)
    export_compounds = pd.concat([compounds[id_cols].reset_index(drop=True), export_y], axis=1)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    export_compounds.to_csv(config.reports_dir / "training_ground_truth_binary.csv", index=False)
    logger.info("Saved %d x %d binary training targets to %s", len(y_full_target),
                y_full_target.shape[1], config.reports_dir / "training_ground_truth_binary.csv")

    smiles = compounds[config.smiles_column].tolist()
    feat_selection_target = labels

    # 3. Train+validation / test split: iterative multi-label stratification.
    split_method = "multilabel"
    split_Y = y_full_target
    # Keep every compound in the *same order* as the reference experiment.
    # The archived fitted feature pipeline is also restored below, so
    # RFECV/SHAP/Boruta, imputation, scaling and PCA are not repeated.
    reuse = None
    if config.reuse_experiment_setup:
        reuse = load_experiment_setup(
            config, config.reuse_experiment_setup, compounds, labels,
            y_full_target, raw_features,
        )
        (idx_train_full, idx_test, saved_pos_train, saved_pos_val), saved_fp, setup_meta = reuse
        logger.info("Reusing experiment setup %s: %d features, %d compounds.",
                    setup_meta["id"], setup_meta["n_selected_descriptors"], len(compounds))
    else:
        idx_train_full, idx_test = stratified_split(
            len(compounds), config.test_size, config.random_state, labels, split_Y, split_method,
        )
    logger.info("Train/test split: %d / %d compounds (stratified on %s).", len(idx_train_full), len(idx_test),
                "every label")
    raw_train_full = raw_features.iloc[idx_train_full].reset_index(drop=True)
    raw_test = raw_features.iloc[idx_test].reset_index(drop=True)
    y_train_full_raw = y_full_target.iloc[idx_train_full].reset_index(drop=True)
    y_test_raw = y_full_target.iloc[idx_test].reset_index(drop=True)
    feat_sel_train_full = feat_selection_target.iloc[idx_train_full].reset_index(drop=True)
    smiles_train_full = compounds.iloc[idx_train_full][config.smiles_column].reset_index(drop=True)
    smiles_test = compounds.iloc[idx_test][config.smiles_column].tolist()

    # 4. Fit feature pipeline on the full training split, transform test ------
    progress(2)
    def _fit_features():
        fp = FeaturePipeline(config=config)
        # Multi-label mode: mutual information (selection + pre-filter) is
        # scored against every label, not class1 only.
        return fp, fp.fit_transform(raw_train_full, feat_sel_train_full,
                                    Y=y_train_full_raw)

    if reuse is not None:
        feat_pipeline = saved_fp
        X_train_full = feat_pipeline.transform(raw_train_full)
    else:
        feat_pipeline, X_train_full = cached("feature_pipeline", _fit_features)
    X_test = feat_pipeline.transform(raw_test)
    logger.info("Selected %d features after cleaning/variance/correlation/MI filtering.", X_train_full.shape[1])
    save_stage_counts(feat_pipeline.stage_counts_, config.reports_dir / "feature_funnel.json")

    # 4b. Inner train/validation split for candidate comparison, Optuna
    #     (when cv_folds < 2), threshold calibration, and the optional
    #     no-Optuna single-model validation report. The external held-out test
    #     split is never used in any of these steps.
    if reuse is not None:
        pos_train, pos_val = saved_pos_train, saved_pos_val
    else:
        pos_train, pos_val = stratified_split(
            len(X_train_full), config.validation_size, config.random_state, feat_sel_train_full,
            y_train_full_raw, split_method,
        )
    X_train = X_train_full.iloc[pos_train].reset_index(drop=True)
    X_val = X_train_full.iloc[pos_val].reset_index(drop=True)
    y_train_raw = y_train_full_raw.iloc[pos_train].reset_index(drop=True)
    y_val_raw = y_train_full_raw.iloc[pos_val].reset_index(drop=True)
    smiles_train = smiles_train_full.iloc[pos_train].tolist()
    smiles_val = smiles_train_full.iloc[pos_val].tolist()

    # Save the exact split assignment for each compound.
    split_record = pd.Series("test", index=range(len(compounds)), dtype=object)
    split_record.iloc[np.asarray(idx_train_full)[pos_train]] = "train"
    split_record.iloc[np.asarray(idx_train_full)[pos_val]] = "validation"
    pd.DataFrame({
        "id": compounds[config.id_column].values if config.id_column in compounds.columns else np.arange(1, len(compounds) + 1),
        "SMILES": compounds[config.smiles_column].values,
        "split": split_record.values,
    }).to_csv(config.reports_dir / "data_split.csv", index=False)
    # A reference setup is independent of the classifier and never gets
    # overwritten when the next classifier is trained. New runs archive the
    # exact ordered indices + already-fitted feature processing.
    active_setup_id = None
    if reuse is None and config.save_experiment_setup:
        saved = save_experiment_setup(
            config, compounds, labels, y_full_target, raw_features, feat_pipeline,
            idx_train_full, idx_test, pos_train, pos_val,
        )
        active_setup_id = saved["id"]
        logger.info("Archived reusable experiment setup: %s", saved["id"])
        (config.reports_dir / "experiment_setup.json").write_text(
            json.dumps({"setup_id": saved["id"], "reused": False}, indent=2), encoding="utf-8",
        )
    elif reuse is not None:
        active_setup_id = setup_meta["id"]
        (config.reports_dir / "experiment_setup.json").write_text(
            json.dumps({"setup_id": setup_meta["id"], "reused": True}, indent=2), encoding="utf-8",
        )
    else:
        (config.reports_dir / "experiment_setup.json").unlink(missing_ok=True)
    # Stratification key for the cross-validation folds inside the train
    # split (Round 1 + Optuna, config.cv_folds) -- class1, same as the splits above.
    strat_train = feat_sel_train_full.iloc[pos_train].to_numpy()

    # 5. Encode labels ----------------------------------------------------------
    class_names = multilabel_class_names
    y_train_full = y_train_full_raw.to_numpy()
    y_train = y_train_raw.to_numpy()
    y_val = y_val_raw.to_numpy()
    y_test = y_test_raw.to_numpy()
    label_encoder = class_names  # saved as the "label_encoder" artifact either way

    # Remove single-model validation/CV reports from earlier runs. In Single
    # model mode, 0/1 folds -> validation metrics; >=2 -> CV metrics on the
    # full train+validation (80%) partition, independent of Optuna.
    for old in (
        config.reports_dir / "validation_metrics.json",
        config.reports_dir / "validation_per_class_stats.csv",
        config.reports_dir / "validation_evaluation_info.json",
    ):
        old.unlink(missing_ok=True)
    for old in (
        config.reports_dir / "single_model_cv_metrics.json",
        config.reports_dir / "single_model_cv_folds.csv",
    ):
        old.unlink(missing_ok=True)
    for suffix in (
        "roc_curves", "roc_curves_per_class", "pr_curves",
        "pr_curves_per_class", "confusion_matrix",
    ):
        (config.plots_dir / f"validation_{suffix}.png").unlink(missing_ok=True)

    # 6. Candidate evaluation on the validation split ----------------------------
    #    `config.selected_model` (None by default) lets the user bypass the
    #    leaderboard entirely and train/evaluate exactly one tabular candidate.
    progress(3)
    single = config.selected_model
    all_candidates = build_candidate_models(config, n_classes=len(class_names))
    if all_candidates:
        # One independent binary classifier per label, same base estimators
        # for each category.
        all_candidates = wrap_models_for_multilabel(all_candidates, config)
    wrap_fn = lambda m: wrap_models_for_multilabel({"_": m}, config)["_"]

    if single is not None and single not in all_candidates:
        raise ValueError(
            f"Unknown selected_model {single!r}. Options: {list(all_candidates)}"
        )

    eval_candidates = (
        {single: all_candidates[single]} if single is not None else all_candidates
    )

    best_name = None
    per_candidate_tuned: dict = {}
    if single is not None:
        # Single model mode: no comparison of any kind -- only the chosen model
        # is trained, (tuned,) and evaluated. Delete a previous automatic run's
        # leaderboard so it can't be shown as if it belonged to this run.
        logger.info("Single model mode (%s): no model comparison -- only this model is trained and evaluated.",
                    single)
        val_results = pd.DataFrame(columns=["model", "val_f1_macro"])
        for stale in (config.reports_dir / "cv_model_comparison.csv", config.plots_dir / "model_comparison.png"):
            stale.unlink(missing_ok=True)
        best_name = single
    elif eval_candidates:
        if config.tune_all_candidates and config.use_optuna:
            logger.info(
                "tune_all_candidates=True: running a full Optuna search (%d trials each) on all %d "
                "enabled candidate(s) before selecting, instead of tuning only the default-hyperparameter "
                "winner.", config.n_optuna_trials, len(eval_candidates),
            )
            rows = []
            for name in eval_candidates:
                tuned_i, study_i = tune_best_model(
                    name, X_train, y_train, X_val, y_val, config, wrap_fn=wrap_fn,
                    storage=ckpt.optuna_storage(), study_name=ckpt.study_name(f"round1_{name}"),
                    cv_strat=strat_train,
                )
                per_candidate_tuned[name] = tuned_i
                # study.best_value is that trial's already-computed validation
                # macro-F1 -- reuse it rather than re-fit/re-score an
                # unfitted model (tune_best_model returns best_model
                # unfitted; the real fit-for-deployment happens below).
                rows.append({"model": name, "val_f1_macro": study_i.best_value})
                logger.info("%s (tuned): validation f1_macro = %.4f", name, study_i.best_value)
            val_results = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
        else:
            val_results = evaluate_candidates_on_validation(
                eval_candidates, X_train, y_train, X_val, y_val, checkpoint=ckpt,
                cv_folds=config.cv_folds, cv_strat=strat_train, random_state=config.random_state,
            )
        plot_model_comparison(val_results, config.plots_dir / "model_comparison.png")
        val_results.to_csv(config.reports_dir / "cv_model_comparison.csv", index=False)
        logger.info("Validation leaderboard:\n%s", val_results.to_string(index=False))
        best_name = val_results.iloc[0]["model"]
        logger.info("Best candidate by validation macro-F1: %s", best_name)
    else:
        raise ValueError("Enable at least one tabular classifier.")

    # 7. Optuna tuning of the winning/selected model family (skipped if
    #    tune_all_candidates above already tuned it), then a final refit on
    #    train+validation combined ---------------------------------
    progress(4)
    tuned_model = None
    if best_name is not None:
        if best_name in per_candidate_tuned:
            tuned_model = per_candidate_tuned[best_name]
            logger.info("%s was already Optuna-tuned while comparing all candidates; reusing that fit.", best_name)
        elif config.use_optuna:
            tuned_model, study = tune_best_model(
                best_name, X_train, y_train, X_val, y_val, config, wrap_fn=wrap_fn,
                storage=ckpt.optuna_storage(), study_name=ckpt.study_name(f"tune_{best_name}"),
                cv_strat=strat_train,
            )
            logger.info("Optuna best %s macro-F1: %.4f (params: %s)",
                        f"{config.cv_folds}-fold CV" if config.cv_folds >= 2 else "validation",
                        study.best_value, study.best_params)
        else:
            tuned_model = all_candidates[best_name]
        # Automatic mode: fit on train only, so the finalist comparison in 8b
        # scores every contender on validation compounds it has never seen;
        # the winner is refit on train+validation afterwards (8c'). Single-
        # model mode has nothing to compare -- fit on train+validation directly.
        _unfitted_tuned = tuned_model
        if single is None:
            tuned_model = cached(f"fit_tuned_{best_name}", lambda: _unfitted_tuned.fit(X_train, y_train))
        else:
            tuned_model = cached(f"fit_tuned_{best_name}", lambda: _unfitted_tuned.fit(X_train_full, y_train_full))

    # 8. Finalize the selected tabular candidate. Keep its train-only fit
    # for validation-based threshold calibration, then refit on train+validation.
    progress(5)
    final_name = f"tuned_{best_name}"
    calib_model_obj = None
    (config.reports_dir / "final_model_comparison.csv").unlink(missing_ok=True)
    if single is None:
        val_proba = _proba(tuned_model, X_val)
        val_f1 = f1_score(y_val, _pred_from_proba(val_proba, config), average="macro")
        logger.info("Validation macro-F1 for %s: %.4f", final_name, val_f1)
        calib_model_obj = tuned_model
        tuned_model = cached(f"refit_{final_name}", lambda: clone(calib_model_obj).fit(X_train_full, y_train_full))
        final_proba = _proba(tuned_model, X_test)
        logger.info("Refit %s on train+validation for the held-out test evaluation.", final_name)
    else:
        final_proba = _proba(tuned_model, X_test)
        if config.calibrate_multilabel_thresholds:
            calib_model_obj = cached(f"calib_fit_{final_name}", lambda: clone(tuned_model).fit(X_train, y_train))
            logger.info("Fit an extra train-only %s for per-label threshold calibration.", final_name)
    logger.info("Final selected model: %s", final_name)
    train_model_obj = tuned_model

    #     model's probabilities into predicted labels -- see
    #     calibrate_per_label_thresholds. A single global 0.5 cutoff
    #     systematically under-predicts rare labels; this is a free,
    #     model-unchanged F1 gain.
    per_label_thresholds = None
    threshold_map = None
    if config.calibrate_multilabel_thresholds:
        # A train-only fit (see 8c'), so thresholds are tuned on validation
        # compounds the model never saw.
        calib_obj = calib_model_obj if calib_model_obj is not None else train_model_obj
        final_proba_val = _proba(calib_obj, X_val)
        threshold_map = calibrate_per_label_thresholds(y_val, final_proba_val, class_names)
        per_label_thresholds = thresholds_to_array(threshold_map, class_names)
        with open(config.reports_dir / "multilabel_thresholds.json", "w") as fh:
            json.dump(threshold_map, fh, indent=2)

    # Single-model validation reporting is the 0/1-fold alternative to CV.
    # Fit on the inner training subset and evaluate on its held-out validation
    # portion. Do not alter the final train+validation model or test evaluation.
    # With Optuna and 0/1 folds, the validation set was also used for tuning,
    # so this descriptive validation score is not an independent estimate.
    validation_stats = None
    if single is not None and config.cv_folds < 2:
        if calib_model_obj is not None:
            # Multi-label threshold calibration already trained a copy on
            # X_train and produced out-of-training-set validation probabilities.
            # Reuse it; do not fit or calibrate anything again.
            validation_proba = final_proba_val
        else:
            validation_model = clone(tuned_model).fit(X_train, y_train)
            validation_proba = _proba(validation_model, X_val)
        validation_pred = _pred_from_proba(validation_proba, config, per_label_thresholds)
        validation_stats = (
            full_evaluation_multilabel(
                y_val, validation_pred, validation_proba, class_names, config,
                prefix="validation",
            )
        )
        validation_info = {
            "trained_on": "train_only",
            "evaluated_on": "validation",
            "optuna_used": bool(config.use_optuna),
            "cv_folds_used": 0,  # This validation report uses a single held-out split.
            "hyperparameters_selected_on_validation": bool(config.use_optuna),
            "thresholds_optimized_on_validation": bool(
                config.calibrate_multilabel_thresholds
            ),
            "feature_pipeline_fitted_on": "train_plus_validation",
        }
        (config.reports_dir / "validation_evaluation_info.json").write_text(
            json.dumps(validation_info, indent=2), encoding="utf-8"
        )
        logger.info(
            "Single-model validation (CV disabled): macro-F1=%.4f, macro PR-AUC=%s, "
            "micro PR-AUC=%s (train-only fit; final test model unchanged).",
            validation_stats["macro_f1"],
            validation_stats["pr_auc"].get("macro"),
            validation_stats["pr_auc"].get("micro"),
        )

    # Standalone CV *reporting* for Single model (with or without Optuna).
    # Fold assignments are drawn from the FULL train+validation (80%) part;
    # the held-out test set is never included. This evaluation does not tune
    # hyperparameters, calibrate per-label thresholds, or change the final fit.
    # Its fold-level predictions use the fixed global threshold (multilabel),
    # rather than thresholds derived from the inner validation split.
    cv_single_stats = None
    if single is not None and config.cv_folds >= 2:
        from .cv_reporting import evaluate_single_model_cv
        logger.info("Reporting %d-fold CV on train+validation for single model %s (Optuna=%s).",
                    config.cv_folds, final_name, config.use_optuna)
        _, cv_single_stats = evaluate_single_model_cv(
            train_model_obj,
            X_train_full, y_train_full, class_names, config,
            cv_strat=feat_sel_train_full.to_numpy(),
        )
        cv_path = config.reports_dir / "single_model_cv_metrics.json"
        cv_path.write_text(json.dumps(cv_single_stats, indent=2), encoding="utf-8")
        pd.DataFrame(cv_single_stats["folds"]).to_csv(
            config.reports_dir / "single_model_cv_folds.csv", index=False
        )

    y_pred = _pred_from_proba(final_proba, config, per_label_thresholds)

    # 9. Full evaluation on held-out test set -----------------------------------
    progress(6)
    stats = full_evaluation_multilabel(y_test, y_pred, final_proba, class_names, config, prefix="test")
    logger.info(
        "Held-out test -- subset_accuracy=%.4f hamming_loss=%.4f macro_f1=%.4f micro_f1=%.4f macro_ROC_AUC=%.4f macro_PR_AUC=%.4f "
        "avg_labels/compound: true=%.2f pred=%.2f",
        stats["subset_accuracy"], stats["hamming_loss"], stats["macro_f1"], stats["micro_f1"],
        stats["roc_auc"]["macro"], stats["pr_auc"]["macro"], stats["avg_labels_per_compound_true"], stats["avg_labels_per_compound_pred"],
    )

    # 9b. Same evaluation, scored on the training split instead of test --
    #     the same fitted tabular model applied to
    #     the data it was actually trained on. Expected to look better than
    #     test; the gap between the two is the standard overfitting check.
    final_proba_train = _proba(train_model_obj, X_train_full)
    y_pred_train = _pred_from_proba(final_proba_train, config, per_label_thresholds)

    train_stats = full_evaluation_multilabel(
        y_train_full, y_pred_train, final_proba_train, class_names, config, prefix="train"
    )
    logger.info(
        "Training set -- subset_accuracy=%.4f hamming_loss=%.4f macro_f1=%.4f micro_f1=%.4f macro_ROC_AUC=%.4f macro_PR_AUC=%.4f",
        train_stats["subset_accuracy"], train_stats["hamming_loss"], train_stats["macro_f1"],
        train_stats["micro_f1"], train_stats["roc_auc"]["macro"], train_stats["pr_auc"]["macro"],
    )

    # 10. Applicability domain fit on the full training split, UC assignment
    #     demo on TEST ------------------------------------------------------
    ad = ApplicabilityDomain(config=config)
    ad.fit(X_train_full)
    in_domain_test = ad.in_domain(X_test)
    labeled_test = assign_multilabel_with_uc(final_proba, class_names, in_domain_test, config, thresholds=per_label_thresholds)
    labeled_test["true_class"] = _labels_to_string(y_test_raw, class_names).values
    labeled_test.insert(0, "SMILES", compounds.iloc[idx_test][config.smiles_column].values)
    labeled_test.to_csv(config.reports_dir / "test_predictions_with_UC.csv", index=False)

    # 10b. Persist an independent, leakage-safe snapshot for the new multilabel
    # post-prediction Label Audit. This snapshot captures the ORIGINAL test
    # predictions before the model/AD is refitted on all available compounds.
    # ARM/KNN may use the train+validation references below, but must never
    # use y_test_raw or the production full-dataset reference as audit evidence.
    from .heldout_label_audit import save_heldout_audit_snapshot

    heldout_snapshot_path = save_heldout_audit_snapshot(
        config.reports_dir,
        class_names=class_names,
        train_smiles=smiles_train_full.tolist(), test_smiles=smiles_test,
        y_train=y_train_full, y_test=y_test,
        probabilities=final_proba, predictions=y_pred,
        thresholds=(per_label_thresholds if per_label_thresholds is not None
                    else config.multilabel_threshold),
        train_descriptors=X_train_full.to_numpy(),
        test_descriptors=X_test.to_numpy(),
        in_domain=in_domain_test, ad_mean_distances=ad.mean_distance(X_test),
        ad_cutoff=ad.cutoff_, model_name=final_name,
        fingerprint_radius=config.morgan_radius,
        fingerprint_n_bits=config.morgan_n_bits,
        fingerprint_chirality=config.morgan_use_chirality,
        train_names=(compounds.iloc[idx_train_full][config.name_column].tolist()
                     if config.name_column in compounds else None),
        test_names=(compounds.iloc[idx_test][config.name_column].tolist()
                    if config.name_column in compounds else None),
        train_ids=(compounds.iloc[idx_train_full][config.id_column].tolist()
                   if config.id_column in compounds else None),
        test_ids=(compounds.iloc[idx_test][config.id_column].tolist()
                  if config.id_column in compounds else None),
    )
    logger.info('Saved held-out Label Audit inputs (training-only reference, original '
                'test predictions and transformed features): %s', heldout_snapshot_path)

    logger.info(
        "UC rate on held-out test set: %.1f%% (%d/%d)",
        (labeled_test["final_label"] == UC_LABEL).mean() * 100,
        (labeled_test["final_label"] == UC_LABEL).sum(),
        len(labeled_test),
    )

    # 11. Refit final model + AD on the FULL dataset (production artifact used
    #     to label every compound, incl. those held out for evaluation above).
    #     The feature pipeline is NOT refit: the production model reuses the
    #     exact cleaning/selection/scaling fit in step 4 on the training split,
    #     so it uses the same features as the model the test metrics describe
    #     (re-running a selector like Boruta on all data can pick a noticeably
    #     different feature set that was never evaluated).
    full_feat_pipeline = feat_pipeline
    X_full = full_feat_pipeline.transform(raw_features)
    logger.info("Production refit reuses the step-4 feature pipeline (%d features).", X_full.shape[1])
    y_full, full_label_encoder = y_full_target.to_numpy(), class_names

    # 11b. Chemical-space plots (PCA/t-SNE/UMAP) over the full dataset, colored
    #      by odor class and labeled with each compound's object number.
    #      Multilabel compounds are colored by their primary (class1) note --
    #      a scatter point can only show one color.
    progress(7)
    point_ids = compounds[config.id_column] if config.id_column in compounds.columns else pd.Series(range(1, len(compounds) + 1))
    names_series = compounds[config.name_column] if config.name_column in compounds.columns else None
    generate_chemical_space_plots(X_full, labels, point_ids, names_series, smiles=pd.Series(smiles), config=config)

    # 12. Refit production model + AD + persist all artifacts --------------------
    progress(8)
    def _fit_production():
        model = clone(tuned_model)
        model.fit(X_full, y_full)
        return model

    production_model = cached("production_model", _fit_production)
    full_proba = _proba(production_model, X_full)

    # The applicability domain is descriptor-space based.
    full_ad = ApplicabilityDomain(config=config)
    full_ad.fit(X_full)
    # Audit integrity marker: the saved full-dataset AD feature matrix must be
    # paired with training_ground_truth_binary.csv in precisely this order.
    import hashlib
    full_ad.training_smiles_sha256_ = hashlib.sha256(
        '\n'.join(str(s) for s in smiles).encode('utf-8')).hexdigest()
    in_domain_full = full_ad.in_domain(X_full)
    labeled_full = assign_multilabel_with_uc(full_proba, class_names, in_domain_full, config, thresholds=per_label_thresholds)
    labeled_full["rule_based_classes"] = _labels_to_string(y_full_target, class_names).values
    labeled_full.insert(0, "SMILES", smiles)
    labeled_full.insert(1, "Name", compounds[config.name_column].values if config.name_column in compounds.columns else "")
    labeled_full.to_excel(config.reports_dir / "all_compounds_labeled.xlsx", index=False)
    logger.info(
        "UC rate across full dataset (production model): %.1f%% (%d/%d)",
        (labeled_full["final_label"] == UC_LABEL).mean() * 100,
        (labeled_full["final_label"] == UC_LABEL).sum(),
        len(labeled_full),
    )

    # 13. Persist artifacts --------------------------------------------------------
    joblib.dump(production_model, config.models_dir / "production_model.joblib")
    full_feat_pipeline.save(str(config.models_dir / "feature_pipeline.joblib"))
    full_ad.save(str(config.models_dir / "applicability_domain.joblib"))
    joblib.dump(full_label_encoder, config.models_dir / "label_encoder.joblib")
    # Persist calibrated per-label thresholds for prediction.
    joblib.dump(per_label_thresholds, config.models_dir / "multilabel_thresholds.joblib")

    # 13b. Optional post-prediction held-out Label Audit. All required
    # inputs were frozen in step 10b. The production model and reference
    # are NEVER used by this audit. The separate held-out true labels are
    # read only after recommendations have been generated and saved.
    heldout_audit_results = None
    if config.run_heldout_label_audit:
        from .heldout_label_audit import run_saved_heldout_audit
        try:
            heldout_audit_results = run_saved_heldout_audit(
                config.reports_dir, knn_representations=config.heldout_audit_knn_modes)
            for mode, audit_result in heldout_audit_results.items():
                logger.info('Held-out audit [%s]: %d test compounds; %d review labels; %d FP-Growth rules.',
                            mode, len(audit_result.compounds), len(audit_result.suggestions),
                            len(audit_result.rules))
        except Exception as exc:
            # An auxiliary audit failure must not destroy a successfully
            # trained production model. Report a visible actionable error.
            logger.exception('Held-out label audit failed; model artifacts are still saved.')
            audit_dir = config.reports_dir / 'heldout_label_audit'
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / 'AUDIT_ERROR.txt').write_text(
                f'{type(exc).__name__}: {exc}\n'
                'Rerun: python scripts/run_heldout_label_audit.py --knn-representation both\n',
                encoding='utf-8')

    # Retain a side-by-side score history for classifiers evaluated with
    # this exact split/feature setup. Current reports/models may be overwritten
    # by the next run, but the experiment archive stays in place.
    if active_setup_id:
        try:
            comparison_path = record_classifier_result(config, active_setup_id, final_name, stats, train_stats, validation_stats, cv_single_stats)
            logger.info("Appended classifier scorecard to %s", comparison_path)
        except Exception as exc:
            logger.warning("Unable to append experiment scorecard: %s", exc)

    # Run completed -- checkpoints are only for resuming an interrupted run.
    ckpt.clear()
    logger.info("Run completed; checkpoints cleared.")

    return {
        "cv_results": val_results,
        "test_stats": stats,
        "train_stats": train_stats,
        "validation_stats": validation_stats,
        "single_model_cv_stats": cv_single_stats,
        "final_model_name": final_name,
        "labeled_test": labeled_test,
        "labeled_full": labeled_full,
        "class_names": class_names,
        "multilabel_thresholds": threshold_map,
        "heldout_audit_results": heldout_audit_results,
    }
