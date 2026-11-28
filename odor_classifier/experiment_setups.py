"""Persist and strictly validate reusable train/validation/test + feature setups.

An experiment setup is a *fitted* feature preprocessor and the exact ordered
row indices used for training, validation and testing.  It is independent of
which classifier is later fitted.  The input data and descriptor settings are
checked before reuse, to avoid silent mixing of incompatible experiments.

Setups are stored outside reports/models, which the next training run may
replace.  Only open setups created locally by a trusted OdorClassifier installation:
joblib files are Python pickles and must not be loaded from untrusted sources.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import joblib
import numpy as np
import pandas as pd


# Options which affect the RAW descriptors (not the subsequent classifier).
DESCRIPTOR_FIELDS = (
    "use_rdkit_descriptors", "use_morgan_fingerprints", "morgan_radius",
    "morgan_n_bits", "morgan_use_chirality", "use_maccs", "use_mordred",
    "mordred_n_proc", "use_mol2vec", "mol2vec_radius",
    "use_field_descriptors", "use_grind_descriptors", "grind_grid_margin",
    "grind_grid_spacing", "grind_max_nodes_per_probe", "grind_min_node_distance",
    "grind_bin_width", "grind_n_bins", "mol2vec_model_path", "random_state",
)


def setups_dir(config) -> Path:
    return Path(config.outputs_dir) / "experiment_setups"


def list_experiment_setups(config) -> list[dict]:
    """Most-recent-first manifests (excluding incomplete/corrupt directories)."""
    base = setups_dir(config)
    manifests = []
    if not base.exists():
        return manifests
    for path in base.glob("*/manifest.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if (entry.get("version") == 1 and entry.get("id") == path.parent.name
                    and (path.parent / "feature_pipeline.joblib").is_file()
                    and (path.parent / "indices.json").is_file()
                    and (path.parent / "data_split.csv").is_file()):
                manifests.append(entry)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(manifests, key=lambda item: item.get("created_utc", ""), reverse=True)


def _setup_path(config, setup_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", setup_id or ""):
        raise ValueError("Invalid experiment setup identifier.")
    return setups_dir(config) / setup_id


def _row_identity(compounds, labels, Y, config) -> str:
    """Hash the full ordered analysis rows and ground-truth targets.

    Duplicate IDs/SMILES remain distinct because row order is included.
    Changing even one target or removing/reordering a compound invalidates
    reuse, rather than accidentally evaluating on a different test set.
    """
    ids = (compounds[config.id_column].tolist() if config.id_column in compounds
           else list(range(1, len(compounds) + 1)))
    smiles = compounds[config.smiles_column].tolist()
    targets = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame({"target": Y})
    records = {
        "ids": [str(v) for v in ids],
        "smiles": [str(v) for v in smiles],
        "primary_labels": [str(v) for v in labels],
        "target_names": list(targets.columns.astype(str)),
        "targets": targets.astype(str).values.tolist(),
    }
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _descriptor_settings(config) -> dict:
    return {name: str(getattr(config, name)) for name in DESCRIPTOR_FIELDS}


def _raw_feature_hash(raw_features: pd.DataFrame) -> str:
    """Ensure reused features have identical VALUES, not just column names.

    This protects comparisons against changes in RDKit/Mordred versions or
    regenerated 3D conformers (even when the settings look identical).
    """
    h = hashlib.sha256()
    h.update(json.dumps(list(raw_features.columns.astype(str)), ensure_ascii=False).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(raw_features, index=False).to_numpy(dtype=np.uint64).tobytes())
    return h.hexdigest()


def _validate_indices(index_info: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        full, test, train, val = (
            np.asarray(index_info[name], dtype=int)
            for name in ("idx_train_full", "idx_test", "pos_train", "pos_val")
        )
        if not all(len(a) and a.ndim == 1 for a in (full, test, train, val)):
            raise ValueError("Empty or malformed split")
        if (len(full) + len(test) != n
                or sorted(np.r_[full, test].tolist()) != list(range(n))
                or sorted(np.r_[train, val].tolist()) != list(range(len(full)))):
            raise ValueError("Saved split indices are not a complete, disjoint partition")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid saved experiment split: {exc}") from exc
    return full, test, train, val


def _assignment(n, full, test, train, val) -> pd.Series:
    result = pd.Series("test", index=range(n), dtype=object, name="split")
    result.iloc[full[train]] = "train"
    result.iloc[full[val]] = "validation"
    return result


def save_experiment_setup(config, compounds, labels, Y, raw_features, feature_pipeline,
                          idx_train_full, idx_test, pos_train, pos_val) -> dict:
    """Archive the exact fitted feature pipeline + 3-way ordered row indices."""
    full, test, train, val = _validate_indices({
        "idx_train_full": list(idx_train_full), "idx_test": list(idx_test),
        "pos_train": list(pos_train), "pos_val": list(pos_val),
    }, len(compounds))
    ident = datetime.now(timezone.utc)
    setup_id = ident.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    folder = _setup_path(config, setup_id)
    folder.mkdir(parents=True, exist_ok=False)
    assignment = _assignment(len(compounds), full, test, train, val)
    manifest = {
        "version": 1, "id": setup_id, "created_utc": ident.isoformat(timespec="seconds"),
        "row_identity_sha256": _row_identity(compounds, labels, Y, config),
        "raw_feature_sha256": _raw_feature_hash(raw_features),
        "descriptor_settings": _descriptor_settings(config),
        "multilabel": bool(config.multilabel),
        "selection_method": config.selection_method,
        "n_compounds": len(compounds),
        "n_selected_descriptors": len(feature_pipeline.selected_features_),
        "selected_descriptors": list(feature_pipeline.selected_features_),
        "model_input_features": (list(feature_pipeline.selected_features_)
                                 if feature_pipeline.pca_ is None
                                 else [f"pc_{i+1}" for i in range(feature_pipeline.pca_.n_components_)]),
        "uses_pca": feature_pipeline.pca_ is not None,
        "split_counts": {split: int((assignment == split).sum())
                         for split in ("train", "validation", "test")},
        "seed": config.random_state,
    }
    row_ids = (compounds[config.id_column].values if config.id_column in compounds
               else np.arange(1, len(compounds) + 1))
    pd.DataFrame({"id": row_ids, "SMILES": compounds[config.smiles_column].values,
                  "split": assignment.values}).to_csv(folder / "data_split.csv", index=False)
    (folder / "indices.json").write_text(json.dumps({
        "idx_train_full": full.tolist(), "idx_test": test.tolist(),
        "pos_train": train.tolist(), "pos_val": val.tolist(),
    }), encoding="utf-8")
    joblib.dump(feature_pipeline, folder / "feature_pipeline.joblib")
    # Write manifest LAST so a partial save never appears as a usable setup.
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_setup_descriptor_settings(config, setup_id: str) -> dict:
    """Cheap validation BEFORE descriptor computation (can take hours)."""
    folder = _setup_path(config, setup_id)
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read experiment setup {setup_id!r}: {exc}") from exc
    if manifest.get("id") != setup_id or manifest.get("version") != 1:
        raise ValueError("Experiment setup manifest is invalid or incompatible.")
    saved_settings = manifest.get("descriptor_settings")
    current_settings = _descriptor_settings(config)
    if not isinstance(saved_settings, dict) or saved_settings != current_settings:
        changed = [name for name in DESCRIPTOR_FIELDS
                   if not isinstance(saved_settings, dict)
                   or saved_settings.get(name) != current_settings[name]]
        raise ValueError(
            "Raw descriptor settings differ from those in the saved setup "
            f"({', '.join(changed)}). Restore those settings or create a new setup."
        )
    return manifest


def load_experiment_setup(config, setup_id: str, compounds, labels, Y, raw_features):
    """Return saved split indices and fitted pipeline, or fail explicitly.

    Data/target changes or descriptor-generation setting changes invalidate a
    saved setup. Classifier/model tuning settings may change freely.
    """
    folder = _setup_path(config, setup_id)
    manifest = validate_setup_descriptor_settings(config, setup_id)
    try:
        indices = json.loads((folder / "indices.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read experiment setup indices {setup_id!r}: {exc}") from exc
    if manifest.get("row_identity_sha256") != _row_identity(compounds, labels, Y, config):
        raise ValueError(
            "The saved experiment setup was fitted on different compounds, row order or labels. "
            "Restore the original dataset and label settings, or create a new setup."
        )
    if manifest.get("raw_feature_sha256") != _raw_feature_hash(raw_features):
        raise ValueError(
            "The raw molecular descriptor values or columns differ from the archived setup. "
            "Restore the original descriptor versions/cache, or create a new setup."
        )
    full, test, train, val = _validate_indices(indices, len(compounds))
    assignment = _assignment(len(compounds), full, test, train, val)
    saved_split = pd.read_csv(folder / "data_split.csv", dtype={"SMILES": "string"})
    if saved_split["split"].tolist() != assignment.tolist():
        raise ValueError("Saved experiment setup has inconsistent split assignments.")
    # This pickle is opened only after checking provenance and dataset/settings.
    pipeline = joblib.load(folder / "feature_pipeline.joblib")
    missing = sorted(set(pipeline.kept_after_variance_) - set(raw_features.columns))
    if missing:
        raise ValueError(f"The current descriptor matrix is missing {len(missing)} required columns, "
                         f"e.g. {missing[:4]}. Restore descriptor settings or start fresh.")
    if list(pipeline.selected_features_) != manifest["selected_descriptors"]:
        raise ValueError("Selected descriptors disagree with the archived feature pipeline.")
    return (full, test, train, val), pipeline, manifest


def record_classifier_result(config, setup_id: str, model_name: str,
                             test_stats: dict, train_stats: dict,
                             validation_stats: dict | None = None,
                             cv_stats: dict | None = None) -> Path:
    """Append a concise comparable scorecard to a persistent setup archive."""
    folder = _setup_path(config, setup_id)
    if not (folder / "manifest.json").is_file():
        raise ValueError(f"Unknown experiment setup {setup_id!r}")

    def nested(stats, name, measure):
        return stats.get(name, {}).get(measure, float("nan"))

    validation_stats = validation_stats or {}
    cv_stats = cv_stats or {}
    cv_mean = cv_stats.get("mean", {})
    cv_std = cv_stats.get("std", {})
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setup_id": setup_id,
        "classifier": model_name,
        "selection_mode": config.selected_model or "automatic",
        "cv_folds": config.cv_folds,
        "optuna_enabled": config.use_optuna,
        "optuna_trials": config.n_optuna_trials if config.use_optuna else 0,
        "per_label_feature_selection": config.per_label_feature_selection,
        "test_macro_f1": test_stats.get("macro_f1", float("nan")),
        "test_micro_f1": test_stats.get("micro_f1", float("nan")),
        "test_macro_pr_auc": nested(test_stats, "pr_auc", "macro"),
        "test_micro_pr_auc": nested(test_stats, "pr_auc", "micro"),
        "test_macro_roc_auc": nested(test_stats, "roc_auc", "macro"),
        "test_accuracy_or_subset_accuracy": test_stats.get(
            "subset_accuracy", test_stats.get("accuracy", float("nan"))),
        "train_macro_f1": train_stats.get("macro_f1", float("nan")),
        "validation_macro_f1": validation_stats.get("macro_f1", float("nan")),
        "validation_micro_f1": validation_stats.get("micro_f1", float("nan")),
        "validation_macro_pr_auc": nested(validation_stats, "pr_auc", "macro"),
        "validation_micro_pr_auc": nested(validation_stats, "pr_auc", "micro"),
        "validation_macro_roc_auc": nested(validation_stats, "roc_auc", "macro"),
        "reported_cv_folds": cv_stats.get("cv_folds", 0),
        **{f"cv_{metric}_mean": cv_mean.get(metric, float("nan")) for metric in (
            "macro_f1", "micro_f1", "macro_pr_auc", "micro_pr_auc", "macro_roc_auc")},
        **{f"cv_{metric}_std": cv_std.get(metric, float("nan")) for metric in (
            "macro_f1", "micro_f1", "macro_pr_auc", "micro_pr_auc", "macro_roc_auc")},
    }
    path = folder / "classifier_comparison.csv"
    if path.is_file():
        existing = pd.read_csv(path)
        history = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])
    tmp = path.with_suffix(".csv.tmp")
    history.to_csv(tmp, index=False)
    tmp.replace(path)
    return path
