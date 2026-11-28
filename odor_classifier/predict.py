"""Multi-label predictions with an applicability-domain UC fallback."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .applicability_domain import ApplicabilityDomain
from .config import Config, DEFAULT_CONFIG, UC_LABEL
from .features import DESCRIPTOR_PREFIXES, FeaturePipeline, compute_raw_features


def _descriptor_config_for_pipeline(feature_pipeline: FeaturePipeline, config: Config) -> Config:
    """Overrides the descriptor-family toggles to match exactly which
    descriptor blocks ``feature_pipeline`` was actually fit on (detected
    from its ``kept_after_variance_`` column-name prefixes), rather than
    trusting the caller's ``config``, which may differ from the fitted
    model configuration. This both keeps
    predict-time features aligned with what the model expects (a family
    the model needs but the caller's config has off would otherwise turn
    into an all-NaN, median-imputed block -- silently worse, not broken)
    and skips computing any block the model doesn't use at all.
    """
    from dataclasses import replace

    present_cols = feature_pipeline.kept_after_variance_
    if not present_cols:
        return config
    overrides = {
        field_name: any(c.startswith(prefix) for c in present_cols)
        for field_name, prefix in DESCRIPTOR_PREFIXES.items()
    }
    # Morgan bits depend on how they were generated, not just on whether the
    # block is present -- compute them exactly as for the training data. Read
    # from the saved config's own values (vars), not getattr: a pipeline saved
    # before morgan_use_chirality existed would otherwise report the CURRENT
    # class default instead of what it was actually trained with (achiral).
    trained_config = getattr(feature_pipeline, "config", None)
    trained = vars(trained_config) if hasattr(trained_config, "__dict__") else {}
    overrides["morgan_use_chirality"] = trained.get("morgan_use_chirality", False)
    for field_name in ("morgan_radius", "morgan_n_bits"):
        if field_name in trained:
            overrides[field_name] = trained[field_name]
    return replace(config, **overrides)


def assign_multilabel_with_uc(
    proba: np.ndarray,
    class_names: List[str],
    in_domain: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Keep each label clearing its cutoff; return UC when outside the
    applicability domain or when no label passes its cutoff.

    ``thresholds``, if given, is a per-label array (see
    ``multilabel.calibrate_per_label_thresholds``) -- otherwise every label
    uses the single ``config.multilabel_threshold`` cutoff.
    """
    cutoffs = thresholds if thresholds is not None else config.multilabel_threshold
    passes = proba >= cutoffs
    predicted_labels = [
        [class_names[i] for i in np.where(row)[0]] for row in passes
    ]
    max_conf = proba.max(axis=1)

    final_labels = []
    for labels, domain_ok in zip(predicted_labels, in_domain):
        final_labels.append(labels if (labels and domain_ok) else [UC_LABEL])

    return pd.DataFrame({
        "predicted_labels": [", ".join(lbls) for lbls in predicted_labels],
        "confidence": max_conf,
        "in_domain": in_domain,
        "final_label": [", ".join(lbls) for lbls in final_labels],
        "is_uc": [lbls == [UC_LABEL] for lbls in final_labels],
    })


def predict_smiles(
    smiles_list: List[str],
    model,
    feature_pipeline: FeaturePipeline,
    label_encoder,
    ad: ApplicabilityDomain,
    config: Config = DEFAULT_CONFIG,
    thresholds: Optional[np.ndarray] = None,
    include_audit_data: bool = False,
    return_transformed_features: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end prediction for a list of raw SMILES strings.

    ``model`` is a tabular multi-label classifier; ``label_encoder`` is its list of class names.

    ``return_transformed_features``: return ``(predictions, X)`` where X is
    the selected, training-scaled (and optionally PCA-reduced) matrix already
    passed to the model and applicability domain. Useful for descriptor KNN
    without recomputing descriptors. Default preserves the original API.

    ``thresholds``, multi-label mode only: per-label cutoffs saved by the
    training run (``multilabel_thresholds.joblib``, see
    ``multilabel.calibrate_per_label_thresholds``). Falls back to the flat
    ``config.multilabel_threshold`` if not given.
    """
    from dataclasses import replace

    from sklearn.multioutput import MultiOutputClassifier

    from .multilabel import multioutput_predict_proba

    # Retain single-process Mordred prediction for compatibility.
    if config.mordred_n_proc != 1:
        config = replace(config, mordred_n_proc=1)
    config = _descriptor_config_for_pipeline(feature_pipeline, config)

    raw = compute_raw_features(smiles_list, config)
    X = feature_pipeline.transform(raw)
    if include_audit_data:
        ad_distances = ad.mean_distance(X)
        in_domain = ad_distances <= ad.cutoff_
    else:
        ad_distances = None
        in_domain = ad.in_domain(X)

    if not isinstance(label_encoder, (list, tuple)):
        raise ValueError("Only multi-label production models are supported.")
    if isinstance(model, MultiOutputClassifier):
        proba = multioutput_predict_proba(model, X)
    else:
        proba = model.predict_proba(X)
    result = assign_multilabel_with_uc(proba, list(label_encoder), in_domain, config, thresholds=thresholds)
    result.insert(0, "SMILES", smiles_list)
    if include_audit_data:
        # The single 'confidence' field is only the maximum class probability.
        # A multilabel audit requires ALL class scores, including labels the
        # model did not predict.  Keep these optional so the basic API and
        # exported prediction format remain unchanged by default.
        result['ad_mean_distance'] = ad_distances
        for col, values in zip(list(label_encoder), proba.T):
            result['proba__' + str(col)] = values
    return (result, X) if return_transformed_features else result
