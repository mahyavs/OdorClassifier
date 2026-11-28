"""Leakage-safe post-prediction audit of the *internal held-out test split*.

The audit uses only train+validation as the structural/ARM reference and the
classifier that was evaluated on the held-out test split. Production artifacts
(refit on every compound) must NEVER be substituted for this snapshot.

The held-out ground truth is saved separately and is opened only *after* the
recommendations are generated, solely to score their agreement with the labels.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Optional
import shutil

import numpy as np
import pandas as pd

from .prediction_label_audit import (
    AuditOptions, TrainingReference, audit_predictions, compare_knn_audits,
    mine_label_rules,
)

INPUT_FOLDER = 'heldout_label_audit_input'
OUTPUT_FOLDER = 'heldout_label_audit'


def _smiles_hash(values) -> str:
    return hashlib.sha256('\n'.join(str(x) for x in values).encode('utf-8')).hexdigest()


def _label_matrix_csv(smiles, y, classes, names=None, ids=None):
    smiles = list(smiles)
    result = pd.DataFrame({'SMILES': smiles})
    if names is not None:
        result.insert(1, 'Name', list(names))
    if ids is not None:
        result.insert(0, 'compound_id', list(ids))
    for j, name in enumerate(classes):
        result['label__' + name] = np.asarray(y)[:, j].astype(int)
    return result


def save_heldout_audit_snapshot(
    reports_dir: str | Path, *, class_names, train_smiles, test_smiles,
    y_train, y_test, probabilities, predictions, thresholds,
    train_descriptors, test_descriptors, in_domain, ad_mean_distances,
    ad_cutoff, model_name: str, fingerprint_radius=2, fingerprint_n_bits=2048,
    fingerprint_chirality=True, train_names=None, test_names=None,
    train_ids=None, test_ids=None,
) -> Path:
    """Persist aligned evaluation artifacts BEFORE fitting the production model.

    No inference or scaling is performed here: inputs are the actual held-out
    predictions and the already-fitted train/validation feature space.
    """
    path = Path(reports_dir) / INPUT_FOLDER
    path.mkdir(parents=True, exist_ok=True)
    classes = list(class_names)
    train_smiles, test_smiles = list(train_smiles), list(test_smiles)
    train_y, test_y = np.asarray(y_train), np.asarray(y_test)
    probs, pred = np.asarray(probabilities, dtype=float), np.asarray(predictions)
    train_x, test_x = np.asarray(train_descriptors, dtype=float), np.asarray(test_descriptors, dtype=float)
    thresh = np.broadcast_to(np.asarray(thresholds, dtype=float), (len(classes),)).copy()
    n_train, n_test, n_classes = len(train_smiles), len(test_smiles), len(classes)
    if not n_train or not n_test or not n_classes:
        raise ValueError('Held-out snapshot requires nonempty training, test, and class arrays.')
    if (train_y.shape != (n_train, n_classes) or test_y.shape != (n_test, n_classes)
            or probs.shape != (n_test, n_classes) or pred.shape != (n_test, n_classes)):
        raise ValueError('Held-out snapshot has misaligned targets, predictions, or probabilities.')
    if train_x.ndim != 2 or train_x.shape[0] != n_train or test_x.shape != (n_test,train_x.shape[1]):
        raise ValueError('Held-out feature matrices do not align with the split.')
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError('Held-out features contain nonfinite values.')
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('Held-out probabilities must be finite and between zero and one.')
    if np.any((thresh < 0) | (thresh > 1)):
        raise ValueError('Decision thresholds must be between zero and one.')
    if not np.array_equal(pred.astype(bool), probs >= thresh):
        raise ValueError('Saved predictions and probability thresholds disagree.')
    ad_flags = np.asarray(in_domain, dtype=bool)
    ad_dist = np.asarray(ad_mean_distances, dtype=float)
    if ad_flags.shape != (n_test,) or ad_dist.shape != (n_test,):
        raise ValueError('AD results do not align with the held-out test split.')
    if not np.isfinite(ad_dist).all() or ad_cutoff is None or not np.isfinite(ad_cutoff):
        raise ValueError('AD cutoff and test mean distances must be finite.')
    if not np.array_equal(ad_flags, ad_dist <= ad_cutoff):
        raise ValueError('AD flags do not match the held-out AD cutoff.')

    train_df = _label_matrix_csv(train_smiles, train_y, classes, train_names, train_ids)
    truth_df = _label_matrix_csv(test_smiles, test_y, classes, test_names, test_ids)
    test_pred = pd.DataFrame({'row_number': np.arange(1, n_test+1), 'SMILES': test_smiles})
    if test_ids is not None:
        test_pred['compound_id'] = list(test_ids)
    test_pred['predicted_labels'] = [', '.join(classes[j] for j in np.flatnonzero(row)) for row in pred]
    test_pred['in_domain'] = ad_flags
    test_pred['ad_mean_distance'] = ad_dist
    for j, name in enumerate(classes):
        test_pred['proba__' + name] = probs[:, j]
        test_pred['pred__' + name] = pred[:, j].astype(int)
    train_df.to_csv(path / 'training_only_reference.csv', index=False)
    truth_df.to_csv(path / 'heldout_test_truth_FOR_EVALUATION_ONLY.csv', index=False)
    test_pred.to_csv(path / 'heldout_test_predictions.csv', index=False)
    # The saved matrices are the exact selected, scaled features used by the
    # held-out predictor and training-only AD. No production AD is involved.
    np.savez_compressed(path / 'transformed_features.npz', train=train_x, test=test_x)
    manifest = {
        'source': 'held-out train+validation/test evaluation; never production-refit model',
        'model_name': model_name, 'class_names': classes,
        'train_rows': n_train, 'test_rows': n_test,
        'train_smiles_sha256': _smiles_hash(train_smiles),
        'test_smiles_sha256': _smiles_hash(test_smiles),
        'features': int(train_x.shape[1]),
        'decision_thresholds': thresh.tolist(),
        'ad_cutoff': float(ad_cutoff),
        'fingerprint_radius': int(fingerprint_radius),
        'fingerprint_n_bits': int(fingerprint_n_bits),
        'fingerprint_chirality': bool(fingerprint_chirality),
        'test_labels_not_used_for_audit': True,
        'test_labels_used_only_for_posthoc_evaluation': True,
    }
    (path / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return path


def _read_snapshot(path: Path):
    manifest_path = path / 'snapshot_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Held-out audit snapshot missing: {manifest_path}. Rerun the multilabel TRAINING pipeline '
            'with the updated package to save the original held-out predictions, fitted '
            'training-only features, and training-only reference labels. Do NOT use '
            'production_model.joblib or training_ground_truth_binary.csv for this evaluation.')
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    classes = meta['class_names']
    train = pd.read_csv(path / 'training_only_reference.csv', low_memory=False)
    pred = pd.read_csv(path / 'heldout_test_predictions.csv', low_memory=False)
    with np.load(path / 'transformed_features.npz', allow_pickle=False) as data:
        train_x = data['train'].copy()
        test_x = data['test'].copy()
    if (len(train) != meta['train_rows'] or len(pred) != meta['test_rows']
            or train_x.shape != (len(train), meta['features'])
            or test_x.shape != (len(pred), meta['features'])):
        raise ValueError('Snapshot row/feature counts do not match its manifest.')
    if (_smiles_hash(train.SMILES) != meta['train_smiles_sha256']
            or _smiles_hash(pred.SMILES) != meta['test_smiles_sha256']):
        raise ValueError('Snapshot SMILES order or reference identity changed since training.')
    if list(pred['row_number']) != list(range(1,len(pred)+1)):
        raise ValueError('Held-out test row indices are not sequential/aligned.')
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError('Snapshot descriptor matrices contain nonfinite values.')
    required = ['proba__' + c for c in classes] + ['pred__' + c for c in classes]
    if any(c not in pred for c in required):
        raise ValueError('Snapshot missing saved class probabilities or binary predictions.')
    probabilities = pred[['proba__' + c for c in classes]].to_numpy(dtype=float)
    predicted = pred[['pred__' + c for c in classes]].to_numpy(dtype=int).astype(bool)
    thresholds = np.asarray(meta['decision_thresholds'], dtype=float)
    if not np.array_equal(predicted, probabilities >= thresholds):
        raise ValueError('Predicted labels are inconsistent with saved held-out model scores.')
    flags = pred['in_domain'].astype(str).str.lower().map({'true': True, 'false':False})
    if flags.isna().any():
        raise ValueError('Unrecognized in_domain flag in saved predictions.')
    mean_dist = pred.ad_mean_distance.to_numpy(dtype=float)
    if not np.array_equal(flags.to_numpy(dtype=bool), mean_dist <= meta['ad_cutoff']):
        raise ValueError('Saved AD flags do not match training-only AD distances and cutoff.')
    return meta, classes, train, pred, train_x, test_x, probabilities, predicted, flags.to_numpy(bool)


def evaluate_heldout_recommendations(result, truth: pd.DataFrame, classes: list[str]):
    """Evaluate recommendations *after* all audit decisions have been frozen.

    True=reference annotated positive, NOT laboratory-confirmed odor identity.
    Precision: correct flags / all flags; recall: correct flags / all mismatches.
    """
    if len(result.compounds) != len(truth) or not np.array_equal(
            result.compounds.SMILES.astype(str).to_numpy(), truth.SMILES.astype(str).to_numpy()):
        raise ValueError('Held-out truth and audit compounds differ in row order or identity.')
    target = truth[['label__' + c for c in classes]].to_numpy(dtype=int).astype(bool)
    decisions = result.decisions.copy()
    decisions['reference_positive'] = target[
        decisions.row_number.to_numpy(dtype=int) - 1,
        pd.Index(classes).get_indexer(decisions.label),
    ]
    decisions['extra_predicted_reference'] = decisions.model_predicted & ~decisions.reference_positive
    decisions['missing_reference_label'] = ~decisions.model_predicted & decisions.reference_positive
    decisions['recommendation_agrees_with_reference'] = np.where(
        decisions.action.eq('review_predicted'), decisions.extra_predicted_reference,
        np.where(decisions.action.isin(('review_add','structural_candidate')),
                 decisions.missing_reference_label, False))
    records = []
    for name, action, denominator_col in (
        ('suspected_overprediction','review_predicted','extra_predicted_reference'),
        ('suggested_addition','review_add','missing_reference_label'),
        ('structural_candidate','structural_candidate','missing_reference_label'),
    ):
        subset = decisions[decisions.action.eq(action)]
        hits = int(subset.recommendation_agrees_with_reference.sum())
        n_flagged = len(subset)
        opportunities = int(decisions[denominator_col].sum())
        records.append({
            'audit_task': name,
            'recommendations': n_flagged,
            'agreeing_with_reference': hits,
            'not_in_agreement': n_flagged - hits,
            'precision': hits/n_flagged if n_flagged else np.nan,
            'total_reference_mismatches': opportunities,
            'recall': hits/opportunities if opportunities else np.nan,
        })
    summary = pd.DataFrame(records)
    label_rows = []
    for class_name in classes:
        cls = decisions[decisions.label.eq(class_name)]
        for name, action, denominator_col in (
            ('suspected_overprediction','review_predicted','extra_predicted_reference'),
            ('suggested_addition','review_add','missing_reference_label'),
        ):
            subset = cls[cls.action.eq(action)]
            hits = int(subset.recommendation_agrees_with_reference.sum())
            flagged = len(subset)
            denom = int(cls[denominator_col].sum())
            label_rows.append({'label':class_name,'audit_task':name,
                               'recommendations':flagged,'agreeing_with_reference':hits,
                               'precision':hits/flagged if flagged else np.nan,
                               'recall':hits/denom if denom else np.nan,
                               'reference_mismatches':denom})
    return summary, pd.DataFrame(label_rows), decisions


def run_saved_heldout_audit(reports_dir: str | Path, *, options: AuditOptions = AuditOptions(),
                            knn_representations: str = 'both', out_dir: Optional[str | Path] = None,
                            evaluate: bool = True):
    """Rerun FP-Growth + KNN audits without retraining the held-out predictor.

    No test labels are loaded until the audit has run for *all* representations.
    """
    options.validate()
    if knn_representations not in ('both', 'morgan', 'descriptor'):
        raise ValueError('knn_representations must be both/morgan/descriptor.')
    reports_dir = Path(reports_dir)
    input_dir = reports_dir / INPUT_FOLDER
    output_dir = Path(out_dir) if out_dir is not None else reports_dir / OUTPUT_FOLDER
    meta, classes, train, pred, train_x, test_x, probabilities, predicted, flags = _read_snapshot(input_dir)
    reference = TrainingReference(
        train, classes, radius=meta['fingerprint_radius'], n_bits=meta['fingerprint_n_bits'],
        use_chirality=meta['fingerprint_chirality'])
    if len(reference.Y) != len(train):
        raise ValueError('Training reference contains invalid SMILES; rerun with a validated training dataset.')
    train_x = train_x[reference.valid_indices]
    rules = mine_label_rules(reference.Y, classes, options)
    representations = ['descriptor', 'morgan'] if knn_representations == 'both' else [knn_representations]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid mixing a new partial/blinded rerun with stale outputs from a
    # previous both-mode or reference-scored run. Distinct experiments can
    # always use --out to preserve historical results.
    for previous_mode in ('descriptor', 'morgan'):
        if previous_mode not in representations:
            shutil.rmtree(output_dir / previous_mode, ignore_errors=True)
    for stale_name in ('knn_comparison_summary.csv', 'knn_comparison_label_decisions.csv',
                       'AUDIT_ERROR.txt'):
        (output_dir / stale_name).unlink(missing_ok=True)
    results = {}
    for representation in representations:
        opts = replace(options, knn_representation=representation)
        result = audit_predictions(
            pred.SMILES.astype(str).tolist(), probabilities, classes, reference,
            thresholds=np.asarray(meta['decision_thresholds'], dtype=float),
            predicted_binary=predicted, in_domain=flags,
            ad_mean_distances=pred.ad_mean_distance.to_numpy(dtype=float),
            ad_cutoff=meta['ad_cutoff'], options=opts, rules=rules,
            training_descriptors=train_x if representation == 'descriptor' else None,
            query_descriptors=test_x if representation == 'descriptor' else None,
            descriptor_distance_cutoff=(meta['ad_cutoff']*opts.descriptor_distance_factor
                                        if representation == 'descriptor' else None),
        )
        result.settings.update({
            'evaluation_split': 'heldout_test',
            'model_fitted_on': 'train_plus_validation_only',
            'knn_and_arm_reference': 'train_plus_validation_only',
            'heldout_reference_labels_used_to_generate_recommendations': False,
            'snapshot_training_sha256': meta['train_smiles_sha256'],
            'snapshot_test_sha256': meta['test_smiles_sha256'],
        })
        run_dir = output_dir / representation
        if not evaluate:
            for stale_name in ('reference_agreement_summary.csv',
                               'reference_agreement_by_label.csv',
                               'reference_agreement_all_decisions.csv'):
                (run_dir / stale_name).unlink(missing_ok=True)
        result.save(run_dir)
        results[representation] = result
    if len(results) == 2:
        comparison, decisions = compare_knn_audits(results['descriptor'],results['morgan'])
        comparison.to_csv(output_dir / 'knn_comparison_summary.csv',index=False)
        decisions.to_csv(output_dir / 'knn_comparison_label_decisions.csv',index=False)
    # Label agreement scoring happens only now, after all recommendation files
    # have been generated, and it never feeds back into selection/thresholds.
    if evaluate:
        truth = pd.read_csv(input_dir / 'heldout_test_truth_FOR_EVALUATION_ONLY.csv')
        for mode, result in results.items():
            metrics, per_class, assessed = evaluate_heldout_recommendations(result, truth, classes)
            metrics.insert(0,'knn_representation',mode)
            metrics.to_csv(output_dir / mode / 'reference_agreement_summary.csv',index=False)
            per_class.to_csv(output_dir / mode / 'reference_agreement_by_label.csv',index=False)
            assessed.to_csv(output_dir / mode / 'reference_agreement_all_decisions.csv',index=False)
    (output_dir / 'README.txt').write_text(
        'Leakage-safe held-out label audit. FP-Growth/KNN references use only train+validation.\n'
        'Model probabilities are from the original held-out test model, NOT the production refit.\n'
        'Reference labels are accessed only after the recommendations are saved, to score agreement.\n'
        'No suggested label automatically changes any prediction or test metric.\n'
        'Do not tune audit thresholds using these held-out metrics for a final unbiased claim.\n',
        encoding='utf-8')
    return results
