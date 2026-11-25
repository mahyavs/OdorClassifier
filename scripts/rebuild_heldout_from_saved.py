"""Rebuild OdorClassifier's INTERNAL held-out model without descriptor recomputation.

Reads a saved experiment setup (split indices and feature pipeline), the
binary labels exported by the original pipeline, and either the cached raw
features or the feature matrix retained by the last production AD.

It NEVER calls compute_raw_features, selects features anew, runs Optuna, or
fits on the held-out labels. The FULL-DATA production model is used only as
an unfitted hyperparameter template, never to predict the held-out rows.

Examples from the OdorClassifier project root:

    python scripts/rebuild_heldout_from_saved.py --setup-id YOUR_SETUP_ID --check-only
    python scripts/rebuild_heldout_from_saved.py --setup-id YOUR_SETUP_ID --features-source cache
    python scripts/rebuild_heldout_from_saved.py --setup-id YOUR_SETUP_ID --features-source production-ad
    python scripts/rebuild_heldout_from_saved.py --setup-id YOUR_SETUP_ID --features-source cache --allow-feature-drift

Use --features-source production-ad only for the original production AD
corresponding to the saved experiment (it stores original scaled X_full).
See STANDALONE_HELDOUT_REBUILD_README.md for provenance and equivalence limits.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

from odor_classifier.applicability_domain import ApplicabilityDomain
from odor_classifier.config import Config
from odor_classifier.experiment_setups import _raw_feature_hash, _validate_indices
from odor_classifier.heldout_label_audit import save_heldout_audit_snapshot, run_saved_heldout_audit
from odor_classifier.multilabel import multioutput_predict_proba
from sklearn.multioutput import MultiOutputClassifier


def _proba(model, X):
    return multioutput_predict_proba(model, X) if isinstance(model, MultiOutputClassifier) else model.predict_proba(X)


def assign_multilabel_with_uc(proba, class_names, in_domain, config, thresholds):
    """Mirror OdorClassifier prediction logic without importing any descriptor generators."""
    matches = np.asarray(proba) >= np.asarray(thresholds)
    labels = [[class_names[j] for j in np.flatnonzero(row)] for row in matches]
    final = [items if (items and ok) else ["UC"] for items, ok in zip(labels, in_domain)]
    return pd.DataFrame({
        "predicted_labels": [", ".join(items) for items in labels],
        "confidence": np.asarray(proba).max(axis=1),
        "in_domain": np.asarray(in_domain, dtype=bool),
        "final_label": [", ".join(items) for items in final],
        "is_uc": [items == ["UC"] for items in final],
    })

LOG = logging.getLogger('standalone_heldout')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_setup(outputs: Path, setup_id: str | None) -> Path:
    archive = outputs / 'experiment_setups'
    if setup_id is None:
        report = outputs / 'reports' / 'experiment_setup.json'
        if report.is_file():
            setup_id = read_json(report).get('setup_id')
    if setup_id is None:
        setups = [p.parent for p in archive.glob('*/manifest.json')]
        if len(setups) == 1:
            return setups[0]
        raise ValueError('Specify --setup-id. Found %d saved setups in %s.' % (len(setups), archive))
    if not setup_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid setup ID.')
    folder = archive / setup_id
    for name in ('manifest.json', 'indices.json', 'data_split.csv', 'feature_pipeline.joblib'):
        if not (folder / name).is_file():
            raise FileNotFoundError('Missing saved experiment setup file: ' + str(folder / name))
    return folder


def load_reference(outputs: Path, setup_dir: Path):
    manifest = read_json(setup_dir / 'manifest.json')
    if manifest.get('id') != setup_dir.name:
        raise ValueError('Setup ID and manifest disagree.')
    target_file = outputs / 'reports' / 'training_ground_truth_binary.csv'
    targets = pd.read_csv(target_file, low_memory=False)
    split = pd.read_csv(setup_dir / 'data_split.csv', dtype={'SMILES': 'string'})
    n = len(targets)
    if n != int(manifest['n_compounds']) or len(split) != n:
        raise ValueError('Training targets and archived split have different row counts. '
                         'Do not combine a different dataset with the original setup.')
    if not np.array_equal(targets.SMILES.astype(str).to_numpy(), split.SMILES.astype(str).to_numpy()):
        raise ValueError('SMILES/order mismatch between training_ground_truth_binary.csv and '
                         'archived data_split.csv. Refusing to mix experiment runs.')
    if 'id' in split and 'No.' in targets:
        if not np.array_equal(split.id.astype(str).to_numpy(), targets['No.'].astype(str).to_numpy()):
            raise ValueError('Compound identifiers differ between the targets and archived setup.')
    # Make sure the data_split.csv is consistent with indices, not merely same SMILES.
    indices = read_json(setup_dir / 'indices.json')
    train_idx, test_idx, train_pos, val_pos = _validate_indices(indices, n)
    assignments = np.full(n, 'test', dtype=object)
    assignments[train_idx[train_pos]] = 'train'
    assignments[train_idx[val_pos]] = 'validation'
    if not np.array_equal(split['split'].astype(str).to_numpy(), assignments):
        raise ValueError('Archived split assignments disagree with indices.json.')

    classes = [col.removeprefix('label__') for col in targets if col.startswith('label__')]
    if not classes:
        raise ValueError('No label__* columns in training_ground_truth_binary.csv.')
    saved_classes_path = outputs / 'models' / 'label_encoder.joblib'
    if saved_classes_path.is_file():
        saved_classes = joblib.load(saved_classes_path)
        if isinstance(saved_classes, (list, tuple, np.ndarray)):
            saved_classes = [str(c) for c in saved_classes]
            if set(saved_classes) != set(classes):
                raise ValueError('Saved model label names disagree with training targets. '
                                 'Production model is from a different run.')
            classes = saved_classes
    y = targets[['label__' + c for c in classes]].to_numpy(dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Binary reference contains values other than 0 or 1.')
    LOG.info('Verified %d compounds, %d labels, %d train+val and %d test.',
             n, len(classes), len(train_idx), len(test_idx))
    return manifest, targets, classes, y, train_idx, test_idx


def transformed_features_from_cache(outputs: Path, setup_dir: Path, manifest, n_rows: int, allow_drift: bool):
    cache = outputs / 'cache' / 'raw_features.parquet'
    if not cache.is_file():
        raise FileNotFoundError('No feature cache at %s. Consider --features-source production-ad.' % cache)
    raw = pd.read_parquet(cache)
    if len(raw) != n_rows:
        raise ValueError('Feature cache row count does not match archived dataset.')
    original_hash = manifest['raw_feature_sha256']
    current_hash = _raw_feature_hash(raw)
    match = current_hash == original_hash
    if not match and not allow_drift:
        raise ValueError('CURRENT CACHE DIFFERS FROM ARCHIVED RAW FEATURES (SHA-256 mismatch). '
                         'For an exact setup restore the original cache; alternatively use '
                         '--features-source production-ad (if its original full-data AD exists), '
                         'or explicitly acknowledge a nonidentical run with --allow-feature-drift.')
    if not match:
        LOG.warning('FEATURE DRIFT ACCEPTED: descriptor values differ from archived experiment. '
                    'This is NOT an exact replication of original feature inputs.')
    pipeline = joblib.load(setup_dir / 'feature_pipeline.joblib')
    X = pipeline.transform(raw.reset_index(drop=True))
    return X, bool(match), str(cache)


def transformed_features_from_production_ad(outputs: Path, setup_dir: Path, manifest, targets: pd.DataFrame,
                                            trust_legacy: bool):
    """Recover the *stored scaled features*, not use the full-data AD as held-out AD."""
    source = outputs / 'models' / 'applicability_domain.joblib'
    if not source.is_file():
        raise FileNotFoundError('Production AD is missing: ' + str(source))
    ad = ApplicabilityDomain.load(str(source))
    if ad._nn is None or not hasattr(ad._nn, '_fit_X'):
        raise ValueError('Saved AD does not contain its fitted feature matrix.')
    X_values = np.asarray(ad._nn._fit_X, dtype=float)
    if X_values.shape[0] != len(targets):
        raise ValueError('Saved production AD has %d rows, but archived dataset has %d. '
                         'The AD came from a different dataset.' % (X_values.shape[0], len(targets)))
    actual_smiles_hash = hashlib.sha256('\n'.join(targets.SMILES.astype(str)).encode('utf-8')).hexdigest()
    stored_hash = getattr(ad, 'training_smiles_sha256_', None)
    if stored_hash is not None and stored_hash != actual_smiles_hash:
        raise ValueError('Saved AD SMILES checksum does not match original row order.')
    if stored_hash is None:
        # On older versions the AD had no SMILES marker. An all_compounds
        # report from the same completed production run is a useful identity check.
        report = outputs / 'reports' / 'all_compounds_labeled.xlsx'
        checked_report = False
        if report.is_file():
            try:
                sheet = pd.read_excel(report, usecols=['SMILES'])
                checked_report = (len(sheet) == len(targets) and np.array_equal(
                    sheet.SMILES.astype(str).to_numpy(), targets.SMILES.astype(str).to_numpy()))
            except Exception:
                checked_report = False
        if not checked_report and not trust_legacy:
            raise ValueError('This legacy AD has no saved SMILES-order checksum, and the '
                             'all_compounds_labeled.xlsx report could not verify its row order. '
                             'Only if you have independently verified these artifacts are from '
                             'the SAME successful run, pass --trust-legacy-ad-order.')
        LOG.warning('Production AD is an older artifact without signed SMILES provenance; '
                    'verified all_compounds_labeled.xlsx=%s, user trusted=%s.',
                    checked_report, trust_legacy)
    columns = manifest.get('model_input_features', [])
    if X_values.ndim != 2 or X_values.shape[1] != len(columns):
        raise ValueError('Production AD feature dimension does not match the saved setup.')
    if not np.isfinite(X_values).all():
        raise ValueError('Saved production AD contains non-finite features.')
    return pd.DataFrame(X_values.copy(), columns=columns), None, str(source)


def model_template_from_production(source: Path):
    if not source.is_file():
        raise FileNotFoundError('Cannot find previous model at ' + str(source))
    prod = joblib.load(source)
    result = clone(prod)
    return result, prod.__class__.__name__


def load_thresholds(outputs: Path, classes: list[str], global_default: float):
    threshold_path = outputs / 'models' / 'multilabel_thresholds.joblib'
    json_path = outputs / 'reports' / 'multilabel_thresholds.json'
    if threshold_path.is_file():
        item = joblib.load(threshold_path)
        if item is not None:
            arr = np.asarray(item, dtype=float)
            if arr.shape != (len(classes),):
                raise ValueError('Saved class-specific thresholds shape disagrees with class names.')
            return arr, str(threshold_path)
    if json_path.is_file():
        item = read_json(json_path)
        if set(item) != set(classes):
            raise ValueError('Saved threshold JSON class names disagree with setup classes.')
        return np.array([item[c] for c in classes], dtype=float), str(json_path)
    LOG.warning('No saved per-class thresholds: using global threshold %.4f. '
                'If the earlier run used different thresholds, results will differ.', global_default)
    return np.full(len(classes), global_default), 'global CLI fallback'


def basic_evaluation(y_true, y_pred, classes):
    return {
        'n_compounds': int(y_true.shape[0]), 'n_classes': len(classes),
        'subset_accuracy': float(accuracy_score(y_true, y_pred)),
        'hamming_loss': float(hamming_loss(y_true, y_pred)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_precision': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'average_true_labels': float(y_true.sum(axis=1).mean()),
        'average_predicted_labels': float(y_pred.sum(axis=1).mean()),
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outputs-dir', type=Path, default=ROOT / 'outputs')
    p.add_argument('--setup-id', default=None, help='Original archived setup ID. Default: reports/experiment_setup.json')
    p.add_argument('--features-source', choices=['cache', 'production-ad'], default='cache')
    p.add_argument('--allow-feature-drift', action='store_true',
                   help='Allow reusing the CURRENT raw cache even when its numerical values differ from the setup. NOT an identical model.')
    p.add_argument('--trust-legacy-ad-order', action='store_true',
                   help='Override unavailable legacy AD row-order provenance. Verify outputs/model and setup are from one run first.')
    p.add_argument('--model-template', type=Path, default=None,
                   help='Original full-data production_model.joblib (hyperparameter TEMPLATE ONLY; never used to score test).')
    p.add_argument('--fitted-heldout-model', type=Path, default=None,
                   help='Optional separately saved original fitted TRAIN+VAL model. If available, skip fitting and reproduce its predictions.')
    p.add_argument('--global-threshold', type=float, default=0.5,
                   help='Fallback only if no saved class-specific thresholds exist.')
    p.add_argument('--ad-k', type=int, default=5)
    p.add_argument('--ad-percentile', type=float, default=97.5)
    p.add_argument('--out', type=Path, default=None,
                   help='Separate output directory, defaults to outputs/standalone_heldout_rebuild')
    p.add_argument('--audit', choices=['both', 'morgan', 'descriptor', 'none'], default='both')
    p.add_argument('--skip-plots', action='store_true', help='Skip ROC/PR curves and confusion matrices.')
    p.add_argument('--check-only', action='store_true', help='Validate alignment and feature provenance without fitting the model.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    outputs = args.outputs_dir.resolve()
    out = (args.out or outputs / 'standalone_heldout_rebuild').resolve()
    if not (0 <= args.global_threshold <= 1):
        raise ValueError('--global-threshold must be within 0..1.')
    setup_dir = resolve_setup(outputs, args.setup_id)
    manifest, targets, classes, y, idx_train, idx_test = load_reference(outputs, setup_dir)
    if args.features_source == 'cache':
        X, features_exact, source = transformed_features_from_cache(
            outputs, setup_dir, manifest, len(targets), args.allow_feature_drift)
    else:
        X, features_exact, source = transformed_features_from_production_ad(
            outputs, setup_dir, manifest, targets, args.trust_legacy_ad_order)
    if len(X) != len(targets):
        raise ValueError('Transformed feature rows do not match target rows.')
    if X.shape[1] != len(manifest.get('model_input_features', [])):
        raise ValueError('Transformed feature columns disagree with original selected feature space.')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Selected/scaled features contain NaN or infinity.')
    saved_classes = len(classes)
    thresh, thresh_source = load_thresholds(outputs, classes, args.global_threshold)
    if np.any((thresh < 0) | (thresh > 1)):
        raise ValueError('Saved prediction thresholds are not in 0..1.')
    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_train = y[idx_train]
    y_test = y[idx_test]
    smiles_train = targets.SMILES.iloc[idx_train].astype(str).tolist()
    smiles_test = targets.SMILES.iloc[idx_test].astype(str).tolist()
    LOG.info('Feature source: %s; %d features; source hash exact=%s.', source, X.shape[1], features_exact)
    LOG.info('Prediction thresholds: %s. Test targets excluded from model fitting.', thresh_source)
    if args.check_only:
        print('CHECK PASSED: %d compounds; %d train+val, %d held-out test, %d classes; %d features.' %
              (len(targets), len(idx_train), len(idx_test), saved_classes, X.shape[1]))
        print('SOURCE:', source, '| raw cache hash match:', features_exact)
        print('No training was performed.')
        return 0

    if args.fitted_heldout_model:
        model = joblib.load(args.fitted_heldout_model)
        LOG.info('Reusing a pre-fitted held-out model from %s (verify it was trained only on train+val).',
                 args.fitted_heldout_model)
        fit_provenance = 'existing held-out fitted model supplied by user'
        model_name = type(model).__name__
    else:
        model_file = args.model_template or (outputs / 'models' / 'production_model.joblib')
        model, model_name = model_template_from_production(model_file)
        LOG.info('Cloning %s estimator hyperparameters, refitting on %d train+val rows only.',
                 model_name, len(idx_train))
        model.fit(X_train, y_train)
        fit_provenance = 'fitted from previous production estimator configuration, on original train+val indices'
    proba_test = np.asarray(_proba(model, X_test), dtype=float)
    if proba_test.shape != y_test.shape or not np.isfinite(proba_test).all():
        raise ValueError('Held-out predicted probabilities have an unexpected or invalid shape.')
    pred_test = (proba_test >= thresh).astype(int)

    # Training-only AD, not the original full-data production AD!
    cfg = Config()
    cfg.ad_k_neighbors = args.ad_k
    cfg.ad_percentile = args.ad_percentile
    cfg.multilabel_threshold = float(args.global_threshold)
    cfg.reports_dir = out / 'reports'
    cfg.plots_dir = out / 'plots'
    cfg.models_dir = out / 'models'
    for folder in (out, cfg.reports_dir, cfg.plots_dir, cfg.models_dir):
        folder.mkdir(parents=True, exist_ok=True)
    ad = ApplicabilityDomain(config=cfg)
    ad.fit(X_train)
    distances = ad.mean_distance(X_test)
    in_domain = distances <= ad.cutoff_

    # Reports are written to a NEW destination; original models/cache/splits
    # are never overwritten by this standalone script.
    labeled = assign_multilabel_with_uc(proba_test, classes, in_domain, cfg, thresholds=thresh)
    labeled.insert(0, 'SMILES', smiles_test)
    labeled['true_class'] = [', '.join(c for j, c in enumerate(classes) if row[j]) for row in y_test]
    labeled.to_csv(cfg.reports_dir / 'test_predictions_with_UC.csv', index=False)
    if not args.skip_plots:
        from odor_classifier.evaluation import full_evaluation_multilabel
        full_evaluation_multilabel(y_test, pred_test, proba_test, classes, cfg, prefix='test')
    metrics = basic_evaluation(y_test, pred_test, classes)
    (cfg.reports_dir / 'simple_heldout_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    joblib.dump(model, cfg.models_dir / 'heldout_model.joblib')
    ad.save(str(cfg.models_dir / 'heldout_applicability_domain.joblib'))

    save_heldout_audit_snapshot(
        cfg.reports_dir,
        class_names=classes, train_smiles=smiles_train, test_smiles=smiles_test,
        y_train=y_train, y_test=y_test, probabilities=proba_test, predictions=pred_test,
        thresholds=thresh, train_descriptors=X_train.to_numpy(), test_descriptors=X_test.to_numpy(),
        in_domain=in_domain, ad_mean_distances=distances, ad_cutoff=ad.cutoff_,
        model_name=model_name, fingerprint_radius=cfg.morgan_radius,
        fingerprint_n_bits=cfg.morgan_n_bits, fingerprint_chirality=cfg.morgan_use_chirality,
        train_names=targets.Name.iloc[idx_train].tolist() if 'Name' in targets else None,
        test_names=targets.Name.iloc[idx_test].tolist() if 'Name' in targets else None,
        train_ids=targets['No.'].iloc[idx_train].tolist() if 'No.' in targets else None,
        test_ids=targets['No.'].iloc[idx_test].tolist() if 'No.' in targets else None,
    )
    notes = {
        'setup_id': setup_dir.name, 'feature_source': source, 'features_source_type': args.features_source,
        'raw_cache_hash_matches_original': features_exact,
        'feature_reuse_warning': ('Original raw descriptor hash matched.' if features_exact is True else
                                  'Original descriptor identity not proven or drift explicitly allowed.'),
        'model': model_name, 'model_provenance': fit_provenance, 'thresholds_source': thresh_source,
        'train_rows': len(idx_train), 'test_rows': len(idx_test), 'classes': classes,
        'ad_k': args.ad_k, 'ad_percentile': args.ad_percentile,
        'heldout_test_not_used_for_training_or_audit_references': True,
        'identical_predictions_not_guaranteed_without_original_fitted_heldout_model': True,
    }
    (out / 'rebuild_provenance.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    if args.audit != 'none':
        try:
            reports = run_saved_heldout_audit(cfg.reports_dir, knn_representations=args.audit)
            for mode, audit_result in reports.items():
                LOG.info('Audit [%s]: %d compounds; %d suggestions.',
                         mode, len(audit_result.compounds), len(audit_result.suggestions))
        except Exception as exc:
            # Preserve the already-generated model and held-out predictions.
            error = out / 'AUDIT_ERROR.txt'
            error.write_text('%s: %s\n' % (type(exc).__name__, exc), encoding='utf-8')
            LOG.exception('The model was trained and evaluated, but audit failed. See %s.', error)
    print('HELD-OUT EVALUATION:', json.dumps(metrics, indent=2))
    print('Saved outputs at:', out)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('ERROR: %s: %s' % (type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
