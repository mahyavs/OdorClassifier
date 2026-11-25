"""Command-line post-prediction multilabel audit for unseen SMILES.

Examples:
    python scripts/run_prediction_label_audit.py --input external.csv
    python scripts/run_prediction_label_audit.py --input external.xlsx --out outputs/reports/external_audit

The input only needs a SMILES column. Any supplied target columns are
*ignored*: suggestions never depend on external ground-truth labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from odor_classifier.config import DEFAULT_CONFIG
from odor_classifier.prediction_label_audit import (TrainingReference, AuditOptions, audit_predictions,
    training_descriptors_from_ad)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='CSV/XLSX containing a SMILES column')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'reports' / 'external_prediction_audit')
    parser.add_argument('--reference', type=Path, default=None,
                        help='Saved production-training binary CSV; defaults to outputs/reports/training_ground_truth_binary.csv')
    parser.add_argument('--knn-representation', choices=['descriptor', 'morgan', 'both'], default='descriptor',
                        help='descriptor: training-scaled features/Euclidean (default); '
                             'morgan: Morgan/Tanimoto; both: run matched comparison')
    parser.add_argument('--descriptor-distance-factor', type=float, default=1.5,
                        help='Accept descriptor neighbors within this multiplier of saved AD cutoff; '
                             'default 1.5; tune separately from Morgan threshold')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--min-tanimoto', type=float, default=0.45)
    parser.add_argument('--min-close-neighbors', type=int, default=3)
    parser.add_argument('--max-antecedents', type=int, default=3)
    parser.add_argument('--min-rule-count', type=int, default=20)
    # Association-rule support and confidence settings.
    parser.add_argument('--min-rule-support', type=float, default=0.005,
                        help='Minimum fraction of training compounds per itemset (default 0.005)')
    parser.add_argument('--min-rule-confidence', type=float, default=0.30)
    parser.add_argument('--min-rule-lift', type=float, default=1.20)
    parser.add_argument('--add-rule-confidence', type=float, default=0.45,
                        help='Minimum ARM confidence to promote a structurally supported label')
    parser.add_argument('--addition-model-fraction', type=float, default=0.65,
                        help='Near-threshold model score as fraction of the class threshold')
    parser.add_argument('--low-neighbor-support', type=float, default=0.20)
    parser.add_argument('--high-neighbor-support', type=float, default=0.65)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f'Input not found: {args.input}')
    table = pd.read_csv(args.input) if args.input.suffix.lower() == '.csv' else pd.read_excel(args.input)
    smiles_col = next((c for c in table.columns if c.strip().lower() == 'smiles'), None)
    if smiles_col is None:
        parser.error('Input file requires a SMILES column.')
    if table[smiles_col].isna().any():
        parser.error('SMILES column contains blank values; remove or correct these rows first.')
    smiles = table[smiles_col].astype(str).str.strip().tolist()

    from odor_classifier.features import FeaturePipeline
    from odor_classifier.applicability_domain import ApplicabilityDomain
    from odor_classifier.predict import predict_smiles

    cfg = DEFAULT_CONFIG
    model_dir = cfg.models_dir
    model = joblib.load(model_dir / 'production_model.joblib')
    fp = FeaturePipeline.load(str(model_dir / 'feature_pipeline.joblib'))
    ad = ApplicabilityDomain.load(str(model_dir / 'applicability_domain.joblib'))
    class_names = joblib.load(model_dir / 'label_encoder.joblib')
    if not isinstance(class_names, (list, tuple)):
        parser.error('The deployed model is not a multilabel model.')
    thresh_file = model_dir / 'multilabel_thresholds.joblib'
    thresholds = joblib.load(thresh_file) if thresh_file.exists() else None

    preds, query_X = predict_smiles(smiles, model, fp, class_names, ad, cfg,
                           thresholds=thresholds, include_audit_data=True,
                           return_transformed_features=True)
    pcols = ['proba__' + c for c in class_names]
    train_cfg = getattr(fp, 'config', None)
    reference = TrainingReference.from_csv(
        args.reference or cfg.reports_dir / 'training_ground_truth_binary.csv',
        list(class_names), radius=int(getattr(train_cfg,'morgan_radius',cfg.morgan_radius)),
        n_bits=int(getattr(train_cfg,'morgan_n_bits',cfg.morgan_n_bits)),
        use_chirality=bool(getattr(train_cfg,'morgan_use_chirality',cfg.morgan_use_chirality)))
    options = AuditOptions(knn_representation=('descriptor' if args.knn_representation == 'both'
                                                  else args.knn_representation),
                           descriptor_distance_factor=args.descriptor_distance_factor,
                           k_neighbors=args.k, min_tanimoto=args.min_tanimoto,
                           min_close_neighbors=args.min_close_neighbors,
                           max_antecedents=args.max_antecedents, min_support_count=args.min_rule_count,
                           min_support_fraction=args.min_rule_support,
                           min_rule_confidence=args.min_rule_confidence,
                           min_rule_lift=args.min_rule_lift,
                           add_rule_confidence=args.add_rule_confidence,
                           addition_model_fraction=args.addition_model_fraction,
                           low_neighbor_support=args.low_neighbor_support,
                           high_neighbor_support=args.high_neighbor_support)
    options.validate()
    representations = (['descriptor', 'morgan'] if args.knn_representation == 'both'
                       else [args.knn_representation])
    reference_X = (training_descriptors_from_ad(ad, reference, expected_columns=query_X.columns)
                   if 'descriptor' in representations else None)
    from odor_classifier.prediction_label_audit import mine_label_rules
    rules = mine_label_rules(reference.Y, list(class_names), options)
    args.out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out / 'predictions_with_all_class_probabilities.csv', index=False)
    reports = {}
    for representation in representations:
        choice = replace(options, knn_representation=representation)
        report = audit_predictions(smiles, preds[pcols].to_numpy(), list(class_names), reference,
                                   thresholds=thresholds if thresholds is not None else cfg.multilabel_threshold,
                                   in_domain=preds['in_domain'].to_numpy(dtype=bool),
                                   ad_mean_distances=preds['ad_mean_distance'].to_numpy(dtype=float),
                                   ad_cutoff=ad.cutoff_, options=choice, rules=rules,
                                   training_descriptors=reference_X if representation == 'descriptor' else None,
                                   query_descriptors=query_X if representation == 'descriptor' else None,
                                   descriptor_distance_cutoff=(ad.cutoff_ * choice.descriptor_distance_factor
                                       if representation == 'descriptor' else None))
        # For a one-mode run, keep the original flat output paths. For the
        # matched comparison, isolate the two outputs so they cannot overwrite.
        output_dir = args.out / representation if len(representations) > 1 else args.out
        report.save(output_dir)
        reports[representation] = report
        cutoff_note = (f'descriptor cutoff={ad.cutoff_ * choice.descriptor_distance_factor:.5g}'
                       if representation == 'descriptor' else f'min Tanimoto={choice.min_tanimoto:.3f}')
        print(f'{representation}: {cutoff_note}, '
              f'{(report.compounds.audit_status == "review").sum()} compounds to review, '
              f'{len(report.suggestions)} label suggestions')
        print(f'Results written to: {output_dir.resolve()}')

    if len(reports) == 2:
        from odor_classifier.prediction_label_audit import compare_knn_audits
        summary, decisions = compare_knn_audits(reports['descriptor'], reports['morgan'])
        summary.to_csv(args.out / 'knn_comparison_summary.csv', index=False)
        decisions.to_csv(args.out / 'knn_comparison_label_decisions.csv', index=False)
        print('Matched comparison files: knn_comparison_summary.csv and '
              'knn_comparison_label_decisions.csv')
    print(f'External compounds: {len(smiles)}; reference compounds: {len(reference.Y)}; '
          f'FP-Growth ARM rules: {len(rules)}')


if __name__ == '__main__':
    main()
