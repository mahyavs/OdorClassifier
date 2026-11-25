"""Audit the INTERNAL held-out test split, not a model-refitted external prediction.

First run the updated multilabel training pipeline once to export the
train-only reference, original held-out test probabilities, and train-fitted
feature matrices into outputs/reports/heldout_label_audit_input/.

Usage:
    python scripts/run_heldout_label_audit.py --knn-representation both
    python scripts/run_heldout_label_audit.py --knn-representation morgan --min-tanimoto 0.45
    python scripts/run_heldout_label_audit.py --knn-representation descriptor --descriptor-distance-factor 1.5
    python scripts/run_heldout_label_audit.py --no-evaluate   # blinded recommendations only

No retraining is required after the held-out snapshot exists. The production
model and production training reference are never read by this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from odor_classifier.config import DEFAULT_CONFIG
from odor_classifier.prediction_label_audit import AuditOptions
from odor_classifier.heldout_label_audit import run_saved_heldout_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reports-dir', type=Path, default=DEFAULT_CONFIG.reports_dir,
                        help='Directory containing heldout_label_audit_input/ (default: outputs/reports)')
    parser.add_argument('--out', type=Path, default=None, help='Override output directory')
    parser.add_argument('--knn-representation', choices=['both','descriptor','morgan'], default='both')
    parser.add_argument('--no-evaluate', action='store_true',
                        help='Do not load test reference annotations or create accuracy tables')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--min-tanimoto', type=float, default=.45)
    parser.add_argument('--descriptor-distance-factor', type=float, default=1.5)
    parser.add_argument('--min-close-neighbors', type=int, default=3)
    parser.add_argument('--low-neighbor-support', type=float, default=.20)
    parser.add_argument('--high-neighbor-support', type=float, default=.65)
    parser.add_argument('--min-rule-count', type=int, default=20)
    parser.add_argument('--min-rule-support', type=float, default=.005)
    parser.add_argument('--min-rule-confidence', type=float, default=.30)
    parser.add_argument('--min-rule-lift', type=float, default=1.20)
    parser.add_argument('--max-antecedents', type=int, default=3)
    parser.add_argument('--add-rule-confidence', type=float, default=.45)
    parser.add_argument('--addition-model-fraction', type=float, default=.65)
    args = parser.parse_args(argv)
    options = AuditOptions(
        k_neighbors=args.k, min_tanimoto=args.min_tanimoto,
        descriptor_distance_factor=args.descriptor_distance_factor,
        min_close_neighbors=args.min_close_neighbors,
        low_neighbor_support=args.low_neighbor_support,
        high_neighbor_support=args.high_neighbor_support,
        min_support_count=args.min_rule_count,
        min_support_fraction=args.min_rule_support,
        min_rule_confidence=args.min_rule_confidence,
        min_rule_lift=args.min_rule_lift,
        max_antecedents=args.max_antecedents,
        add_rule_confidence=args.add_rule_confidence,
        addition_model_fraction=args.addition_model_fraction,
    )
    results = run_saved_heldout_audit(args.reports_dir, options=options,
        knn_representations=args.knn_representation, out_dir=args.out,
        evaluate=not args.no_evaluate)
    for mode, report in results.items():
        a = report.decisions.action.value_counts()
        print(f'{mode}: {len(report.compounds)} held-out compounds; '
              f'{a.get("review_predicted",0)} suspected extra labels; '
              f'{a.get("review_add",0)} addition suggestions; '
              f'{a.get("structural_candidate",0)} structural candidates; '
              f'{len(report.rules)} FP-Growth rules')
        if not args.no_evaluate:
            import pandas as pd
            dest = args.out or args.reports_dir / 'heldout_label_audit'
            metrics = pd.read_csv(dest / mode / 'reference_agreement_summary.csv')
            for row in metrics.itertuples():
                print(f'  {row.audit_task}: reference agreement precision={row.precision:.3f}, '
                      f'recall={row.recall:.3f}')
    print('Audit outputs:', (args.out or args.reports_dir / 'heldout_label_audit').resolve())


if __name__ == '__main__':
    main()
