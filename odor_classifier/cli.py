"""Command-line entry point.

Usage:
    python -m odor_classifier.cli train [--force-recompute] [--no-optuna] [--trials N]
    python -m odor_classifier.cli predict --smiles "CCO" "c1ccccc1O"
"""
from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .pipeline import run_full_pipeline
from .predict import predict_smiles


def _train(args: argparse.Namespace) -> None:
    config = DEFAULT_CONFIG
    if args.trials is not None:
        config.n_optuna_trials = args.trials
    if args.no_optuna:
        config.use_optuna = False
    results = run_full_pipeline(config, force_recompute_features=args.force_recompute)
    print("\n=== Final model:", results["final_model_name"], "===")
    print("Held-out test statistics:")
    for k, v in results["test_stats"].items():
        if k in ("per_class_report", "roc_auc", "pr_auc"):
            continue
        print(f"  {k}: {v:.4f}")
    print(f"  macro ROC-AUC: {results['test_stats']['roc_auc']['macro']:.4f}")
    print(f"  macro PR-AUC (AP): {results['test_stats']['pr_auc']['macro']:.4f}")
    print(f"  micro PR-AUC (AP): {results['test_stats']['pr_auc']['micro']:.4f}")


def _predict(args: argparse.Namespace) -> None:
    config = DEFAULT_CONFIG
    model = joblib.load(config.models_dir / "production_model.joblib")
    from .features import FeaturePipeline
    feature_pipeline = FeaturePipeline.load(str(config.models_dir / "feature_pipeline.joblib"))
    label_encoder = joblib.load(config.models_dir / "label_encoder.joblib")
    from .applicability_domain import ApplicabilityDomain
    ad = ApplicabilityDomain.load(str(config.models_dir / "applicability_domain.joblib"))

    result = predict_smiles(args.smiles, model, feature_pipeline, label_encoder, ad, config)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(result.to_string(index=False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="odor_classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run the full train/evaluate/label pipeline")
    train_p.add_argument("--force-recompute", action="store_true", help="Ignore cached descriptor features")
    train_p.add_argument("--no-optuna", action="store_true", help="Skip hyperparameter tuning")
    train_p.add_argument("--trials", type=int, default=None, help="Number of Optuna trials")
    train_p.set_defaults(func=_train)

    predict_p = sub.add_parser("predict", help="Predict classes for new SMILES")
    predict_p.add_argument("--smiles", nargs="+", required=True)
    predict_p.set_defaults(func=_predict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
