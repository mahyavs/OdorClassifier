# OdorClassifier: Multi-label molecular odor prediction

OdorClassifier is a Python framework for investigating quantitative structure–odor relationships (QSORs) and predicting multiple odor categories from molecular structures represented by SMILES. It combines molecular descriptors, fingerprints and molecular embeddings with machine-learning classifiers. A post-prediction label audit uses structural similarity and odor-label association rules to flag potentially excessive or missing predicted labels.

## Contents

- [Dataset and odor dictionary](#dataset-and-odor-dictionary)
- [Installation](#installation)
- [Computational workflow](#computational-workflow)
- [Training and evaluation](#training-and-evaluation)
- [Prediction and label auditing](#prediction-and-label-auditing)
- [Output files](#output-files)
- [Repository structure](#repository-structure)
- [Reproducibility and testing](#reproducibility-and-testing)
- [Citation and data availability](#citation-and-data-availability)

## Dataset and odor dictionary

| File | Description |
| --- | --- |
| `data/compounds.xlsx` | Final curated dataset of 5,288 compounds. |
| `data/class_dictionary.xlsx` | Dictionary defining 22 odor categories and their associated original odor terms. |


During training, the original odor descriptors are mapped directly to the dictionary and converted to a binary compound-by-category matrix. Repeated occurrences of a category within one compound contribute a single positive target. All 22 categories are retained regardless of frequency. `class1`–`class10` are provided for transparency, not as the source of the model's binary targets. 

## Installation

Run all commands from the repository root. Create and activate a dedicated Python environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Mol2vec model

When Mol2vec is enabled, the code requires the `mol2vec` package (for `mol2alt_sentence`) and a pretrained Word2Vec model readable by Gensim. Install the compatible Mol2vec distribution used in the original experiments and place the corresponding pretrained model at `data/mol2vec_model_300dim.pkl`. The model file is not included here. If you only want to run the workflow without Mol2vec, use `--no-mol2vec`. This changes the feature set and is not equivalent to reproducing an experiment using Mol2vec.

## Computational workflow

1. **Target construction:** Map original odor descriptors to the 22 dictionary-defined binary targets.
2. **Molecular feature generation:** Compute the enabled descriptor, fingerprint and embedding blocks.
3. **Data partitioning:** Split the dataset into training, validation and held-out test subsets, using iterative multi-label stratification when the dependency is available. The existing code falls back to primary-category stratification if it is unavailable.
4. **Feature processing and selection:** Remove unusable features, impute missing values, filter by variance and correlation, apply the configured selection method, scale features and optionally apply PCA.
5. **Model training and optimization:** Fit binary classifiers for each odor category using the selected classifier family. Compare enabled candidates or train one selected family, with optional Optuna hyperparameter optimization.
6. **Evaluation:** Evaluate predictions using multi-label classification metrics, optional per-label threshold calibration, and applicability-domain assessment. The held-out test is evaluated using a model refitted on train+validation.
7. **Production refitting:** Refit the selected classifier and applicability domain on the full dataset, reusing the feature pipeline fitted on the training subset. Save models, reports and chemical-space visualizations.
8. **Post-prediction label audit:** Combine nearest-neighbor label support and FP-Growth association rules to flag possible excessive or missing labels. The internal test audit uses the evaluation snapshot frozen before production refitting and train+validation references only; the test truth is kept separate for evaluating recommendations.

### Molecular representations

| Feature family | Configuration flag in `odor_classifier/config.py` |
| --- | --- |
| RDKit physicochemical descriptors | `use_rdkit_descriptors` |
| Morgan fingerprints | `use_morgan_fingerprints` |
| MACCS structural keys | `use_maccs` |
| Mordred descriptors | `use_mordred` |
| Mol2vec embeddings | `use_mol2vec` |
| 3D geometric descriptors (WHIM, GETAWAY, RDF, MoRSE, AUTOCORR3D) | `use_field_descriptors` |
| GRIND descriptors | `use_grind_descriptors` |

Available feature-selection methods are mutual information (`mutual_info`), Boruta (`boruta`), recursive feature elimination with cross-validation (`rfecv`), L1-regularized logistic regression (`lasso`), SHAP-based ranking (`shap`) and no additional selection (`none`). Methods can be applied sequentially.

### Classifier families

| Classifier | Command-line model key |
| --- | --- |
| Support vector machine | `svm` |
| Gaussian Naive Bayes | `naive_bayes` |
| Extra Trees | `extra_trees` |
| CatBoost | `catboost` |
| Random Forest | `random_forest` |
| Histogram Gradient Boosting | `hist_gradient_boosting` |
| XGBoost | `xgboost` |
| LightGBM | `lightgbm` |
| Multilayer perceptron | `mlp` |

## Training and evaluation

Display the available command-line options:

```bash
python scripts/run_pipeline.py --help
```

Run the pipeline with the script's defaults:

```bash
python scripts/run_pipeline.py
```

Omitting `--selected-model` compares the enabled classifier families. By default, candidate comparison precedes tuning of the selected model. Use `--tune-all-candidates` to tune every enabled candidate before selection.

Key options:

| Argument | Purpose |
| --- | --- |
| `--selected-model` | Train one classifier family. |
| `--selection-method` | Select one or more feature-selection methods in execution order. |
| `--trials` | Number of Optuna trials. |
| `--cv-folds` | Number of cross-validation folds. |
| `--test-size`, `--validation-size` | Test fraction and validation fraction of the non-test subset. |
| `--random-state` | Random seed. |
| `--no-calibrate-thresholds` | Disable per-label threshold calibration. |
| `--use-field-descriptors`, `--use-grind-descriptors` | Enable optional 3D feature families. |
| `--use-pca` | Enable PCA in feature processing. |
| `--reuse-experiment-setup` | Reuse a saved split and fitted feature pipeline. |
| `--fresh-start` | Discard existing training checkpoints. |
| `--force-recompute` | Recompute cached raw molecular features. |
| `--no-heldout-label-audit` | Skip automatic auditing while retaining its input snapshot. |

The remaining parameters and feature-family flags are documented in `odor_classifier/config.py` and the command-line help. The CLI defaults are not necessarily the same as direct Python `Config` defaults (notably threshold calibration). Record the actual entry point and arguments used for each experiment.

The supplied workbook is already curated. `scripts/curate_compounds.py` is retained for reference; it removes invalid SMILES, pure elements, metal-containing compounds and molecules above 250 g/mol, and calculates molecular weights. It overwrites the input workbook after creating a backup if one does not exist. It is not necessary to rerun it on the supplied final dataset.

## Prediction and label auditing

### Prediction of new molecules

After a completed training run, prepare a CSV or Excel file containing a `SMILES` column. For example, `new_compounds.csv` can contain:

```csv
SMILES
CCO
c1ccccc1O
```

Run prediction and the post-prediction audit using both structural representations:

```bash
python scripts/run_prediction_label_audit.py --input new_compounds.csv --knn-representation both
```

This command uses the saved production model, fitted feature pipeline, applicability domain and calibrated per-label thresholds if present. It writes model probabilities and audit recommendations under `outputs/reports/external_prediction_audit/` by default. External ground-truth odor columns, if supplied, are not used to generate suggestions.

For basic prediction only, a separate entry point is available:

```bash
python -m odor_classifier.cli predict --smiles "CCO" "c1ccccc1O"
```

### Held-out test label audit

The training workflow normally runs the held-out audit automatically. To rerun it from a completed experiment's saved inputs:

```bash
python scripts/run_heldout_label_audit.py --knn-representation both
```

The audit supports two nearest-neighbor representations:

- `descriptor`: distances in the fitted molecular feature space (Euclidean distance).
- `morgan`: Morgan fingerprint similarity (Tanimoto coefficient).
- `both`: run both representations and compare their suggestions.

FP-Growth association rules are combined with structural neighbor support and model predictions. Suggestions are candidates for review, not experimentally confirmed odor annotations. The held-out evaluation compares recommendations with separately stored reference labels only after generation of the recommendations. Do not substitute the production model for the internal held-out model.

For available audit parameters:

```bash
python scripts/run_heldout_label_audit.py --help
python scripts/run_prediction_label_audit.py --help
```

`scripts/rebuild_heldout_from_saved.py` can reconstruct a held-out snapshot from compatible saved experimental artifacts; consult its help and provenance checks before using it.

## Output files

Training creates an `outputs/` directory containing reports, models, plots, cached features and experiment artifacts. Important files include:

| Path (relative to `outputs/`) | Description |
| --- | --- |
| `reports/dictionary_target_audit.csv` | Dictionary category coverage and positive target counts. |
| `reports/training_ground_truth_binary.csv` | Binary targets for the supplied compound set. |
| `reports/data_split.csv` | Training, validation and held-out test assignments. |
| `reports/feature_funnel.json` | Feature counts at successive processing stages. |
| `reports/cv_model_comparison.csv` | Candidate comparison when automatic model selection is used. |
| `reports/test_predictions_with_UC.csv` | Held-out test predictions including unclassified assignments. |
| `reports/heldout_label_audit_input/` | Frozen evaluation predictions, references, thresholds and separately stored truth. |
| `reports/heldout_label_audit/` | Held-out audit results, when enabled. |
| `reports/all_compounds_labeled.xlsx` | Full-dataset production labeling report. |
| `models/production_model.joblib` | Fitted production classifier. |
| `models/feature_pipeline.joblib` | Fitted production feature pipeline. |
| `models/label_encoder.joblib` | Ordered model target categories. |
| `models/applicability_domain.joblib` | Fitted applicability-domain model. |
| `models/multilabel_thresholds.joblib` | Per-label thresholds, if calibrated. |
| `plots/` | Evaluation and chemical-space visualizations. |

## Repository structure

```text
.
├── data/
│   ├── compounds.xlsx
│   └── class_dictionary.xlsx
├── odor_classifier/
│   ├── config.py                  # Parameters and paths
│   ├── target_labels.py           # Dictionary-to-binary targets
│   ├── descriptors.py             # Molecular descriptors
│   ├── features.py                # Feature construction and processing
│   ├── feature_selection.py       # Feature-selection methods
│   ├── modeling.py                # Candidate classifiers
│   ├── multilabel.py              # Binary classifier wrappers
│   ├── optuna_search.py           # Hyperparameter search
│   ├── pipeline.py                # Training and evaluation workflow
│   ├── predict.py                 # Inference and UC assignment
│   ├── applicability_domain.py    # Applicability-domain assessment
│   ├── prediction_label_audit.py  # Structural and association-rule audit
│   ├── heldout_label_audit.py     # Held-out evaluation of audit suggestions
│   ├── arm_fpgrowth.py             # FP-Growth association rule mining
│   └── ...
├── scripts/
│   ├── curate_compounds.py
│   ├── run_pipeline.py
│   ├── run_heldout_label_audit.py
│   ├── run_prediction_label_audit.py
│   └── rebuild_heldout_from_saved.py
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```
