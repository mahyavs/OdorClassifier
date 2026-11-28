"""Central configuration for the odor_classifier package.

All paths and tunable parameters live here so the rest of the package
never hard-codes a path or a magic number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for _dir in (OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

UC_LABEL = "UC"


@dataclass
class Config:
    """Runtime configuration for a full train/evaluate/predict run."""

    # ---- input data ----
    compounds_path: Path = DATA_DIR / "compounds.xlsx"
    class_dictionary_path: Path = DATA_DIR / "class_dictionary.xlsx"
    mol2vec_model_path: Path = DATA_DIR / "mol2vec_model_300dim.pkl"

    smiles_column: str = "SMILES"
    name_column: str = "Name"
    id_column: str = "No."  # per-compound object number, used to label chemical-space plots
    label_column: str = "class1"  # optional stored primary class, used only for consistency checks

    # ---- feature engineering ----
    # Each descriptor/fingerprint/embedding family below is independently
    # toggleable (compute_raw_features skips whatever's off) -- selecting
    # "one or two or more" of them is exactly what these flags are for. At
    # least one must stay on; compute_raw_features raises if all are off.
    use_rdkit_descriptors: bool = True  # ~210 RDKit 2D physicochemical descriptors
    use_morgan_fingerprints: bool = True  # circular substructure fingerprint (a.k.a. ECFP)
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    # Include stereochemistry (R/S chiral centres, as written in the SMILES)
    # in the Morgan atom environments, so specified enantiomers/diastereomers
    # (e.g. dextro- vs laevo-carvone) get different fingerprints. Centres the
    # SMILES leaves unspecified can't be told apart either way.
    morgan_use_chirality: bool = True
    use_maccs: bool = True  # 167 predefined structural-alert keys
    use_mordred: bool = True  # ~1600 2D physicochemical/topological descriptors
    mordred_n_proc: Optional[int] = None  # None => mordred uses multiprocessing.cpu_count()
    use_mol2vec: bool = True  # pretrained substructure embedding
    mol2vec_radius: int = 1
    # 3D geometric descriptors (WHIM/GETAWAY/RDF/MoRSE/AUTOCORR3D):
    # spatial-autocorrelation and radial-distribution descriptors weighted by
    # atomic mass/charge/polarizability/van-der-Waals volume, computed
    # directly from atom positions over a single generated+optimized 3D
    # conformer per molecule -- alignment-independent like GRIND (below),
    # but geometric/statistical rather than a real interaction-energy field.
    # Off by default: every other descriptor family here is 2D-only and
    # cheap; embedding+optimizing a 3D conformer per compound is
    # meaningfully slower and occasionally fails outright (those compounds
    # fall back to a NaN row, same as any other descriptor failure
    # elsewhere in this module).
    use_field_descriptors: bool = False
    # GRIND (GRid-INdependent Descriptors, see grind_descriptors.py): real
    # GRID-style molecular interaction fields (hydrogen-bond acceptor/donor,
    # hydrophobic, and electrostatic probe-molecule interaction energies
    # computed on an actual 3D grid via RDKit's rdMIF module), reduced to
    # an alignment-independent 1D profile of distances between each
    # molecule's own favorable interaction "hotspots" (Pastor & Cruciani's
    # MACC-2 transform). Off by default for the same reason as
    # use_field_descriptors -- 3D conformer-dependent and slower than the
    # 2D-only families above.
    use_grind_descriptors: bool = False
    grind_grid_margin: float = 3.0  # distance [A] the grid extends past the molecule's surface
    grind_grid_spacing: float = 0.75  # [A] between adjacent grid points
    grind_max_nodes_per_probe: int = 6  # favorable-hotspot cap per probe
    grind_min_node_distance: float = 2.0  # [A] two hotspots must be at least this far apart
    grind_bin_width: float = 1.0  # [A] per distance bin in the MACC-2 profile
    grind_n_bins: int = 15  # number of distance bins (0 to grind_bin_width * grind_n_bins Angstroms)

    variance_threshold: float = 1e-4
    correlation_threshold: float = 0.95
    max_features_after_selection: int = 1000

    # variable ("feature") selection strategy applied after the variance/
    # correlation filters and before scaling. One of:
    #   "mutual_info" -- fast, model-free ranking
    #   "boruta"      -- all-relevant wrapper selection around a RandomForest
    #   "rfecv"       -- cross-validated recursive feature elimination
    #   "lasso"       -- SelectFromModel around L1-regularized logistic regression
    #   "shap"        -- top-K by mean |SHAP value| from a quick LightGBM model
    #   "none"        -- no variable selection (keep every filtered feature)
    # or a comma-separated chain, e.g. "boruta,shap" -- run in that order,
    # each method on the previous one's selected columns.
    selection_method: str = "mutual_info"
    boruta_max_iter: int = 50
    rfecv_step: float = 0.1
    rfecv_min_features: int = 30
    # boruta/rfecv/lasso/shap first keep only this many columns, ranked by
    # mutual information, since they scale badly with thousands of columns
    # (see feature_selection._pre_reduce). 0 = no pre-filter (every column
    # goes to the selector -- Boruta/RFECV can then be very slow).
    selection_pre_reduce_to: int = 800

    # Resume an interrupted classification run from its checkpoints
    # (outputs/checkpoints/, see checkpoint.py) when the settings and data
    # are unchanged. False = discard any checkpoints and start fresh (use
    # this after editing the code). Checkpoints are always written either
    # way, and deleted when a run completes.
    resume_from_checkpoint: bool = True

    # Optional reproducible classifier-comparison experiment. A saved setup
    # stores the exact train/validation/test assignments and a fitted feature
    # pipeline (including RFECV, imputations, scaling and optional PCA).
    # None = create a fresh setup; otherwise load this saved setup ID.
    reuse_experiment_setup: Optional[str] = None
    save_experiment_setup: bool = True

    # optional final PCA reduction, applied after variable selection+scaling.
    use_pca: bool = False
    pca_n_components: float = 0.95  # int -> exact #components, float in (0,1) -> retained variance

    # ---- modeling ----
    # validation_size is the fraction of the *training* split (itself
    # test_size-held-out from the full dataset) held out as the validation
    # split. Round-1 model comparison and Optuna tuning score each model /
    # trial by stratified k-fold cross-validation inside the remaining
    # train part (cv_folds folds, mean macro-F1); with cv_folds < 2 they
    # instead fit on train and score on the validation split. The
    # validation split itself is always kept for the decisions that need a
    # model that never saw it: multi-label threshold calibration. The final
    # chosen model is refit on train+validation before being scored once on
    # the untouched test split. In Single model mode, independently of Optuna,
    # cv_folds >= 2 REPORTS CV on train+validation (80%) using the chosen
    # fixed/tuned hyperparameters. cv_folds 0/1 instead REPORTS held-out
    # validation metrics. This reporting does not change Optuna's inner
    # tuning folds, calibration, or final training/test fitting behavior.
    test_size: float = 0.2
    validation_size: float = 0.15
    cv_folds: int = 3
    # Both splits use iterative multi-label stratification; see splitting.py.
    random_state: int = 42
    n_optuna_trials: int = 25
    use_optuna: bool = True
    # In automatic model-selection mode (selected_model=None), candidates
    # are normally compared with their default hyperparameters first, and
    # only the winner is then Optuna-tuned -- a model that would have won
    # after tuning but loses on defaults never gets the chance. Setting
    # this True instead runs the full Optuna search (n_optuna_trials each)
    # on every enabled candidate before picking the best one by *tuned*
    # validation macro-F1. Every classical model family has
    # a tuning search space defined, so this covers all of them. Multiplies
    # total tuning time by roughly the number of enabled candidates --
    # opt-in, off by default. No effect in single-model mode, where the
    # one selected candidate is already tuned either way.
    tune_all_candidates: bool = False
    use_svm: bool = True
    use_naive_bayes: bool = True
    use_extra_trees: bool = True
    use_catboost: bool = True
    use_random_forest: bool = True
    use_hist_gradient_boosting: bool = True
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_mlp: bool = True
    # None compares enabled tabular candidates; a model key runs just that classifier.
    selected_model: Optional[str] = None

    # ---- chemical space plots ----
    # PCA computes this many components (whichever is smaller vs. #features/#samples)
    # and saves the coordinates and explained-variance ratios.
    chem_space_pca_components: int = 10
    # t-SNE/UMAP dimensions aren't individually orderable like PCA's, so a fixed
    # small number (enough for the 2D default and the 3D view) is computed once.
    chem_space_tsne_umap_components: int = 3
    chem_space_max_annotated_points: int = 400  # above this, static plots skip per-point number labels
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: float = 30.0
    # t-SNE and UMAP both fit on pairwise distances, which get noisy and slow
    # in the full ~400-dimensional selected feature space -- standard practice
    # (and sklearn's own TSNE default of init="pca") is to pre-reduce with a
    # quick linear PCA first. Only takes effect when the feature matrix has
    # more columns than this; harmless/skipped once already-reduced (e.g. when
    # config.use_pca is also on). Set to 0 to disable pre-reduction entirely.
    tsne_umap_pca_components: int = 50

    # ---- applicability domain / UC assignment ----
    # The post-prediction held-out Label Audit uses ONLY train+validation
    # references and the original held-out test probabilities. Morgan and
    # descriptor representations are evaluated separately, with default
    # thresholds kept unchanged for later controlled sensitivity studies.
    run_heldout_label_audit: bool = True
    heldout_audit_knn_modes: str = 'both'  # 'descriptor', 'morgan', or 'both'

    ad_k_neighbors: int = 5
    ad_percentile: float = 97.5  # training-distance percentile used as cutoff

    # ---- multi-label targets ----
    # All dictionary categories are represented by binary targets.
    multilabel: bool = field(default=True, init=False)
    multilabel_threshold: float = 0.5  # per-label probability cutoff for "this label applies"
    # Calibrate each label's own decision threshold on the validation split
    # (maximizing that label's F1) instead of using multilabel_threshold for
    # every label uniformly -- rare labels are systematically under-confident
    # under one global cutoff, so this is a real, free per-class F1 gain.
    calibrate_multilabel_thresholds: bool = False
    # Instead of every label's binary sub-classifier seeing the exact same
    # shared selected feature subset, each one additionally narrows down to
    # its own top-K most informative features (mutual information against
    # that label alone) before fitting -- a 2025 paper on this exact task
    # found this gave competitive performance at 98-99% further
    # dimensionality reduction per label. Applied only inside each label's
    # own classifier pipeline (wrap_models_for_multilabel); the shared
    # feature-selection/applicability-domain/chemical-space space used
    # everywhere else in this module is completely unaffected. Off by
    # default -- adds a per-label selection step to every fit.
    per_label_feature_selection: bool = False
    per_label_max_features: int = 30

    # ---- dimensionality-reduction methods (Dimensionality Reduction tab) --
    # on-demand only (computed from cached descriptors + the fitted
    # production feature pipeline, or from freshly-computed descriptors
    # when pointed at a different uploaded file), not part of the standard
    # training-time pipeline (which separately always computes PCA/t-SNE/
    # UMAP for the Chemical space tab regardless of these flags). Every
    # method is independently toggleable. PCA/t-SNE/UMAP default on (cheap,
    # already-standard methods); the seven more advanced ones default off
    # since several (Isomap/LLE/MDS in particular) scale badly with
    # compound count and nobody should pay that cost without asking for it.
    use_pca_method: bool = True
    use_tsne_method: bool = True
    use_umap_method: bool = True
    use_kernel_pca: bool = False
    use_isomap: bool = False
    use_lle: bool = False  # Locally Linear Embedding
    use_mds: bool = False  # Multi-Dimensional Scaling
    use_factor_analysis: bool = False
    use_ica: bool = False  # Independent Component Analysis (FastICA)
    use_spectral_embedding: bool = False

    # ---- misc ----
    outputs_dir: Path = OUTPUTS_DIR
    plots_dir: Path = PLOTS_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR


DEFAULT_CONFIG = Config()
