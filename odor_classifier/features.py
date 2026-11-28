"""Combine RDKit descriptors, Mordred descriptors, fingerprints and mol2vec
embeddings into a single feature matrix, then clean/select/scale it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from .config import Config, DEFAULT_CONFIG
from .descriptors import compute_mordred_descriptors, compute_rdkit_descriptors, smiles_to_mols
from .feature_selection import select_features
from .field_descriptors import compute_field_descriptors
from .fingerprints import compute_maccs_keys, compute_morgan_fingerprints
from .grind_descriptors import compute_grind_descriptors
from .mol2vec_features import compute_mol2vec_embeddings

logger = logging.getLogger(__name__)

# Column-name prefix each descriptor block below writes, keyed by the
# config flag that gates it. Shared with predict.py (to detect which
# blocks a loaded FeaturePipeline actually needs) and pipeline.py (to
# detect a stale raw-features cache after the enabled set changes).
DESCRIPTOR_PREFIXES = {
    "use_rdkit_descriptors": "rdkit_",
    "use_morgan_fingerprints": "morgan_",
    "use_maccs": "maccs_",
    "use_mordred": "mordred_",
    "use_mol2vec": "mol2vec_",
    "use_field_descriptors": "field_",
    "use_grind_descriptors": "grind_",
}


def normalize_pca_n_components(value: float):
    """sklearn's PCA is picky about the *type*, not just the value, of
    n_components: a float must be strictly in (0, 1) (interpreted as
    "retain this much variance"); an exact component count must be an
    actual Python int, not e.g. the float 1.0 -- passing 1.0 raises
    "n_components=1.0 must be ... a float in the range (0.0, 1.0) ...".
    """
    return int(round(value)) if value >= 1 else value


def safe_variance_filter(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, List[str]]:
    """VarianceThreshold, degrading gracefully instead of raising when
    every column's variance falls at or below ``threshold``.

    That can genuinely happen -- a tiny training split (few rows), or
    several rows sharing the same underlying SMILES so their descriptors
    come out bit-for-bit identical, or a user-lowered threshold on a
    dataset with naturally low-variance columns -- and shouldn't abort an
    entire training run over one filtering step. When it does, every
    column is kept as-is instead (downstream correlation-dropping/
    selection still narrows things down). Returns (filtered_df,
    kept_column_names).
    """
    selector = VarianceThreshold(threshold=threshold)
    try:
        arr = selector.fit_transform(df)
        kept = list(df.columns[selector.get_support()])
    except ValueError:
        logger.warning(
            "Variance threshold %.6f left zero columns (every feature was constant in this split); "
            "skipping the variance filter for this fit instead of failing.", threshold,
        )
        return df, list(df.columns)
    return pd.DataFrame(arr, columns=kept, index=df.index), kept


def compute_raw_features(smiles: List[str], config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Compute every configured descriptor/fingerprint/embedding block and
    concatenate them column-wise. No cleaning/scaling/selection here.

    Each block is independently gated by its own ``config.use_*`` flag, so
    the caller can select any one or combination of
    descriptor families. At least one must be enabled.
    """
    if not any(getattr(config, flag) for flag in DESCRIPTOR_PREFIXES):
        raise ValueError(f"At least one descriptor family ({'/'.join(DESCRIPTOR_PREFIXES)}) must be enabled.")

    mols = smiles_to_mols(smiles)
    n_invalid = sum(1 for m in mols if m is None)
    if n_invalid:
        logger.warning("%d/%d SMILES failed to parse; those rows get NaN features.", n_invalid, len(mols))

    blocks = []
    if config.use_rdkit_descriptors:
        blocks.append(compute_rdkit_descriptors(mols))
    if config.use_morgan_fingerprints:
        blocks.append(compute_morgan_fingerprints(mols, config))
    if config.use_maccs:
        blocks.append(compute_maccs_keys(mols))
    if config.use_mordred:
        blocks.append(compute_mordred_descriptors(mols, config))
    if config.use_mol2vec:
        blocks.append(compute_mol2vec_embeddings(mols, config))
    if config.use_field_descriptors:
        blocks.append(compute_field_descriptors(mols, config))
    if config.use_grind_descriptors:
        blocks.append(compute_grind_descriptors(mols, config))

    features = pd.concat(blocks, axis=1)
    features.index = range(len(smiles))
    return features


def log_stage_counts(stage_counts: List[Tuple[str, int]]) -> None:
    """Log the feature funnel (#features after each pipeline stage)."""
    logger.info("Feature funnel:\n%s", "\n".join(f"  {name}: {n}" for name, n in stage_counts))


def save_stage_counts(stage_counts: Optional[List[Tuple[str, int]]], path: Path) -> None:
    """Write feature counts by processing stage to JSON."""
    if not stage_counts:
        return
    Path(path).write_text(
        json.dumps([{"stage": name, "n_features": int(n)} for name, n in stage_counts], indent=2),
        encoding="utf-8",
    )


@dataclass
class FeaturePipeline:
    """Fit-once, reuse-many feature cleaner/selector/scaler.

    Stateful steps (fit on the training split only, to avoid leakage):
      1. drop all-NaN / near-constant columns
      2. median-impute remaining NaNs
      3. variance threshold
      4. drop one of each highly-correlated pair
      5. mutual-information ranking -> keep top K
      6. standard-scale
    """

    config: Config = field(default_factory=lambda: DEFAULT_CONFIG)
    feature_names_: Optional[List[str]] = field(default=None, repr=False)
    impute_values_: Optional[pd.Series] = field(default=None, repr=False)
    variance_selector_: Optional[VarianceThreshold] = field(default=None, repr=False)
    kept_after_variance_: Optional[List[str]] = field(default=None, repr=False)
    kept_after_corr_: Optional[List[str]] = field(default=None, repr=False)
    selected_features_: Optional[List[str]] = field(default=None, repr=False)
    scaler_: Optional[StandardScaler] = field(default=None, repr=False)
    pca_: Optional[PCA] = field(default=None, repr=False)
    # (stage name, #features) after each step of fit_transform -- the
    # feature funnel is logged and saved.
    stage_counts_: Optional[List[Tuple[str, int]]] = field(default=None, repr=False)

    def fit_transform(self, raw: pd.DataFrame, y: pd.Series, Y=None) -> pd.DataFrame:
        """``Y``: multi-label target matrix (rows aligned with ``raw``), so
        mutual-information selection uses every label -- see select_features."""
        df = raw.copy()
        self.stage_counts_ = [("Raw descriptors", raw.shape[1])]

        # Treat +/-inf (a handful of Mordred/RDKit ratio descriptors can
        # produce these on edge-case molecules) as missing, same as NaN.
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop columns that are entirely missing.
        df = df.dropna(axis=1, how="all")

        # Median-impute remaining missing values.
        self.impute_values_ = df.median(numeric_only=True)
        df = df.fillna(self.impute_values_)
        # Any column that was fully NaN has no median -> drop it.
        df = df.dropna(axis=1, how="any")
        self.stage_counts_.append(("After dropping empty columns", df.shape[1]))

        # Variance threshold (degrades gracefully instead of raising if it
        # would leave zero columns -- see safe_variance_filter).
        self.variance_selector_ = VarianceThreshold(threshold=self.config.variance_threshold)
        df, self.kept_after_variance_ = safe_variance_filter(df, self.config.variance_threshold)
        self.stage_counts_.append((f"After variance filter (<{self.config.variance_threshold:g})", df.shape[1]))

        # Drop highly correlated columns (keep the first of each correlated pair).
        df = self._drop_correlated(df)
        self.kept_after_corr_ = list(df.columns)
        self.stage_counts_.append((f"After correlation filter (>{self.config.correlation_threshold:g})", df.shape[1]))

        # Variable selection (mutual_info / boruta / rfecv / ..., "none", or a
        # comma-separated chain -- see config.selection_method).
        selected_cols = select_features(df, y, self.config, stage_counts=self.stage_counts_,
                                        Y=None if Y is None else np.asarray(Y))
        df = df[selected_cols]
        self.selected_features_ = list(df.columns)

        # Scale.
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(df)
        result = pd.DataFrame(scaled, columns=df.columns, index=df.index)

        # Optional PCA reduction (fit on train only).
        if self.config.use_pca:
            n_components = normalize_pca_n_components(self.config.pca_n_components)
            max_available = min(result.shape[0], result.shape[1])
            if isinstance(n_components, int) and n_components > max_available:
                # An exact-count request (>=1) that exceeds what's actually left
                # after selection: clamp instead of raising sklearn's cryptic
                # "n_components=N must be between 0 and min(n_samples,
                # n_features)=M" when too few features survived selection for
                # the requested exact component count.
                logger.warning(
                    "PCA target of exactly %d components exceeds what's available after selection "
                    "(%d train rows x %d selected feature(s)) -- using %d component(s) instead.",
                    n_components, result.shape[0], result.shape[1], max_available,
                )
                n_components = max(1, max_available)
            self.pca_ = PCA(n_components=n_components, random_state=self.config.random_state)
            reduced = self.pca_.fit_transform(result.values)
            cols = [f"pc_{i + 1}" for i in range(reduced.shape[1])]
            result = pd.DataFrame(reduced, columns=cols, index=result.index)
            logger.info(
                "PCA reduced %d selected features to %d components (%.1f%% variance explained).",
                len(selected_cols), reduced.shape[1], self.pca_.explained_variance_ratio_.sum() * 100,
            )
            self.stage_counts_.append(("PCA components", reduced.shape[1]))

        log_stage_counts(self.stage_counts_)
        return result

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("FeaturePipeline must be fit before calling transform().")
        df = raw.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in self.kept_after_variance_:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.kept_after_variance_]
        df = df.fillna(self.impute_values_.reindex(df.columns))
        df = df.fillna(0.0)
        df = df[self.selected_features_]
        scaled = self.scaler_.transform(df)
        result = pd.DataFrame(scaled, columns=df.columns, index=df.index)

        if self.pca_ is not None:
            reduced = self.pca_.transform(result.values)
            cols = [f"pc_{i + 1}" for i in range(reduced.shape[1])]
            result = pd.DataFrame(reduced, columns=cols, index=result.index)

        return result

    def _drop_correlated(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] <= 1:
            return df
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > self.config.correlation_threshold).any()]
        return df.drop(columns=to_drop)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        return joblib.load(path)
