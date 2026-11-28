"""Ground-truth label extraction and validation.

Original odor columns are mapped using the Excel dictionary.
``UC`` is reserved for unclassified model predictions.
"""
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from .config import Config, DEFAULT_CONFIG, UC_LABEL

logger = logging.getLogger(__name__)


def extract_labels(df: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.Series:
    """Return the primary odor class, for feature selection and visualization.

    The first mapped original odor defines the primary class; stored
    class annotations do not determine the training targets.
    It is never the multi-label training target (see multilabel.py).
    """
    from .io_utils import load_class_dictionary
    from .target_labels import build_binary_targets

    _, primary, _ = build_binary_targets(df, load_class_dictionary(config=config), strict_unmapped=True)
    return primary


def validate_against_dictionary(
    df: pd.DataFrame,
    odor_to_class: Dict[str, str],
    config: Config = DEFAULT_CONFIG,
    odor_column: str = "odor1",
) -> float:
    """Sanity-check: fraction of rows where re-deriving the class from
    ``odor_column`` via the dictionary agrees with the stored label column.
    Not used as part of the ML pipeline -- purely a consistency check on
    the provided data.
    """
    if odor_column not in df.columns:
        return float("nan")
    # Compare the actual STORED primary class, even when multi-label mode
    # derives targets directly from original odor descriptors.
    if config.label_column not in df.columns:
        return float("nan")
    stored = df[config.label_column].astype("string").str.strip()
    rederived = df[odor_column].astype(str).str.strip().str.lower().map(odor_to_class)
    comparable = stored.notna() & rederived.notna()
    if comparable.sum() == 0:
        return float("nan")
    agree = (stored[comparable].str.lower() == rederived[comparable].str.lower()).mean()
    return float(agree)


def known_classes_from_labels(labels: pd.Series) -> list[str]:
    """Sorted list of distinct classes observed in the ground truth (excludes UC)."""
    classes = sorted({c for c in labels.dropna().unique() if c != UC_LABEL})
    return classes


def compute_class_cooccurrence(compounds: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Calculate Jaccard co-occurrence from dictionary-derived binary labels."""
    from .multilabel import extract_multilabel_targets, jaccard_cooccurrence

    Y = extract_multilabel_targets(compounds, config)
    return jaccard_cooccurrence(Y)

