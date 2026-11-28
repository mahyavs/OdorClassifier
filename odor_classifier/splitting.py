"""Train/test and train/validation splitting for the classification
pipeline -- one function shared by the pipeline (pipeline.run_full_pipeline)
and split-analysis utilities.

The training pipeline uses multi-label stratification. The primary-class
stratification helper is retained for the existing dependency fallback:

- ``"class1"``: stratify on each compound's primary class (scikit-learn
  ``train_test_split(stratify=...)``) -- every primary class keeps its
  proportion, secondary labels are spread at random. Identical to the
  pipeline's original behavior, so earlier runs stay reproducible.
- ``"multilabel"``: iterative stratification over every label (Sechidis et
  al., 2011; ``iterative-stratification`` package) -- multi-label mode only;
  falls back to class1 with a warning if the package isn't installed.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SPLIT_METHODS = ("class1", "multilabel")


def stratified_split(
    n: int,
    test_size: float,
    random_state: int,
    class1,
    Y=None,
    method: str = "class1",
) -> tuple[np.ndarray, np.ndarray]:
    """(idx_kept, idx_held_out) positions for one split of ``n`` rows.

    ``class1``: per-row primary class (stratification key for "class1").
    ``Y``: per-row binary label matrix (rows x labels) for "multilabel";
    None means class1 stratification.
    """
    if method not in SPLIT_METHODS:
        raise ValueError(f"Unknown split_stratification {method!r}. Choose one of {list(SPLIT_METHODS)}.")
    if method == "multilabel" and Y is not None:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

            splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            idx_kept, idx_held = next(splitter.split(np.zeros(n), np.asarray(Y)))
            return idx_kept, idx_held
        except ImportError:
            logger.warning(
                "Package 'iterative-stratification' not installed (pip install iterative-stratification) -- "
                "splitting stratified on class1 instead of on every label."
            )
    return train_test_split(np.arange(n), test_size=test_size, random_state=random_state, stratify=class1)


def split_assignment(
    n: int, test_size: float, validation_size: float, random_state: int,
    class1, Y=None, method: str = "class1",
) -> pd.Series:
    """Per-row "train" / "validation" / "test", reproducing the pipeline's two
    splits exactly: test held out from all rows, then validation held out
    from the remaining training rows (both with the same method).
    """
    class1 = pd.Series(np.asarray(class1))
    idx_train_full, idx_test = stratified_split(n, test_size, random_state, class1, Y, method)
    Y_train_full = None if Y is None else np.asarray(Y)[idx_train_full]
    pos_train, pos_val = stratified_split(
        len(idx_train_full), validation_size, random_state,
        class1.iloc[idx_train_full].reset_index(drop=True), Y_train_full, method,
    )
    assignment = np.empty(n, dtype=object)
    assignment[idx_test] = "test"
    assignment[idx_train_full[pos_train]] = "train"
    assignment[idx_train_full[pos_val]] = "validation"
    return pd.Series(assignment, name="split")


def split_label_table(Y: pd.DataFrame, assignment: pd.Series) -> pd.DataFrame:
    """Per label and split: compounds carrying the label, and that count as
    a share of the split's compounds (prevalence) -- with a good
    stratification each label's prevalence is about the same in every split.
    """
    rows = []
    for split in ("train", "validation", "test"):
        mask = (assignment == split).to_numpy()
        n_split = int(mask.sum())
        counts = Y.loc[mask].sum(axis=0)
        for label, count in counts.items():
            rows.append({"label": label, "split": split, "compounds": int(count),
                         "pct_of_split": count / max(n_split, 1) * 100})
    return pd.DataFrame(rows)


def split_balance(Y: pd.DataFrame, assignment: pd.Series, test_size: float) -> pd.DataFrame:
    """Per label: share of its compounds that landed in the test set, vs the
    target ``test_size`` -- the deviation shows how well stratified it is.
    """
    total = Y.sum(axis=0)
    in_test = Y.loc[(assignment == "test").to_numpy()].sum(axis=0)
    share = in_test / total.replace(0, np.nan)
    return pd.DataFrame({
        "label": Y.columns, "compounds": total.values, "in_test": in_test.values,
        "test_share_pct": share.values * 100,
        "deviation_pts": (share.values - test_size) * 100,
    })
