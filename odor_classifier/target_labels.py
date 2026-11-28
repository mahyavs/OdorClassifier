"""Authoritative odor-dictionary -> K-column binary target conversion.

Each original odor1..odorN contributes its dictionary class. A compound
receives each distinct class at most once. Existing class1..class10 columns
are never read here and can be absent, incomplete or obsolete.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import pandas as pd

from .config import UC_LABEL

_ODOR_PATTERN = re.compile(r"^odor(\d+)$", flags=re.IGNORECASE)
_MISSING = {"", "nan", "none", "<na>", "null", UC_LABEL.casefold()}


def odor_columns(df: pd.DataFrame) -> list[str]:
    """All odor1, odor2, ... columns, sorted numerically, without a hard limit."""
    columns = [str(c) for c in df.columns if _ODOR_PATTERN.fullmatch(str(c).strip())]
    return sorted(columns, key=lambda c: int(_ODOR_PATTERN.fullmatch(c.strip()).group(1)))


def normalize_term(value: object) -> str:
    """Normalize dictionary lookup keys, never the stored original text."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def dictionary_index(dictionary: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return unique classes in dictionary order and odor -> class mapping.

    Duplicated class rows are harmless, but conflicting mappings of the SAME
    odor term to DIFFERENT classes raise an error rather than silently using
    whichever class happened to be first.
    """
    if "class" not in dictionary.columns:
        raise KeyError("Dictionary must contain a 'class' column.")
    cols = [c for c in dictionary if re.fullmatch(r"type\d+", str(c).strip(), re.I)]
    if not cols:
        raise KeyError("Dictionary must contain type1, type2, ... columns.")
    classes: list[str] = []
    mapping: dict[str, str] = {}
    for _, row in dictionary.iterrows():
        cls = str(row["class"]).strip() if pd.notna(row["class"]) else ""
        if not cls or cls.casefold() in _MISSING:
            continue
        if cls not in classes:
            classes.append(cls)
        for col in cols:
            term = normalize_term(row[col])
            if term in _MISSING:
                continue
            if term in mapping and mapping[term] != cls:
                raise ValueError(
                    f"Dictionary ambiguity: odor {term!r} is assigned to both "
                    f"{mapping[term]!r} and {cls!r}. Resolve this before training."
                )
            mapping[term] = cls
    if not classes:
        raise ValueError("Dictionary contains no valid class names.")
    return classes, mapping


def dictionary_target_audit(
    dictionary: pd.DataFrame,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Account for every dictionary class, including intentionally excluded UC.

    Every trainable dictionary class must have its own target column.
    Classes with zero positives remain columns;
    they are reported rather than quietly omitted.
    """
    trainable, _ = dictionary_index(dictionary)
    result = []
    seen = set()
    for raw in dictionary["class"]:
        cls = str(raw).strip() if pd.notna(raw) else ""
        key = cls.casefold()
        if key in seen:
            continue
        seen.add(key)
        if not cls or key in {"nan", "none", "<na>", "null"}:
            status = "Excluded: empty/invalid class name"
            positives = None
        elif key == UC_LABEL.casefold():
            status = "Excluded: UC is a prediction fallback, not an odor target"
            positives = None
        elif cls not in trainable:
            status = "Excluded: invalid/reserved name"
            positives = None
        elif targets is None:
            status = "Expected target column"
            positives = None
        elif cls in targets.columns:
            positives = int(targets[cls].sum())
            status = ("Present (no positive compounds)" if positives == 0
                      else "Present")
        else:
            status = "MISSING target column — investigate"
            positives = None
        result.append({"Dictionary class": cls or "(blank)",
                       "Status": status, "Positive compounds": positives})
    return pd.DataFrame(result, columns=["Dictionary class", "Status", "Positive compounds"])


def build_binary_targets(
    compounds: pd.DataFrame,
    dictionary: pd.DataFrame,
    *,
    strict_unmapped: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (Y, primary, unmapped_report) with exactly K dictionary columns.

    Y: one binary column per unique dictionary class, in dictionary order.
    primary: first *mapped* original odor's class, used only for single-class
        visualizations and one-target feature selectors, not the multi-label training target.
    unmapped_report: each unknown odor term and source row/column. Unless
        strict_unmapped=False, reject unknown odors before ML training.

    Input DataFrame index is preserved. A row without any known odor receives
    an all-zero target and a missing primary class; callers must explicitly
    choose how to handle such rows, never silently invent a negative label.
    """
    cols = odor_columns(compounds)
    if not cols:
        raise KeyError("No original odor1, odor2, ... columns found. "
                       "Stored class1..class10 columns are intentionally not used.")
    classes, mapping = dictionary_index(dictionary)
    matrix = []
    primaries = []
    unknowns = []
    for pos, (row_idx, row) in enumerate(compounds[cols].iterrows()):
        selected: set[str] = set()
        primary = pd.NA
        for col in cols:
            term = normalize_term(row[col])
            if term in _MISSING:
                continue
            cls = mapping.get(term)
            if cls is None:
                unknowns.append({"row_index": row_idx, "excel_row": pos + 2,
                                 "odor_column": col, "odor_term": str(row[col]).strip()})
                continue
            if pd.isna(primary):
                primary = cls
            selected.add(cls)
        matrix.append([int(cls in selected) for cls in classes])
        primaries.append(primary)
    report = pd.DataFrame(unknowns, columns=["row_index", "excel_row", "odor_column", "odor_term"])
    if strict_unmapped and not report.empty:
        terms = report["odor_term"].value_counts().head(8).to_dict()
        raise ValueError(f"{len(report)} unmapped original odor descriptors; "
                         f"fix the dictionary/input before training. Examples: {terms}")
    Y = pd.DataFrame(matrix, index=compounds.index, columns=classes, dtype="int8")
    primary_series = pd.Series(primaries, index=compounds.index, name="primary_odor_class", dtype="string")
    return Y, primary_series, report
