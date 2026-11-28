"""Descriptive statistics and co-occurrence analysis of dictionary-derived odor labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .labeling import extract_labels
from .multilabel import extract_multilabel_targets, jaccard_cooccurrence

COOCCURRENCE_KINDS = {
    "count": "Compounds with both labels",
    "jaccard": "Jaccard: both / either",
    "conditional": "P(column label | row label)",
    "lift": "Lift: observed / expected if independent",
}


def build_label_matrix(
    compounds: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the binary label matrix and primary class for labeled compounds."""
    primary = extract_labels(compounds, config)
    valid = primary.notna()
    compounds = compounds.loc[valid].reset_index(drop=True)
    primary = primary.loc[valid].reset_index(drop=True)

    Y = extract_multilabel_targets(compounds, config)
    return Y, primary


def label_summary(Y: pd.DataFrame) -> dict:
    per_compound = Y.sum(axis=1)
    counts = Y.sum(axis=0)
    nonzero = counts[counts > 0]
    return {
        "n_compounds": len(Y),
        "n_labels": Y.shape[1],
        "mean_labels": float(per_compound.mean()),
        "max_labels": int(per_compound.max()) if len(Y) else 0,
        "n_unique_combinations": int(len(np.unique(Y.to_numpy(), axis=0))) if len(Y) else 0,
        "n_single_label": int((per_compound == 1).sum()),
        "imbalance_ratio": float(nonzero.max() / nonzero.min()) if len(nonzero) else float("nan"),
    }


def label_counts_table(Y: pd.DataFrame, primary: pd.Series) -> pd.DataFrame:
    """Per label: compounds carrying it, share of compounds, and how many
    carry it as their primary (class1) vs a secondary (class2..class10) label.
    """
    counts = Y.sum(axis=0)
    as_primary = primary.value_counts().reindex(Y.columns, fill_value=0)
    table = pd.DataFrame({
        "label": Y.columns,
        "compounds": counts.values,
        "pct_of_compounds": counts.values / max(len(Y), 1) * 100,
        "as_primary": as_primary.values,
    })
    table["as_primary"] = np.minimum(table["as_primary"], table["compounds"])
    table["as_secondary"] = table["compounds"] - table["as_primary"]
    return table.sort_values("compounds", ascending=False).reset_index(drop=True)


def labels_per_compound(Y: pd.DataFrame) -> pd.Series:
    """Number of compounds carrying exactly k DISTINCT labels, indexed by k
    -- what the multi-label model is trained on (one yes/no per class).
    """
    return Y.sum(axis=1).value_counts().sort_index().rename("compounds")


def class_entries_per_compound(compounds: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.Series:
    """Count nonmissing original odor descriptors (odor1..odorN) per compound.

    This count can be larger than the number of distinct dictionary-mapped
    classes. Stored class annotations are not used.
    """
    from .target_labels import odor_columns, normalize_term

    primary = extract_labels(compounds, config)
    compounds = compounds.loc[primary.notna()].reset_index(drop=True)
    cols = odor_columns(compounds)
    return compounds[cols].apply(
        lambda row: sum(normalize_term(v) not in ("", "nan", "none", "<na>", "null", "uc") for v in row),
        axis=1,
    )


def cooccurrence_matrix(Y: pd.DataFrame, kind: str = "count") -> pd.DataFrame:
    """Label x label co-occurrence. The diagonal is NaN (a label with itself)
    for every kind -- per-label totals are in label_counts_table.

    count:       compounds carrying both labels (symmetric)
    jaccard:     both / either (symmetric, 0-1)
    conditional: P(column | row) = both / compounds with the row label
                 (NOT symmetric: "of the woody compounds, x% are also spicy")
    lift:        P(both) / (P(row) * P(column)); 1 = independent, >1 = occur
                 together more often than chance, <1 = less often (symmetric)
    """
    arr = Y.to_numpy(dtype=float)
    both = arr.T @ arr
    totals = np.diag(both).copy()
    n = max(len(Y), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "count":
            mat = both.copy()
        elif kind == "jaccard":
            mat = jaccard_cooccurrence(Y).to_numpy(dtype=float)
        elif kind == "conditional":
            mat = both / totals[:, None]
        elif kind == "lift":
            mat = (both / n) / np.outer(totals / n, totals / n)
        else:
            raise ValueError(f"Unknown co-occurrence kind {kind!r}. Choose one of {list(COOCCURRENCE_KINDS)}.")
    mat = np.where(np.isfinite(mat), mat, np.nan)
    np.fill_diagonal(mat, np.nan)
    return pd.DataFrame(mat, index=Y.columns, columns=Y.columns)


def clustered_label_order(Y: pd.DataFrame) -> list[str]:
    """Labels ordered by hierarchical clustering on Jaccard distance, so
    labels that tend to occur together sit next to each other in the heatmap.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    if Y.shape[1] < 3:
        return list(Y.columns)
    dist = 1.0 - jaccard_cooccurrence(Y).to_numpy(dtype=float)
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return [Y.columns[i] for i in order]


def top_label_pairs(Y: pd.DataFrame, sort_by: str = "count", n: int = 20, min_count: int = 1) -> pd.DataFrame:
    """Strongest label pairs, each with every co-occurrence measure.
    ``min_count`` drops pairs seen together too rarely for lift/conditional
    values to mean anything.
    """
    labels = list(Y.columns)
    mats = {kind: cooccurrence_matrix(Y, kind).to_numpy() for kind in COOCCURRENCE_KINDS}
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            count = mats["count"][i, j]
            if not count >= min_count:
                continue
            rows.append({
                "label_a": labels[i], "label_b": labels[j], "count": int(count),
                "jaccard": mats["jaccard"][i, j],
                "P(b | a)": mats["conditional"][i, j], "P(a | b)": mats["conditional"][j, i],
                "lift": mats["lift"][i, j],
            })
    pairs = pd.DataFrame(rows, columns=["label_a", "label_b", "count", "jaccard", "P(b | a)", "P(a | b)", "lift"])
    if sort_by == "conditional":  # either direction counts -- rank by the stronger one
        order = pairs[["P(b | a)", "P(a | b)"]].max(axis=1).sort_values(ascending=False).index
        pairs = pairs.loc[order]
    else:
        pairs = pairs.sort_values(sort_by, ascending=False)
    return pairs.head(n).reset_index(drop=True)
