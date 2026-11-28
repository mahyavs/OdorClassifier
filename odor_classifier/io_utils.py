"""Loading the two source Excel files and building the odor -> class dictionary."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Config, DEFAULT_CONFIG

_UNNAMED_COLUMN_RE = re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE)


def _is_unnamed_column(raw) -> bool:
    s = "" if raw is None else str(raw).strip()
    return s == "" or s.lower() in ("none", "nan") or bool(_UNNAMED_COLUMN_RE.match(s))


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops trailing/blank columns that carry no real header -- pandas's
    own "Unnamed: N" placeholder, an empty string, or a genuinely missing
    header cell (which different read engines can hand back as literal
    None or NaN). A real-world Excel export commonly has a block of cells
    formatted (but never filled in) far past the actual data range, which
    both openpyxl and pandas read back as extra header-less columns --
    left in place, these carry no usable data or identity and flood every
    column picker (SMILES/Name/Color/Feature) with dozens or hundreds of
    meaningless duplicate entries, e.g. "(none)" x342 in a picker that
    lists every column by name.
    """
    keep = [c for c in df.columns if not _is_unnamed_column(c)]
    return df[keep]


def read_table(path: Path | str) -> pd.DataFrame:
    """Reads a compound sheet from either a CSV or an Excel (.xlsx/.xls)
    file, dispatching by file extension. The built-in compounds.xlsx is
    Excel, but other inputs may be CSV. Strips out any
    trailing header-less columns (see drop_unnamed_columns) before
    returning, so every caller gets a clean column list without needing to
    handle that itself.
    """
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    return drop_unnamed_columns(df)


def load_compounds(path: Path | str | None = None, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load the compound sheet (SMILES, odor descriptors, rule-based classes)."""
    path = Path(path) if path is not None else config.compounds_path
    df = read_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    if config.smiles_column not in df.columns:
        raise KeyError(
            f"Column '{config.smiles_column}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[config.smiles_column]).reset_index(drop=True)
    df[config.smiles_column] = df[config.smiles_column].astype(str).str.strip()
    return df


def load_class_dictionary(path: Path | str | None = None, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Read and validate the class dictionary used by multi-label training.

    Each distinct value in ``class`` is one target. ``type1``, ``type2``, ...
    map original odor words to that target. No fixed class count is assumed.
    """
    from .target_labels import dictionary_index

    path = Path(path) if path is not None else Path(config.class_dictionary_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Class dictionary not found: {path.resolve()}. "
            f"Place class_dictionary.xlsx in the project's data folder "
            f"({path.parent.resolve()}) and rerun the command."
        )
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "class" not in df.columns:
        raise KeyError(f"Column 'class' not found in {path}. Available columns: {list(df.columns)}")
    # Catch missing type columns, blank classes and ambiguous odor mappings
    # before a long descriptor/model run.
    dictionary_index(df)
    return df


def build_odor_to_class_map(class_dict_df: pd.DataFrame) -> Dict[str, str]:
    """Flatten the type1..typeN columns of the class dictionary into a lookup.

    Returns a mapping of lowercase/stripped odor word -> class name. If the
    same odor word appears under two different class rows (shouldn't happen
    in a well-formed dictionary, but we don't assume it), the first
    occurrence wins.
    """
    type_cols = [c for c in class_dict_df.columns if c.lower().startswith("type")]
    mapping: Dict[str, str] = {}
    for _, row in class_dict_df.iterrows():
        class_name = str(row["class"]).strip()
        for col in type_cols:
            val = row[col]
            if pd.isna(val):
                continue
            word = str(val).strip().lower()
            if not word:
                continue
            mapping.setdefault(word, class_name)
    return mapping


def get_known_classes(class_dict_df: pd.DataFrame) -> list[str]:
    """Unique, ordered list of class names defined in the dictionary (excludes UC)."""
    seen: list[str] = []
    for name in class_dict_df["class"].astype(str).str.strip():
        if name not in seen:
            seen.append(name)
    return seen
