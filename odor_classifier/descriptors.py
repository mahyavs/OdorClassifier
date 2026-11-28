"""RDKit physicochemical descriptors and Mordred 2D descriptors from SMILES."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from .config import Config, DEFAULT_CONFIG

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

_RDKIT_DESCRIPTOR_FUNCS = Descriptors._descList  # list[(name, function)]


def smiles_to_mols(smiles: List[str]) -> List[Optional[Chem.Mol]]:
    return [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]


def compute_rdkit_descriptors(mols: List[Optional[Chem.Mol]]) -> pd.DataFrame:
    """All ~210 RDKit-registered 2D/3D-free molecular descriptors."""
    names = [name for name, _ in _RDKIT_DESCRIPTOR_FUNCS]
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([np.nan] * len(names))
            continue
        values = []
        for _, func in _RDKIT_DESCRIPTOR_FUNCS:
            try:
                values.append(func(mol))
            except Exception:
                values.append(np.nan)
        rows.append(values)
    return pd.DataFrame(rows, columns=[f"rdkit_{n}" for n in names])


def compute_mordred_descriptors(
    mols: List[Optional[Chem.Mol]], config: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """2D Mordred descriptors (~1600). Molecules that fail to parse yield NaN rows."""
    from mordred import Calculator, descriptors as mordred_descriptors

    calc = Calculator(mordred_descriptors, ignore_3D=True)
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_idx]

    if not valid_mols:
        cols = [str(d) for d in calc.descriptors]
        return pd.DataFrame(np.nan, index=range(len(mols)), columns=[f"mordred_{c}" for c in cols])

    result = calc.pandas(valid_mols, nproc=config.mordred_n_proc, quiet=True)
    result = result.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    result.columns = [f"mordred_{c}" for c in result.columns]

    full = pd.DataFrame(np.nan, index=range(len(mols)), columns=result.columns)
    full.iloc[valid_idx] = result.values
    full = full.reset_index(drop=True)
    return full
