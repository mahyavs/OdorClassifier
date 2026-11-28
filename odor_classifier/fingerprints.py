"""Structural fingerprints: Morgan (ECFP-like) and MACCS keys."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

from .config import Config, DEFAULT_CONFIG


def compute_morgan_fingerprints(
    mols: List[Optional[Chem.Mol]], config: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    # includeChirality: R/S chiral centres written in the SMILES change the
    # atom environments, so specified stereoisomers get different bits.
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.morgan_radius, fpSize=config.morgan_n_bits,
        includeChirality=config.morgan_use_chirality,
    )
    n_bits = config.morgan_n_bits
    rows = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        rows[i] = arr
    cols = [f"morgan_{i}" for i in range(n_bits)]
    return pd.DataFrame(rows, columns=cols)


def compute_maccs_keys(mols: List[Optional[Chem.Mol]]) -> pd.DataFrame:
    n_bits = 167
    rows = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        rows[i] = arr
    cols = [f"maccs_{i}" for i in range(n_bits)]
    return pd.DataFrame(rows, columns=cols)
