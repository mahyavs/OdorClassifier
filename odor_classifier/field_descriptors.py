"""3D 'molecular field' descriptors: WHIM, GETAWAY, RDF, MoRSE, and
AUTOCORR3D -- spatial-autocorrelation and radial-distribution descriptors
weighted by atomic mass/charge/polarizability/van-der-Waals volume, computed
over a single generated+optimized 3D conformer per molecule.

Every other descriptor family in this package (RDKit descriptors, Morgan/
MACCS fingerprints, Mordred, mol2vec) is 2D-only. These five RDKit
calculators are the closest equivalent to the steric/electrostatic
*molecular interaction fields* used in classical 3D-QSAR (CoMFA/CoMSIA)
available without a full grid-based field calculation and the per-compound
molecular alignment step that approach requires -- a practical middle
ground for a library this size, where alignment quality would otherwise
dominate result quality.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# (column prefix, RDKit 3D descriptor function). Output widths aren't
# hardcoded -- they vary slightly across RDKit versions/weighting schemes --
# so they're probed once from a reference molecule (_field_widths) instead.
_CALCULATORS: List[Tuple[str, Callable]] = [
    ("whim", rdMolDescriptors.CalcWHIM),
    ("getaway", rdMolDescriptors.CalcGETAWAY),
    ("rdf", rdMolDescriptors.CalcRDF),
    ("morse", rdMolDescriptors.CalcMORSE),
    ("autocorr3d", rdMolDescriptors.CalcAUTOCORR3D),
]

_widths_cache: Optional[Dict[str, int]] = None


def _embed_3d(mol: Chem.Mol, random_state: int) -> Optional[Chem.Mol]:
    """Adds Hs, embeds one 3D conformer (ETKDGv3), and force-field-optimizes
    it (MMFF94, falling back to UFF if MMFF parameters aren't available for
    this molecule). Returns None if any step fails -- disconnected salts,
    unusual valences, and unlucky embedding seeds can all fail here, and
    those compounds just get a NaN row like any other descriptor failure
    elsewhere in this package.
    """
    try:
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_state
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            AllChem.UFFOptimizeMolecule(m)
        return m
    except Exception:
        return None


def _field_widths() -> Dict[str, int]:
    """Descriptor-vector length per calculator, probed once (module-level
    cache) from a small reference molecule rather than hardcoded -- these
    can vary across RDKit versions, and probing keeps column counts correct
    for whatever RDKit is actually installed.
    """
    global _widths_cache
    if _widths_cache is not None:
        return _widths_cache

    ref = _embed_3d(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"), random_state=0)  # aspirin
    if ref is None:
        raise RuntimeError("Could not embed the reference molecule used to size 3D field descriptors.")
    widths: Dict[str, int] = {}
    for prefix, func in _CALCULATORS:
        try:
            widths[prefix] = len(list(func(ref)))
        except Exception as exc:
            raise RuntimeError(f"Could not determine output width for 3D descriptor '{prefix}': {exc}") from exc
    _widths_cache = widths
    return widths


def compute_field_descriptors(
    mols: List[Optional[Chem.Mol]], config: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """3D field-like descriptors (WHIM/GETAWAY/RDF/MoRSE/AUTOCORR3D) for
    every molecule. Embedding a 3D conformer is much slower and less
    reliable than the 2D descriptors elsewhere in this pipeline, so
    molecules that fail embedding/optimization -- or a specific calculator
    that fails on an otherwise-embedded molecule -- get NaN for just that
    part of the row (median-imputed downstream by FeaturePipeline, same as
    any other missing descriptor) instead of failing the whole run.
    """
    widths = _field_widths()
    col_names = [f"field_{prefix}_{i}" for prefix, w in widths.items() for i in range(w)]

    n_failed = 0
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([np.nan] * len(col_names))
            continue
        m3d = _embed_3d(mol, config.random_state)
        if m3d is None:
            n_failed += 1
            rows.append([np.nan] * len(col_names))
            continue
        values: List[float] = []
        for prefix, func in _CALCULATORS:
            w = widths[prefix]
            try:
                seg = [float(v) for v in func(m3d)]
                if len(seg) != w:
                    seg = (seg + [np.nan] * w)[:w]
            except Exception:
                seg = [np.nan] * w
            values.extend(seg)
        rows.append(values)

    if n_failed:
        logger.warning(
            "3D field descriptors: %d/%d molecules failed conformer embedding/optimization; "
            "those rows are NaN (median-imputed downstream like any other missing descriptor).",
            n_failed, len(mols),
        )
    return pd.DataFrame(rows, columns=col_names)
