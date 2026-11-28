"""GRIND (GRid-INdependent Descriptors): real GRID-style molecular
interaction fields (MIFs), reduced to an alignment-independent 1D profile
via a MACC-2-style maximum auto-/cross-correlation transform.

Unlike ``field_descriptors.py`` (WHIM/GETAWAY/RDF/MoRSE/AUTOCORR3D --
geometric/statistical descriptors computed directly from atomic
coordinates and simple atomic properties), this is a genuine physics-based
interaction-field descriptor: each grid point's value is an actual
probe-molecule interaction energy, exactly what classical CoMFA/GRID
methods compute -- reduced afterward to distances between a molecule's own
favorable interaction "hotspots" instead of requiring every molecule to be
rigidly aligned in a shared 3D frame first (hence "grid-independent").

Pipeline (Pastor, Cruciani et al. 2000, J. Med. Chem. 43, 3233-3243):
  1. MIF calculation  -- place probes (H-bond acceptor, H-bond donor,
     hydrophobic, electrostatic) on a 3D grid around the molecule and
     compute the probe-molecule interaction energy at every gridpoint,
     using RDKit's ``rdMIF`` module (real GRID-style force-field
     interaction energies, not a geometric approximation).
  2. Node filtering   -- per probe, keep only the most favorable (most
     negative-energy) gridpoints, enforcing a minimum mutual distance so
     several adjacent grid points don't all represent the same physical
     hotspot.
  3. MACC-2 autocorrelation -- for every pair of kept nodes (within one
     probe's own hotspots, or across two different probes' hotspots), bin
     the pair by distance and keep the strongest (most negative) energy
     product observed in each bin. The resulting profile depends only on
     the relative distances between a molecule's own interaction hotspots
     -- never its absolute position or orientation.

RDKit's dedicated "DRY"-style hydrophobic GRID probe isn't available in
``rdMIF``; a plain nonpolar sp3-carbon MMFF van-der-Waals probe (no
electrostatic component) is used as the closest standard equivalent.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMIF

from .config import Config, DEFAULT_CONFIG
from .field_descriptors import _embed_3d

logger = logging.getLogger(__name__)

# (probe name, constructor(mol) -> rdMIF descriptor object). 'O'/'NH' are
# GRID's classic carbonyl-oxygen (H-bond acceptor) and amide-NH (H-bond
# donor) probes; MMFF type 1 is a plain sp3 carbon, the closest available
# stand-in for GRID's DRY hydrophobic probe; Coulomb adds the electrostatic
# field GRID/CoMFA also use alongside the steric/H-bond ones.
_PROBES: List[Tuple[str, Callable[[Chem.Mol], object]]] = [
    ("hbond_acceptor", lambda m: rdMIF.HBond(m, probeAtomType="O")),
    ("hbond_donor", lambda m: rdMIF.HBond(m, probeAtomType="NH")),
    ("hydrophobic", lambda m: rdMIF.MMFFVdWaals(m, probeAtomType=1)),
    ("electrostatic", lambda m: rdMIF.Coulomb(m, probeCharge=1.0)),
]


def _compute_mif(mol: Chem.Mol, probe_name: str, make_descriptor, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (values, coords) -- one interaction energy and one xyz
    position per grid point, for a single probe.
    """
    grid = rdMIF.ConstructGrid(mol, margin=config.grind_grid_margin, spacing=config.grind_grid_spacing)
    descriptor = make_descriptor(mol)
    rdMIF.CalculateDescriptors(grid, descriptor)
    n = grid.GetSize()
    values = np.empty(n, dtype=np.float64)
    coords = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        values[i] = grid.GetVal(i)
        pt = grid.GetGridPointLoc(i)
        coords[i] = (pt.x, pt.y, pt.z)
    return values, coords


def _select_hotspot_nodes(
    values: np.ndarray, coords: np.ndarray, max_nodes: int, min_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily keeps up to ``max_nodes`` of the most favorable (most
    negative-energy) grid points, skipping any candidate closer than
    ``min_distance`` to an already-kept node -- step 2 of GRIND (avoids
    several neighboring grid points all describing the same hotspot).
    Unfavorable (non-negative) points are never selected.
    """
    order = np.argsort(values)  # most negative (favorable) first
    kept_vals: List[float] = []
    kept_coords: List[np.ndarray] = []
    for idx in order:
        if values[idx] >= 0 or len(kept_vals) >= max_nodes:
            break
        pos = coords[idx]
        if kept_coords and min(np.linalg.norm(pos - k) for k in kept_coords) < min_distance:
            continue
        kept_vals.append(values[idx])
        kept_coords.append(pos)
    return np.array(kept_vals), np.array(kept_coords).reshape(-1, 3)


def _macc2_profile(
    nodes_a: Tuple[np.ndarray, np.ndarray], nodes_b: Tuple[np.ndarray, np.ndarray],
    same_probe: bool, bin_width: float, n_bins: int,
) -> np.ndarray:
    """Step 3: bins every (node-from-a, node-from-b) pair by distance and
    keeps the strongest energy product per bin -- the "maximum auto-/
    cross-correlation" MACC-2 encodes. Every node is a favorable
    (negative-energy) hotspot (see ``_select_hotspot_nodes``), so a pair's
    product is always positive, and *larger* means both hotspots are
    simultaneously more favorable -- hence keeping the maximum, not the
    minimum, per bin. ``same_probe`` excludes a node pairing with itself
    and avoids double-counting each unordered pair when a and b are the
    same probe's node list. A bin with no pair at that distance stays 0
    (no signal), which is always distinguishable from a real, positive
    computed value.
    """
    vals_a, coords_a = nodes_a
    vals_b, coords_b = nodes_b
    profile = np.zeros(n_bins)
    for i in range(len(vals_a)):
        j_start = i + 1 if same_probe else 0
        for j in range(j_start, len(vals_b)):
            dist = float(np.linalg.norm(coords_a[i] - coords_b[j]))
            b = int(dist // bin_width)
            if b >= n_bins:
                continue
            product = vals_a[i] * vals_b[j]
            if product > profile[b]:
                profile[b] = product
    return profile


def _column_names(config: Config) -> List[str]:
    names = [p[0] for p in _PROBES]
    cols = []
    for i, a in enumerate(names):
        for b in names[i:]:
            for k in range(config.grind_n_bins):
                cols.append(f"grind_{a}_{b}_bin{k}")
    return cols


def compute_grind_descriptors(mols: List[Optional[Chem.Mol]], config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """GRIND profile for every molecule. Embedding a 3D conformer and
    computing four grid-based interaction fields is slower and less
    reliable than the 2D descriptors elsewhere in this package, so
    molecules that fail embedding (or any field computation) get a NaN
    row -- median-imputed downstream by FeaturePipeline, same as any other
    missing descriptor -- instead of failing the whole run.
    """
    col_names = _column_names(config)
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
        try:
            AllChem.ComputeGasteigerCharges(m3d)
            nodes = {}
            for probe_name, make_descriptor in _PROBES:
                values, coords = _compute_mif(m3d, probe_name, make_descriptor, config)
                nodes[probe_name] = _select_hotspot_nodes(
                    values, coords, config.grind_max_nodes_per_probe, config.grind_min_node_distance,
                )

            row: List[float] = []
            names = [p[0] for p in _PROBES]
            for i, a in enumerate(names):
                for b in names[i:]:
                    profile = _macc2_profile(
                        nodes[a], nodes[b], same_probe=(a == b),
                        bin_width=config.grind_bin_width, n_bins=config.grind_n_bins,
                    )
                    row.extend(profile.tolist())
            rows.append(row)
        except Exception as exc:
            logger.debug("GRIND computation failed for a molecule, using NaN row: %s", exc)
            n_failed += 1
            rows.append([np.nan] * len(col_names))

    if n_failed:
        logger.warning(
            "GRIND descriptors: %d/%d molecules failed conformer embedding or field computation; "
            "those rows are NaN (median-imputed downstream like any other missing descriptor).",
            n_failed, len(mols),
        )
    return pd.DataFrame(rows, columns=col_names)
