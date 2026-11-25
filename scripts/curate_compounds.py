"""One-off curation pass over data/compounds.xlsx:

  1. Remove compounds that are pure elements (every atom -- including
     implicit hydrogens -- is the same element, e.g. bare "[He]", "[S]",
     "O=O"), since these are elemental substances, not odor compounds.
  2. Remove compounds containing any metal atom (alkali, alkaline earth,
     transition, post-transition, lanthanide, actinide -- e.g. Ca, Na),
     anywhere in the (possibly multi-fragment/salt) structure.
  3. Compute molecular weight (RDKit) for every remaining compound and add
     it as a "MolecularWeight" column.
  4. Remove compounds with MolecularWeight > 250 g/mol.

The original file is backed up once (compounds_before_curation.xlsx) before
being overwritten, so the operation is reversible.

Usage:
    python scripts/curate_compounds.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "compounds.xlsx"
BACKUP_PATH = DATA_PATH.with_name("compounds_before_curation.xlsx")
MW_CUTOFF = 250.0

# Metal atomic numbers used for structural curation.
METAL_ATOMIC_NUMS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
}


def is_pure_element(mol: Chem.Mol) -> bool:
    """True if every atom (incl. implicit H, made explicit) is the same element."""
    mol_h = Chem.AddHs(mol)
    elements = {atom.GetSymbol() for atom in mol_h.GetAtoms()}
    return len(elements) == 1


def has_metal(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() in METAL_ATOMIC_NUMS for atom in mol.GetAtoms())


def main() -> None:
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} compounds from {DATA_PATH}")

    if not BACKUP_PATH.exists():
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"Backed up original (untouched) to {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH} (not overwritten)")

    mols = [Chem.MolFromSmiles(s) for s in df["SMILES"].astype(str)]
    n_invalid = sum(m is None for m in mols)
    if n_invalid:
        print(f"WARNING: {n_invalid} SMILES failed to parse; those rows are dropped too.")

    pure_element_mask = [m is not None and is_pure_element(m) for m in mols]
    metal_mask = [m is not None and has_metal(m) for m in mols]
    remove_struct_mask = [
        (m is None) or pe or hm
        for m, pe, hm in zip(mols, pure_element_mask, metal_mask)
    ]

    keep_idx = [i for i, r in enumerate(remove_struct_mask) if not r]
    df_struct = df.iloc[keep_idx].reset_index(drop=True)
    mols_struct = [mols[i] for i in keep_idx]

    df_struct["MolecularWeight"] = [Descriptors.MolWt(m) for m in mols_struct]

    n_before_mw = len(df_struct)
    df_final = df_struct[df_struct["MolecularWeight"] <= MW_CUTOFF].reset_index(drop=True)
    n_removed_mw = n_before_mw - len(df_final)

    df_final.to_excel(DATA_PATH, index=False)

    print()
    print(f"Removed -- invalid SMILES:        {n_invalid}")
    print(f"Removed -- pure elements:          {sum(pure_element_mask)}")
    print(f"Removed -- contains a metal atom:  {sum(metal_mask)}")
    print(f"Removed -- structural (union):      {len(df) - n_before_mw}")
    print(f"Removed -- MolecularWeight > {MW_CUTOFF:g}:  {n_removed_mw}")
    print(f"Final compound count:               {len(df_final)}  (was {len(df)})")
    print(f"Saved curated dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
