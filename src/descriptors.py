"""Molecular descriptors and fingerprints.

The featurization used at *training* time and at *inference* time MUST be
identical, so both paths go through :func:`featurize_one` / :func:`featurize_many`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

from . import config
from .utils import get_logger

# RDKit is very chatty about parse failures; we handle them ourselves.
RDLogger.DisableLog("rdApp.*")

logger = get_logger("descriptors")


@dataclass
class FeatureConfig:
    """Snapshot of everything needed to reproduce a feature vector."""

    morgan_radius: int = config.MORGAN_RADIUS
    morgan_nbits: int = config.MORGAN_NBITS
    use_solvent: bool = config.USE_SOLVENT_FEATURES
    descriptor_names: list[str] = field(default_factory=lambda: list(config.DESCRIPTOR_NAMES))

    @property
    def n_solvent_features(self) -> int:
        return 3 if self.use_solvent else 0

    @property
    def dim(self) -> int:
        return self.morgan_nbits + len(self.descriptor_names) + self.n_solvent_features


def mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
    """Parse SMILES to an RDKit Mol, returning None on any failure."""
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:  # pragma: no cover - RDKit rarely raises here
        return None


def canonical_smiles(smiles: str | None) -> str | None:
    """Return canonical SMILES or None if the input cannot be parsed."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Compute the RDKit descriptor block used across the app."""
    return {
        "mol_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
        "num_h_acceptors": float(Lipinski.NumHAcceptors(mol)),
        "num_h_donors": float(Lipinski.NumHDonors(mol)),
    }


def morgan_fingerprint(mol: Chem.Mol, radius: int, nbits: int) -> np.ndarray:
    """Return a Morgan (ECFP-like) bit vector as a float32 numpy array."""
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    ConvertToNumpyArray(bitvect, arr)
    return arr


def morgan_bitvect(mol: Chem.Mol, radius: int | None = None, nbits: int | None = None):
    """Return the raw RDKit ExplicitBitVect (for Tanimoto similarity)."""
    radius = config.MORGAN_RADIUS if radius is None else radius
    nbits = config.MORGAN_NBITS if nbits is None else nbits
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _solvent_block(solvent_smiles: str | None) -> np.ndarray:
    """3-value numeric descriptor block for the solvent (zeros if unknown)."""
    mol = mol_from_smiles(solvent_smiles)
    if mol is None:
        return np.zeros(3, dtype=np.float32)
    return np.array(
        [Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol)],
        dtype=np.float32,
    )


def featurize_one(
    chromo_smiles: str,
    solvent_smiles: str | None,
    cfg: FeatureConfig,
) -> np.ndarray | None:
    """Build the full feature vector for one (chromophore, solvent) pair.

    Returns None when the chromophore SMILES is invalid so callers can skip it.
    """
    mol = mol_from_smiles(chromo_smiles)
    if mol is None:
        return None
    try:
        fp = morgan_fingerprint(mol, cfg.morgan_radius, cfg.morgan_nbits)
        desc = compute_descriptors(mol)
        desc_vec = np.array([desc[name] for name in cfg.descriptor_names], dtype=np.float32)
        parts = [fp, desc_vec]
        if cfg.use_solvent:
            parts.append(_solvent_block(solvent_smiles))
        return np.concatenate(parts).astype(np.float32)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Featurization failed for %s: %s", chromo_smiles, exc)
        return None


def featurize_many(
    chromo_smiles: Sequence[str],
    solvent_smiles: Sequence[str | None] | None,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[int]]:
    """Featurize many molecules, returning (X, valid_indices).

    Rows whose chromophore SMILES fail parsing are silently dropped; the
    returned index list maps X rows back to positions in the input sequence.
    """
    if solvent_smiles is None:
        solvent_smiles = [None] * len(chromo_smiles)
    rows: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, (cs, ss) in enumerate(zip(chromo_smiles, solvent_smiles)):
        vec = featurize_one(cs, ss, cfg)
        if vec is not None:
            rows.append(vec)
            valid_idx.append(i)
    if not rows:
        return np.empty((0, cfg.dim), dtype=np.float32), []
    return np.vstack(rows), valid_idx


def feature_names(cfg: FeatureConfig) -> list[str]:
    """Human-readable name for every feature column (for debugging/export)."""
    names = [f"morgan_{i}" for i in range(cfg.morgan_nbits)]
    names += list(cfg.descriptor_names)
    if cfg.use_solvent:
        names += ["solvent_mw", "solvent_logp", "solvent_tpsa"]
    return names
