"""Tests for descriptor / fingerprint featurization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.descriptors import (
    FeatureConfig,
    canonical_smiles,
    compute_descriptors,
    featurize_many,
    featurize_one,
    mol_from_smiles,
)


def test_canonical_and_invalid():
    assert canonical_smiles("c1ccccc1") == "c1ccccc1"
    assert canonical_smiles("OCC") == "CCO"
    assert canonical_smiles("not_a_smiles") is None


def test_compute_descriptors_keys():
    mol = mol_from_smiles("CCO")
    desc = compute_descriptors(mol)
    for key in ["mol_weight", "logp", "tpsa", "aromatic_rings",
                "rotatable_bonds", "num_h_acceptors", "num_h_donors"]:
        assert key in desc
    assert desc["mol_weight"] > 0


def test_featurize_dimension_with_solvent():
    cfg = FeatureConfig(use_solvent=True)
    vec = featurize_one("c1ccccc1", "CCO", cfg)
    assert vec is not None
    assert vec.shape[0] == cfg.dim
    assert vec.shape[0] == cfg.morgan_nbits + len(cfg.descriptor_names) + 3


def test_featurize_without_solvent():
    cfg = FeatureConfig(use_solvent=False)
    vec = featurize_one("c1ccccc1", None, cfg)
    assert vec.shape[0] == cfg.morgan_nbits + len(cfg.descriptor_names)


def test_featurize_many_skips_invalid():
    cfg = FeatureConfig()
    X, idx = featurize_many(["c1ccccc1", "bad", "CCO"], ["", "", ""], cfg)
    assert X.shape[0] == 2
    assert idx == [0, 2]
    assert not np.isnan(X).any()
