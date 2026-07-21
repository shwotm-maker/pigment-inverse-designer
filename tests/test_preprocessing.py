"""Tests for cleaning, canonicalisation and splitting."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import bemis_murcko_scaffold, clean_dataframe, split_dataset


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chromophore_smiles": [
                "c1ccccc1",            # benzene, valid
                "c1ccccc1",            # duplicate
                "not_a_smiles",        # invalid
                "c1ccc2ccccc2c1",      # naphthalene
                "CCO",                 # valid but missing absorption below
            ],
            "solvent_smiles": ["CCO", "CCO", "CCO", "O", "O"],
            "solvent_name": ["", "", "", "", ""],
            "absorption_max": [255, 255, 300, 275, None],
            "emission_max": [285, 285, None, 322, None],
            "extinction": [None, None, None, None, None],
            "quantum_yield": [0.1, 0.1, 0.2, 0.2, None],
            "source": ["t", "t", "t", "t", "t"],
        }
    )


def test_clean_drops_invalid_and_duplicates():
    clean, report = clean_dataframe(_raw_frame())
    # benzene (dedup to 1) + naphthalene = 2 clean rows.
    assert report.n_invalid_smiles == 1
    assert report.n_missing_target == 1  # CCO row has no absorption
    assert report.n_duplicates == 1
    assert len(clean) == 2
    assert set(clean["canonical_smiles"]) == {"c1ccccc1", "c1ccc2ccccc2c1"}


def test_descriptors_present():
    clean, _ = clean_dataframe(_raw_frame())
    for col in ["mol_weight", "logp", "tpsa", "aromatic_rings", "rotatable_bonds"]:
        assert col in clean.columns
    assert (clean["mol_weight"] > 0).all()


def test_scaffold_string():
    assert bemis_murcko_scaffold("c1ccc2ccccc2c1") != ""
    assert bemis_murcko_scaffold("not_a_smiles") == ""


def test_split_sizes_random():
    df = pd.DataFrame(
        {
            "canonical_smiles": [f"C{'C' * i}" for i in range(1, 21)],
            "solvent_smiles": [""] * 20,
            "absorption_max": list(range(300, 320)),
        }
    )
    train, val, test = split_dataset(df, method="random", test_frac=0.2, val_frac=0.2)
    assert len(train) + len(val) + len(test) == 20
    assert len(test) == 4 and len(val) == 4
