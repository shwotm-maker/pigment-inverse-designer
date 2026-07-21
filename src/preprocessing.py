"""Data cleaning, descriptor enrichment and train/val/test splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config
from .descriptors import canonical_smiles, compute_descriptors, mol_from_smiles
from .utils import get_logger, safe_float, set_global_seed

logger = get_logger("preprocessing")


@dataclass
class CleaningReport:
    """Summary statistics produced while cleaning a raw table."""

    n_input: int = 0
    n_invalid_smiles: int = 0
    n_missing_target: int = 0
    n_duplicates: int = 0
    n_output: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.n_input,
            "invalid_smiles": self.n_invalid_smiles,
            "missing_absorption": self.n_missing_target,
            "duplicates_removed": self.n_duplicates,
            "clean_rows": self.n_output,
        }


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate, canonicalise, deduplicate and enrich a canonical-schema frame.

    Steps (README section 3): SMILES validity -> canonical SMILES ->
    numeric absorption -> drop missing target -> dedup -> RDKit descriptors.
    """
    report = CleaningReport(n_input=len(df))
    records: list[dict] = []

    for _, row in df.iterrows():
        raw_smiles = row.get("chromophore_smiles")
        mol = mol_from_smiles(raw_smiles)
        if mol is None:
            report.n_invalid_smiles += 1
            continue

        abs_val = safe_float(row.get("absorption_max"))
        if abs_val is None or not (100.0 <= abs_val <= 1400.0):
            report.n_missing_target += 1
            continue

        can = canonical_smiles(raw_smiles)
        if can is None:
            report.n_invalid_smiles += 1
            continue

        solv_raw = row.get("solvent_smiles")
        solv_can = canonical_smiles(solv_raw)  # None if solvent is a name/blank
        desc = compute_descriptors(mol)

        records.append(
            {
                "canonical_smiles": can,
                "solvent_smiles": solv_can if solv_can is not None else "",
                "solvent_name": _as_str(row.get("solvent_name")),
                "absorption_max": abs_val,
                "emission_max": safe_float(row.get("emission_max")),
                "extinction": safe_float(row.get("extinction")),
                "quantum_yield": safe_float(row.get("quantum_yield")),
                "source": _as_str(row.get("source")) or "unknown",
                **desc,
            }
        )

    clean = pd.DataFrame.from_records(records)
    if clean.empty:
        report.n_output = 0
        logger.warning("Cleaning produced 0 valid rows.")
        return clean, report

    before = len(clean)
    clean = clean.drop_duplicates(subset=["canonical_smiles", "solvent_smiles"]).reset_index(drop=True)
    report.n_duplicates = before - len(clean)
    report.n_output = len(clean)
    logger.info("Cleaning report: %s", report.as_dict())
    return clean, report


def _as_str(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def bemis_murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES ('' if it cannot be computed)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def split_dataset(
    df: pd.DataFrame,
    method: str = config.SPLIT_METHOD_DEFAULT,
    test_frac: float = config.TEST_FRACTION,
    val_frac: float = config.VAL_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test).

    ``method='scaffold'`` keeps molecules that share a Bemis-Murcko scaffold in
    the same split, giving a harder and more honest generalisation estimate.
    ``method='random'`` is simpler but similar structures may leak across
    splits (surfaced to the user in the UI/README).
    """
    set_global_seed(seed)
    if method not in {"scaffold", "random"}:
        raise ValueError(f"Unknown split method: {method!r}")

    if method == "random":
        return _random_split(df, test_frac, val_frac, seed)
    return _scaffold_split(df, test_frac, val_frac, seed)


def _random_split(df, test_frac, val_frac, seed):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = shuffled.iloc[:n_test]
    val = shuffled.iloc[n_test : n_test + n_val]
    train = shuffled.iloc[n_test + n_val :]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _scaffold_split(df, test_frac, val_frac, seed):
    scaffolds: dict[str, list[int]] = {}
    for idx, smi in df["canonical_smiles"].items():
        scaf = bemis_murcko_scaffold(smi) or f"__singleton_{idx}"
        scaffolds.setdefault(scaf, []).append(idx)

    # Largest scaffold groups first -> assign whole groups to fill test/val.
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(df)
    n_test_target = int(round(n * test_frac))
    n_val_target = int(round(n * val_frac))

    test_idx: list[int] = []
    val_idx: list[int] = []
    train_idx: list[int] = []
    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        elif len(val_idx) < n_val_target:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    train = df.loc[train_idx].reset_index(drop=True)
    val = df.loc[val_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)
    logger.info(
        "Scaffold split: train=%d val=%d test=%d (%d scaffolds)",
        len(train), len(val), len(test), len(scaffolds),
    )
    return train, val, test
