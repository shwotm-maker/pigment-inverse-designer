"""Search existing dataset molecules close to a target absorption wavelength.

Two rankings are offered (README section 5):
* by experimental absorption maximum
* by model-predicted absorption maximum
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rdkit import DataStructs

from . import config
from .descriptors import mol_from_smiles, morgan_bitvect
from .model import ModelBundle, predict_with_uncertainty
from .utils import get_logger

logger = get_logger("candidate_search")


def add_model_predictions(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Attach predicted absorption + uncertainty columns to a cleaned frame."""
    out = df.copy()
    out["predicted_absorption_nm"] = np.nan
    out["uncertainty"] = np.nan
    preds, unc, valid_idx = predict_with_uncertainty(
        bundle,
        out["canonical_smiles"].tolist(),
        out["solvent_smiles"].tolist(),
    )
    if valid_idx:
        out.iloc[valid_idx, out.columns.get_loc("predicted_absorption_nm")] = preds
        out.iloc[valid_idx, out.columns.get_loc("uncertainty")] = unc
    return out


def _training_bitvects(train_smiles: list[str]) -> list:
    """Precompute Morgan bit vectors for the training set (skip invalid)."""
    vects = []
    for smi in train_smiles:
        mol = mol_from_smiles(smi)
        if mol is not None:
            vects.append(morgan_bitvect(mol))
    return vects


def max_training_similarity(smiles: str, train_bitvects: list) -> float:
    """Max Tanimoto similarity of ``smiles`` against the training fingerprints."""
    mol = mol_from_smiles(smiles)
    if mol is None or not train_bitvects:
        return 0.0
    fp = morgan_bitvect(mol)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_bitvects)
    return float(max(sims)) if sims else 0.0


def search_existing(
    df_with_pred: pd.DataFrame,
    target_nm: float,
    by: str = "experimental",
    tolerance_nm: float = config.DEFAULT_TARGET_TOLERANCE_NM,
    n_results: int = config.DEFAULT_N_RESULTS,
    solvent_filter: str | None = None,
) -> pd.DataFrame:
    """Return existing molecules ranked by closeness to ``target_nm``.

    Parameters
    ----------
    by:
        'experimental' ranks on measured absorption; 'predicted' ranks on the
        model prediction (useful where experimental values are sparse).
    solvent_filter:
        Optional exact canonical solvent SMILES to restrict results.
    """
    if by not in {"experimental", "predicted"}:
        raise ValueError("by must be 'experimental' or 'predicted'")
    col = "absorption_max" if by == "experimental" else "predicted_absorption_nm"

    work = df_with_pred.copy()
    if solvent_filter:
        work = work[work["solvent_smiles"] == solvent_filter]
    work = work.dropna(subset=[col])
    if work.empty:
        return work

    work["target_difference_nm"] = work[col] - float(target_nm)
    work["abs_difference"] = work["target_difference_nm"].abs()
    within = work[work["abs_difference"] <= tolerance_nm]
    ranked = (within if not within.empty else work).sort_values("abs_difference")
    return ranked.head(n_results).reset_index(drop=True)
