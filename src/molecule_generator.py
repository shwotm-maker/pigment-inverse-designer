"""Virtual candidate generation via RDKit BRICS decomposition/recombination.

Pipeline (README section 6):
  parents -> BRICS fragments -> BRICS recombination -> validity ->
  dedup -> novelty vs dataset -> constraints -> model prediction -> scoring.

Generated molecules are ALWAYS labelled "Virtual candidate". BRICS
recombination does NOT guarantee real synthesizability.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS, Crippen, Descriptors, rdMolDescriptors

from . import config
from .candidate_search import max_training_similarity, _training_bitvects
from .descriptors import canonical_smiles, mol_from_smiles
from .model import ModelBundle, predict_with_uncertainty
from .scoring import compute_score
from .utils import get_logger, set_global_seed

logger = get_logger("molecule_generator")

ProgressCB = Callable[[float, str], None]


@dataclass
class Constraints:
    """User-configurable molecular filters (README section 7)."""

    min_mol_weight: float = config.DEFAULT_CONSTRAINTS["min_mol_weight"]
    max_mol_weight: float = config.DEFAULT_CONSTRAINTS["max_mol_weight"]
    allowed_elements: list[str] = field(
        default_factory=lambda: list(config.DEFAULT_CONSTRAINTS["allowed_elements"])
    )
    excluded_elements: list[str] = field(
        default_factory=lambda: list(config.DEFAULT_CONSTRAINTS["excluded_elements"])
    )
    max_logp: float = config.DEFAULT_CONSTRAINTS["max_logp"]
    max_formal_charge: int = config.DEFAULT_CONSTRAINTS["max_formal_charge"]
    max_rotatable_bonds: int = config.DEFAULT_CONSTRAINTS["max_rotatable_bonds"]
    min_similarity: float = config.DEFAULT_CONSTRAINTS["min_similarity"]
    max_similarity: float = config.DEFAULT_CONSTRAINTS["max_similarity"]

    def evaluate(self, mol: Chem.Mol, max_similarity_val: float) -> tuple[bool, float, list[str]]:
        """Return (all_passed, satisfied_fraction, failed_reasons)."""
        checks: list[bool] = []
        reasons: list[str] = []

        mw = Descriptors.MolWt(mol)
        ok = self.min_mol_weight <= mw <= self.max_mol_weight
        checks.append(ok)
        if not ok:
            reasons.append(f"MW {mw:.0f} outside [{self.min_mol_weight:.0f}, {self.max_mol_weight:.0f}]")

        elements = {a.GetSymbol() for a in mol.GetAtoms()}
        allowed = set(self.allowed_elements)
        ok = elements.issubset(allowed) if allowed else True
        checks.append(ok)
        if not ok:
            reasons.append(f"disallowed elements: {sorted(elements - allowed)}")

        excluded = set(self.excluded_elements)
        ok = elements.isdisjoint(excluded)
        checks.append(ok)
        if not ok:
            reasons.append(f"excluded elements present: {sorted(elements & excluded)}")

        logp = Crippen.MolLogP(mol)
        ok = logp <= self.max_logp
        checks.append(ok)
        if not ok:
            reasons.append(f"LogP {logp:.1f} > {self.max_logp:.1f}")

        charge = abs(Chem.GetFormalCharge(mol))
        ok = charge <= self.max_formal_charge
        checks.append(ok)
        if not ok:
            reasons.append(f"|formal charge| {charge} > {self.max_formal_charge}")

        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ok = rot <= self.max_rotatable_bonds
        checks.append(ok)
        if not ok:
            reasons.append(f"rotatable bonds {rot} > {self.max_rotatable_bonds}")

        ok = self.min_similarity <= max_similarity_val <= self.max_similarity
        checks.append(ok)
        if not ok:
            reasons.append(
                f"similarity {max_similarity_val:.2f} outside "
                f"[{self.min_similarity:.2f}, {self.max_similarity:.2f}]"
            )

        fraction = sum(checks) / len(checks) if checks else 0.0
        return all(checks), fraction, reasons


def _select_parents(df_with_pred: pd.DataFrame, target_nm: float, n_parents: int) -> list[str]:
    """Pick the closest existing molecules (by experimental abs) as BRICS seeds."""
    work = df_with_pred.dropna(subset=["absorption_max"]).copy()
    work["abs_diff"] = (work["absorption_max"] - target_nm).abs()
    top = work.sort_values("abs_diff").head(n_parents)
    return top["canonical_smiles"].tolist()


def _brics_fragments(parent_smiles: list[str]) -> list[Chem.Mol]:
    """Decompose parents into a de-duplicated set of BRICS fragment mols."""
    frags: set[str] = set()
    for smi in parent_smiles:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            frags |= BRICS.BRICSDecompose(mol)
        except Exception:
            continue
    mols = [Chem.MolFromSmiles(f) for f in frags]
    return [m for m in mols if m is not None]


def _recombine(frag_mols: list[Chem.Mol], build_limit: int) -> list[str]:
    """Run BRICSBuild and return unique canonical SMILES of valid products."""
    seen: set[str] = set()
    if not frag_mols:
        return []
    try:
        builder = BRICS.BRICSBuild(frag_mols, scrambleReagents=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("BRICSBuild failed to start: %s", exc)
        return []

    for mol in itertools.islice(builder, build_limit):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            can = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if can and can not in seen:
            seen.add(can)
    return list(seen)


def generate_candidates(
    df_with_pred: pd.DataFrame,
    bundle: ModelBundle,
    target_nm: float,
    solvent_smiles: str | None,
    constraints: Constraints | None = None,
    tolerance_nm: float = config.DEFAULT_TARGET_TOLERANCE_NM,
    n_parents: int = config.TOP_PARENTS_FOR_BRICS,
    max_candidates: int = config.MAX_VIRTUAL_CANDIDATES,
    build_limit: int = config.BRICS_BUILD_LIMIT,
    n_results: int = config.DEFAULT_N_RESULTS,
    progress_cb: ProgressCB | None = None,
) -> pd.DataFrame:
    """Generate, filter, predict and score virtual candidates.

    Returns a DataFrame ranked by composite score (best first). Never raises for
    individual invalid structures -- they are skipped.
    """
    set_global_seed()
    constraints = constraints or Constraints()

    def report(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(max(0.0, min(1.0, frac)), msg)

    report(0.05, "Selecting parent molecules")
    parents = _select_parents(df_with_pred, target_nm, n_parents)
    if not parents:
        logger.warning("No parent molecules available for BRICS.")
        return pd.DataFrame()

    report(0.15, "Decomposing into BRICS fragments")
    frag_mols = _brics_fragments(parents)
    logger.info("Collected %d BRICS fragments from %d parents", len(frag_mols), len(parents))

    report(0.30, "Recombining fragments (BRICS build)")
    built = _recombine(frag_mols, build_limit)
    logger.info("BRICS produced %d unique valid structures", len(built))

    # Remove structures already present in the dataset (novelty requirement).
    existing = set(df_with_pred["canonical_smiles"].astype(str))
    novel = [s for s in built if s not in existing]
    novel = novel[:max_candidates]
    if not novel:
        logger.warning("No novel structures survived generation.")
        return pd.DataFrame()

    report(0.55, f"Scoring {len(novel)} candidates")
    train_bitvects = _training_bitvects(list(existing))

    preds, unc, valid_idx = predict_with_uncertainty(
        bundle, novel, [solvent_smiles] * len(novel)
    )
    pred_map = {novel[i]: (float(preds[k]), float(unc[k])) for k, i in enumerate(valid_idx)}

    records: list[dict] = []
    total = len(novel)
    for j, smi in enumerate(novel):
        if smi not in pred_map:
            continue
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        pred_nm, unc_nm = pred_map[smi]
        sim = max_training_similarity(smi, train_bitvects)
        passed, fraction, reasons = constraints.evaluate(mol, sim)
        if not passed:
            continue

        delta = pred_nm - target_nm
        breakdown = compute_score(
            delta_nm=delta,
            uncertainty_nm=unc_nm,
            max_similarity=sim,
            constraint_fraction=fraction,
            is_valid=True,
            tolerance_nm=tolerance_nm,
        )
        records.append(
            {
                "candidate_type": "Virtual candidate",
                "canonical_smiles": smi,
                "predicted_absorption_nm": round(pred_nm, 1),
                "experimental_absorption_nm": np.nan,
                "target_difference_nm": round(delta, 1),
                "uncertainty": round(unc_nm, 1),
                "max_training_similarity": round(sim, 3),
                "molecular_weight": round(Descriptors.MolWt(mol), 1),
                "logp": round(Crippen.MolLogP(mol), 2),
                "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
                "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
                "score": breakdown.total,
                "warning": config.CANDIDATE_DISCLAIMER,
            }
        )
        if j % 25 == 0:
            report(0.55 + 0.4 * (j / max(total, 1)), f"Scored {j}/{total}")

    if not records:
        logger.warning("All candidates were filtered out by constraints.")
        return pd.DataFrame()

    report(1.0, "Done")
    result = pd.DataFrame(records).sort_values("score", ascending=False)
    return result.head(n_results).reset_index(drop=True)
