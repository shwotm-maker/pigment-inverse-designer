"""Composite 0-100 candidate score.

The score blends five components (README section 8). It is a *screening aid*
only -- it is deliberately NOT called a "synthesizability score" and makes no
claim about real-world synthesis, safety or performance.

Components (default weights, see config.SCORE_WEIGHTS):
* wavelength (40) : closeness of predicted absorption to the target
* uncertainty (20): lower model tree-variance is better
* similarity  (20): reward a novelty "sweet spot" band vs the training set
* constraints (15): fraction of user constraints satisfied
* validity    (5) : structure parses / sanitises cleanly
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class ScoreBreakdown:
    """Per-component sub-scores plus the final 0-100 total."""

    wavelength: float
    uncertainty: float
    similarity: float
    constraints: float
    validity: float

    @property
    def total(self) -> float:
        return round(
            self.wavelength + self.uncertainty + self.similarity
            + self.constraints + self.validity,
            2,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "wavelength": round(self.wavelength, 2),
            "uncertainty": round(self.uncertainty, 2),
            "similarity": round(self.similarity, 2),
            "constraints": round(self.constraints, 2),
            "validity": round(self.validity, 2),
            "total": self.total,
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _wavelength_component(delta_nm: float, tolerance_nm: float) -> float:
    """1.0 when on target, decaying linearly to 0 at 3x the tolerance."""
    span = max(tolerance_nm, 1.0) * 3.0
    return _clamp01(1.0 - abs(delta_nm) / span)


def _uncertainty_component(uncertainty_nm: float) -> float:
    """1.0 for zero variance, 0.0 at/above the reference std."""
    ref = max(config.UNCERTAINTY_REFERENCE_NM, 1.0)
    return _clamp01(1.0 - uncertainty_nm / ref)


def _similarity_component(similarity: float) -> float:
    """Peak inside the novelty sweet-spot band, tapering outside it.

    Too high => barely different from an existing molecule; too low => the
    model is extrapolating and the prediction is less trustworthy.
    """
    low, high = config.SIMILARITY_SWEET_SPOT
    if low <= similarity <= high:
        return 1.0
    if similarity < low:
        return _clamp01(similarity / low) if low > 0 else 1.0
    return _clamp01((1.0 - similarity) / max(1.0 - high, 1e-6))


def compute_score(
    *,
    delta_nm: float,
    uncertainty_nm: float,
    max_similarity: float,
    constraint_fraction: float,
    is_valid: bool,
    tolerance_nm: float = config.DEFAULT_TARGET_TOLERANCE_NM,
    weights: dict[str, float] | None = None,
) -> ScoreBreakdown:
    """Compute the weighted 0-100 candidate score and its breakdown.

    Parameters
    ----------
    delta_nm:
        Predicted absorption minus target (nm); sign ignored.
    uncertainty_nm:
        Tree-variance uncertainty indicator (nm).
    max_similarity:
        Max Tanimoto similarity to the training set (0-1).
    constraint_fraction:
        Fraction of user constraints satisfied (0-1).
    is_valid:
        Whether the structure passed RDKit sanitisation.
    """
    w = weights or config.SCORE_WEIGHTS
    return ScoreBreakdown(
        wavelength=w["wavelength"] * _wavelength_component(delta_nm, tolerance_nm),
        uncertainty=w["uncertainty"] * _uncertainty_component(uncertainty_nm),
        similarity=w["similarity"] * _similarity_component(_clamp01(max_similarity)),
        constraints=w["constraints"] * _clamp01(constraint_fraction),
        validity=w["validity"] * (1.0 if is_valid else 0.0),
    )
