"""Tests for the composite candidate score."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.scoring import compute_score


def _score(**kw) -> float:
    base = dict(
        delta_nm=0.0, uncertainty_nm=0.0, max_similarity=0.5,
        constraint_fraction=1.0, is_valid=True, tolerance_nm=30.0,
    )
    base.update(kw)
    return compute_score(**base).total


def test_perfect_candidate_is_max():
    total = _score()
    assert abs(total - 100.0) < 1e-6


def test_score_within_bounds():
    for delta in (-200, -30, 0, 30, 200):
        for unc in (0, 20, 80):
            for sim in (0.0, 0.5, 1.0):
                total = _score(delta_nm=delta, uncertainty_nm=unc, max_similarity=sim)
                assert 0.0 <= total <= 100.0


def test_closer_wavelength_scores_higher():
    assert _score(delta_nm=0) > _score(delta_nm=60)


def test_lower_uncertainty_scores_higher():
    assert _score(uncertainty_nm=0) > _score(uncertainty_nm=40)


def test_invalid_loses_validity_points():
    valid = _score(is_valid=True)
    invalid = _score(is_valid=False)
    assert abs((valid - invalid) - config.SCORE_WEIGHTS["validity"]) < 1e-6


def test_constraint_fraction_scales():
    assert _score(constraint_fraction=1.0) > _score(constraint_fraction=0.5)


def test_weights_sum_to_100():
    assert abs(sum(config.SCORE_WEIGHTS.values()) - 100.0) < 1e-6
