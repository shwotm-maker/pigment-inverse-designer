"""Chemprop (D-MPNN) inference wrapper exposing the same surface the app needs.

The rest of the codebase only calls ``predict_with_uncertainty(bundle, ...)`` and
reads ``bundle.metrics / train_absorption / split_method``. This module provides a
``ChempropBundle`` that satisfies that contract, so swapping the baseline for a
graph neural network requires no changes in the UI, candidate search, or scoring.

All heavy imports (torch / chemprop / lightning) are done lazily inside functions
so that installations without those packages keep working on the ExtraTrees
baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .utils import get_logger

logger = get_logger("chemprop_infer")

CHEMPROP_DIR = config.MODELS_DIR / "chemprop"
CHEMPROP_CKPT = CHEMPROP_DIR / "model.ckpt"
CHEMPROP_META = CHEMPROP_DIR / "meta.joblib"


@dataclass
class ChempropBundle:
    """Drop-in replacement for ModelBundle backed by a Chemprop D-MPNN."""

    model: object
    trainer: object
    featurizer: object
    metrics: dict = field(default_factory=dict)
    train_absorption: np.ndarray = field(default_factory=lambda: np.array([]))
    split_method: str = "scaffold"
    version: str = "chemprop-0.1"
    feature_config: object = None
    is_chemprop: bool = True


def chemprop_available() -> bool:
    return CHEMPROP_CKPT.exists() and CHEMPROP_META.exists()


def load_chemprop_bundle() -> ChempropBundle:
    """Load the trained checkpoint + metadata into a ChempropBundle."""
    import joblib
    from chemprop import featurizers, models
    from lightning import pytorch as pl

    meta = joblib.load(CHEMPROP_META)
    mpnn = models.MPNN.load_from_checkpoint(str(CHEMPROP_CKPT), map_location="cpu")
    mpnn.eval()
    trainer = pl.Trainer(logger=False, enable_progress_bar=False,
                         accelerator="cpu", devices=1)
    logger.info("Loaded Chemprop bundle (%s)", meta.get("version"))
    return ChempropBundle(
        model=mpnn,
        trainer=trainer,
        featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(),
        metrics=meta.get("metrics", {}),
        train_absorption=np.asarray(meta.get("train_absorption", []), dtype=float),
        split_method=meta.get("split_method", "scaffold"),
        version=meta.get("version", "chemprop-0.1"),
    )


def predict_chemprop(
    bundle: ChempropBundle,
    chromo_smiles: list[str],
    solvent_smiles: list[str | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Predict absorption max (nm) for a list of SMILES.

    Returns (predictions_nm, uncertainty_nm, valid_indices) to match
    ``model.predict_with_uncertainty``. The D-MPNN ignores the solvent (the
    baseline used it as a descriptor); solvent_smiles is accepted and ignored.

    A single D-MPNN has no ensemble variance, so uncertainty is reported as a
    flat model-level reference (the held-out test RMSE) rather than a fake
    per-molecule confidence interval.
    """
    import torch
    from chemprop import data

    valid_idx: list[int] = []
    points = []
    for i, smi in enumerate(chromo_smiles):
        if not smi:
            continue
        try:
            points.append(data.MoleculeDatapoint.from_smi(smi))
            valid_idx.append(i)
        except Exception:
            continue

    if not points:
        return np.array([]), np.array([]), []

    dset = data.MoleculeDataset(points, bundle.featurizer)
    loader = data.build_dataloader(dset, shuffle=False, batch_size=64)
    with torch.inference_mode():
        batches = bundle.trainer.predict(bundle.model, loader)
    preds = np.concatenate([b.numpy() for b in batches]).reshape(-1)

    ref_rmse = float(bundle.metrics.get("test", {}).get("rmse",
                                                         config.UNCERTAINTY_REFERENCE_NM))
    uncertainty = np.full_like(preds, ref_rmse, dtype=float)
    return preds, uncertainty, valid_idx
