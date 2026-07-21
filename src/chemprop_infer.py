"""Chemprop (D-MPNN) inference wrapper exposing the same surface the app needs.

The rest of the codebase only calls ``predict_with_uncertainty(bundle, ...)`` and
reads ``bundle.metrics / train_absorption / split_method``. This module provides a
``ChempropBundle`` that satisfies that contract, so swapping the baseline for a
graph neural network requires no changes in the UI, candidate search, or scoring.

Two model flavours are supported, selected by the saved ``meta.joblib`` flag:

* single component (chromophore only)
* multicomponent (chromophore + solvent) with MVE, so the solvent actually
  influences the prediction and each molecule gets a real, per-example
  uncertainty (sqrt of the predicted variance) instead of a flat constant.

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

# Fallback solvent when the caller supplies none (multicomponent needs a valid
# SMILES for every component). Dichloromethane -- the most common solvent in the
# training data -- is a neutral default.
DEFAULT_SOLVENT_SMILES = "ClCCl"


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
    multicomponent: bool = False
    has_mve: bool = False
    default_solvent: str = DEFAULT_SOLVENT_SMILES
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
    multicomponent = bool(meta.get("multicomponent", False))

    if multicomponent:
        model = models.MulticomponentMPNN.load_from_checkpoint(
            str(CHEMPROP_CKPT), map_location="cpu")
    else:
        model = models.MPNN.load_from_checkpoint(str(CHEMPROP_CKPT), map_location="cpu")
    model.eval()

    trainer = pl.Trainer(logger=False, enable_progress_bar=False,
                         accelerator="cpu", devices=1)
    logger.info("Loaded Chemprop bundle (%s, multicomponent=%s, mve=%s)",
                meta.get("version"), multicomponent, meta.get("has_mve"))
    return ChempropBundle(
        model=model,
        trainer=trainer,
        featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(),
        metrics=meta.get("metrics", {}),
        train_absorption=np.asarray(meta.get("train_absorption", []), dtype=float),
        split_method=meta.get("split_method", "scaffold"),
        version=meta.get("version", "chemprop-0.1"),
        multicomponent=multicomponent,
        has_mve=bool(meta.get("has_mve", False)),
        default_solvent=meta.get("default_solvent", DEFAULT_SOLVENT_SMILES),
    )


def _split_mean_var(arr: np.ndarray, n: int):
    """Split a Chemprop prediction array into (mean, variance-or-None).

    MVE outputs two values per target (mean, variance). Layout can be (n, 2),
    (n, 1, 2) or (n, 2) flattened -- handle them defensively.
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 3 and a.shape[-1] >= 2:          # (n, tasks, 2)
        return a[:, 0, 0], a[:, 0, 1]
    if a.ndim == 2 and a.shape[1] == 2:            # (n, 2)
        return a[:, 0], a[:, 1]
    return a.reshape(n, -1)[:, 0], None            # mean only


def predict_chemprop(
    bundle: ChempropBundle,
    chromo_smiles: list[str],
    solvent_smiles: list[str | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Predict absorption max (nm) + uncertainty for a list of SMILES.

    Returns (predictions_nm, uncertainty_nm, valid_indices) to match
    ``model.predict_with_uncertainty``.

    * multicomponent model -> the solvent is a real second input (missing
      solvent falls back to ``bundle.default_solvent``).
    * MVE model -> uncertainty is sqrt(predicted variance), a genuine
      per-molecule value. Otherwise a flat model-level reference (test RMSE).
    """
    import torch
    from chemprop import data

    if solvent_smiles is None:
        solvent_smiles = [None] * len(chromo_smiles)

    valid_idx: list[int] = []
    chromo_dps, solvent_dps = [], []
    for i, smi in enumerate(chromo_smiles):
        if not smi:
            continue
        try:
            c_dp = data.MoleculeDatapoint.from_smi(smi)
        except Exception:
            continue
        if bundle.multicomponent:
            solv = solvent_smiles[i] if i < len(solvent_smiles) else None
            solv = solv or bundle.default_solvent
            try:
                s_dp = data.MoleculeDatapoint.from_smi(solv)
            except Exception:
                try:
                    s_dp = data.MoleculeDatapoint.from_smi(bundle.default_solvent)
                except Exception:
                    continue
            solvent_dps.append(s_dp)
        chromo_dps.append(c_dp)
        valid_idx.append(i)

    if not chromo_dps:
        return np.array([]), np.array([]), []

    if bundle.multicomponent:
        dset = data.MulticomponentDataset([
            data.MoleculeDataset(chromo_dps, bundle.featurizer),
            data.MoleculeDataset(solvent_dps, bundle.featurizer),
        ])
    else:
        dset = data.MoleculeDataset(chromo_dps, bundle.featurizer)

    # drop_last=False: the model is in eval mode, so batch-norm uses running
    # statistics and a final size-1 batch is safe. Without this, Chemprop drops
    # a trailing single sample, desyncing predictions from ``valid_idx``.
    loader = data.build_dataloader(dset, shuffle=False, batch_size=64, drop_last=False)
    with torch.inference_mode():
        batches = bundle.trainer.predict(bundle.model, loader)
    raw = np.concatenate([b.numpy() for b in batches], axis=0)

    n = len(valid_idx)
    mean, var = _split_mean_var(raw, n)
    mean = np.asarray(mean, dtype=float).reshape(-1)

    if bundle.has_mve and var is not None:
        uncertainty = np.sqrt(np.clip(np.asarray(var, dtype=float).reshape(-1), 0, None))
    else:
        ref = float(bundle.metrics.get("test", {}).get("rmse",
                                                        config.UNCERTAINTY_REFERENCE_NM))
        uncertainty = np.full_like(mean, ref)

    return mean, uncertainty, valid_idx
