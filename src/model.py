"""Baseline absorption-wavelength regressor (ExtraTrees / RandomForest).

The model bundle saved to disk is a dict so that inference can reproduce the
exact featurization used at training time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .descriptors import FeatureConfig, featurize_many
from .utils import get_logger, set_global_seed, timer

logger = get_logger("model")

TARGET_COLUMN = "absorption_max"


@dataclass
class ModelBundle:
    """Everything required to make and interpret predictions."""

    model: object
    feature_config: FeatureConfig
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    train_absorption: np.ndarray = field(default_factory=lambda: np.array([]))
    split_method: str = config.SPLIT_METHOD_DEFAULT
    version: str = "0.1.0"


def _make_estimator() -> object:
    common = dict(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_leaf=getattr(config, "MIN_SAMPLES_LEAF", 1),
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_SEED,
    )
    if config.MODEL_TYPE == "random_forest":
        return RandomForestRegressor(**common)
    return ExtraTreesRegressor(**common)


def _build_xy(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Featurize a cleaned frame into (X, y), dropping unfeaturizable rows."""
    X, valid_idx = featurize_many(
        df["canonical_smiles"].tolist(),
        df["solvent_smiles"].tolist(),
        cfg,
    )
    y = df.iloc[valid_idx][TARGET_COLUMN].to_numpy(dtype=np.float32)
    return X, y


def _evaluate(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "n": int(len(y)),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    split_method: str = config.SPLIT_METHOD_DEFAULT,
) -> ModelBundle:
    """Train the regressor and evaluate on train/val/test partitions."""
    set_global_seed()
    cfg = feature_config or FeatureConfig()

    with timer("featurize+fit", logger):
        X_train, y_train = _build_xy(train_df, cfg)
        if len(y_train) == 0:
            raise ValueError("No trainable rows after featurization.")
        model = _make_estimator()
        model.fit(X_train, y_train)

    metrics: dict[str, dict[str, float]] = {"train": _evaluate(model, X_train, y_train)}
    for name, part in (("val", val_df), ("test", test_df)):
        if part is not None and len(part):
            Xp, yp = _build_xy(part, cfg)
            metrics[name] = _evaluate(model, Xp, yp)
        else:
            metrics[name] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}

    logger.info("Metrics: %s", metrics)
    return ModelBundle(
        model=model,
        feature_config=cfg,
        metrics=metrics,
        train_absorption=y_train,
        split_method=split_method,
    )


def predict_with_uncertainty(
    bundle: ModelBundle,
    chromo_smiles: list[str],
    solvent_smiles: list[str | None],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Predict absorption + a tree-variance uncertainty indicator.

    Returns (predictions, uncertainty_nm, valid_indices). ``uncertainty_nm`` is
    the standard deviation of per-tree predictions -- an internal model-variance
    reference value, NOT a statistical confidence interval (README section 4).
    """
    cfg = bundle.feature_config
    X, valid_idx = featurize_many(chromo_smiles, solvent_smiles, cfg)
    if len(valid_idx) == 0:
        return np.array([]), np.array([]), []

    estimators = getattr(bundle.model, "estimators_", None)
    if estimators:
        per_tree = np.stack([est.predict(X) for est in estimators], axis=0)
        mean_pred = per_tree.mean(axis=0)
        std_pred = per_tree.std(axis=0)
    else:  # pragma: no cover - non-ensemble fallback
        mean_pred = bundle.model.predict(X)
        std_pred = np.zeros_like(mean_pred)
    return mean_pred, std_pred, valid_idx


def save_bundle(bundle: ModelBundle, path: Path = config.MODEL_PATH) -> Path:
    """Persist a model bundle with joblib (compressed to keep the file small)."""
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved model bundle to %s", path)
    return path


def load_bundle(path: Path = config.MODEL_PATH) -> ModelBundle | None:
    """Load a model bundle, returning None if it does not exist."""
    if not Path(path).exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load model bundle: %s", exc)
        return None
