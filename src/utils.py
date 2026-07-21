"""Shared utilities: logging, RNG seeding, small helpers."""
from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from . import config

_LOGGER_CONFIGURED = False


def get_logger(name: str = "pigment") -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)


def set_global_seed(seed: int | None = None) -> None:
    """Seed Python, NumPy and hash randomisation for reproducibility."""
    seed = config.RANDOM_SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that logs elapsed wall-clock time for a block."""
    log = logger or get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s finished in %.2fs", label, time.perf_counter() - start)


def safe_float(value: object) -> float | None:
    """Best-effort conversion to float, returning None on failure."""
    try:
        if value is None:
            return None
        f = float(str(value).strip())
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
