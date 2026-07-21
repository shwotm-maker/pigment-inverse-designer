"""Rendering helpers: RDKit 2D depictions and matplotlib diagnostic plots.

Also contains the demo-only color <-> wavelength approximation (README section 9)
and a rough wavelength->RGB mapping for visual context. These are illustrative
approximations, NOT colorimetric calculations.
"""
from __future__ import annotations

import colorsys
import io
import zipfile

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from rdkit.Chem import Draw

from .descriptors import mol_from_smiles
from .utils import get_logger

logger = get_logger("visualization")


# ---------------------------------------------------------------------------
# Molecule depiction
# ---------------------------------------------------------------------------
def mol_image(smiles: str, size: tuple[int, int] = (320, 240)):
    """Return a PIL image of the molecule, or None if it cannot be parsed."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Draw.MolToImage(mol, size=size)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to draw %s: %s", smiles, exc)
        return None


def mol_png_bytes(smiles: str, size: tuple[int, int] = (320, 240)) -> bytes | None:
    """Return PNG bytes for a molecule image (for downloads)."""
    img = mol_image(smiles, size)
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def candidates_images_zip(smiles_list: list[str], size: tuple[int, int] = (320, 240)) -> bytes:
    """Bundle candidate structure PNGs into a single ZIP archive (bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, smi in enumerate(smiles_list, start=1):
            png = mol_png_bytes(smi, size)
            if png is not None:
                zf.writestr(f"candidate_{i:03d}.png", png)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Diagnostic plots (return matplotlib Figures for st.pyplot)
# ---------------------------------------------------------------------------
def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs Predicted"):
    """Scatter of measured vs predicted absorption with the y=x reference."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(y_true):
        ax.scatter(y_true, y_pred, alpha=0.5, s=18, edgecolor="none")
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Experimental absorption max (nm)")
    ax.set_ylabel("Predicted absorption max (nm)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    """Histogram of prediction errors (predicted - experimental)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(y_true):
        errors = y_pred - y_true
        ax.hist(errors, bins=30, alpha=0.8)
        ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Prediction error (nm)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    fig.tight_layout()
    return fig


def absorption_histogram(values: np.ndarray, title: str = "Absorption distribution"):
    """Histogram of absorption maxima in the (training) data."""
    fig, ax = plt.subplots(figsize=(5, 4))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values):
        ax.hist(values, bins=30, alpha=0.8, color="#4C72B0")
    ax.set_xlabel("Absorption max (nm)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo-only color <-> wavelength approximation
# ---------------------------------------------------------------------------
def wavelength_to_rgb(wavelength: float) -> tuple[int, int, int]:
    """Rough visible-spectrum wavelength (nm) -> RGB, for illustration only."""
    w = float(wavelength)
    if w < 380 or w > 780:
        return (128, 128, 128)
    if w < 440:
        r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # Intensity fall-off near the edges of the visible range.
    if w < 420:
        factor = 0.3 + 0.7 * (w - 380) / 40
    elif w > 700:
        factor = 0.3 + 0.7 * (780 - w) / 80
    else:
        factor = 1.0
    return tuple(int(max(0.0, min(1.0, c)) * factor * 255) for c in (r, g, b))


def hex_to_absorption_nm(hex_color: str) -> float:
    """DEMO ONLY: map a picked *appearance* color to an approximate absorption
    maximum via its complementary hue.

    There is no one-to-one relationship between perceived color and absorption
    maximum; real color depends on the full spectrum, crystal form, particle
    size, concentration, dispersion state and measurement conditions. Use direct
    nm entry when scientific judgement is required.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Complementary hue (absorbed light is opposite the perceived color).
    comp_hue = (h + 0.5) % 1.0
    # Map hue [0,1) crudely onto the 400-700 nm visible band.
    return round(400.0 + comp_hue * 300.0, 0)
