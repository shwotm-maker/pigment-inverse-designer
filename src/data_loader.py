"""Dataset acquisition and loading.

Responsibilities
----------------
* Query the Figshare public API for the "DB for chromophore" article and
  resolve the real download URL of the CSV/Excel file(s).
* Load any CSV (downloaded, user-supplied, or the built-in sample) and map its
  columns onto our canonical schema using keyword rules from ``config``.
* Provide a self-contained sample dataset so the whole app runs offline.

The canonical schema produced by :func:`load_and_map` is:
    chromophore_smiles, solvent_smiles, solvent_name,
    absorption_max, emission_max, extinction, quantum_yield, source
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from . import config
from .utils import get_logger

logger = get_logger("data_loader")

CANONICAL_COLUMNS = [
    "chromophore_smiles",
    "solvent_smiles",
    "solvent_name",
    "absorption_max",
    "emission_max",
    "extinction",
    "quantum_yield",
]


# ---------------------------------------------------------------------------
# Figshare API
# ---------------------------------------------------------------------------
def fetch_figshare_files(article_id: int = config.FIGSHARE_ARTICLE_ID, timeout: int = 30) -> list[dict]:
    """Return the list of file records for a Figshare article.

    Each record contains at least ``name`` and ``download_url``. Raises
    ``requests.RequestException`` on network errors so the caller can fall back.
    """
    url = config.FIGSHARE_API_URL.format(article_id=article_id)
    logger.info("Querying Figshare API: %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    logger.info("Figshare returned %d file(s)", len(files))
    return files


def download_dataset(
    article_id: int = config.FIGSHARE_ARTICLE_ID,
    dest_dir: Path = config.RAW_DIR,
    timeout: int = 120,
) -> list[Path]:
    """Download all tabular files (.csv/.xlsx/.xls) from the article.

    Returns the list of saved paths. Never raises for individual file failures;
    logs and continues so a partial download is still usable.
    """
    saved: list[Path] = []
    files = fetch_figshare_files(article_id, timeout=timeout)
    for f in files:
        name = f.get("name", "")
        dl = f.get("download_url")
        if not dl or not name.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        dest = dest_dir / name
        try:
            logger.info("Downloading %s ...", name)
            r = requests.get(dl, timeout=timeout)
            r.raise_for_status()
            dest.write_bytes(r.content)
            saved.append(dest)
            logger.info("Saved %s (%d bytes)", dest, len(r.content))
        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", name, exc)
    return saved


# ---------------------------------------------------------------------------
# Reading raw tables
# ---------------------------------------------------------------------------
def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/Excel file into a DataFrame, trying a few encodings."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # Last resort: let pandas raise a helpful error.
    return pd.read_csv(path)


def find_raw_table(raw_dir: Path = config.RAW_DIR) -> Path | None:
    """Return the most relevant raw data file, preferring real over sample."""
    candidates = sorted(
        [p for p in raw_dir.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    )
    if not candidates:
        return None
    # Prefer a non-sample file if one exists.
    non_sample = [p for p in candidates if "sample" not in p.name.lower()]
    return (non_sample or candidates)[0]


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------
def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Map raw columns onto the canonical schema using keyword rules.

    Returns (mapped_df, mapping) where mapping[target] is the original column
    name (or None if not found). Missing targets are created as empty columns
    so downstream code never KeyErrors.
    """
    lower_to_orig = {str(c).lower().strip(): c for c in df.columns}
    mapping: dict[str, str | None] = {}
    claimed: set[str] = set()

    for target, keywords in config.COLUMN_MAPPING_RULES.items():
        found: str | None = None
        for kw in keywords:
            for low, orig in lower_to_orig.items():
                if orig in claimed:
                    continue
                if kw in low:
                    found = orig
                    break
            if found:
                break
        if found:
            claimed.add(found)
        mapping[target] = found

    out = pd.DataFrame()
    for target in CANONICAL_COLUMNS:
        src = mapping.get(target)
        out[target] = df[src] if src is not None else pd.NA
    return out, mapping


def load_and_map(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load a raw table (or the sample) and map it to the canonical schema."""
    if path is None:
        path = find_raw_table()
    if path is None:
        logger.warning("No raw data file found; using built-in sample dataset.")
        df = ensure_sample_dataset()
    else:
        logger.info("Loading raw table: %s", path)
        df = _read_table(path)
    mapped, mapping = map_columns(df)
    mapped["source"] = (path.name if path is not None else "built-in sample")
    return mapped, mapping


# ---------------------------------------------------------------------------
# Built-in offline sample dataset
# ---------------------------------------------------------------------------
# (name, chromophore SMILES, solvent SMILES, absorption_max_nm, emission_max_nm, quantum_yield)
# Approximate literature-style values, curated for a runnable proof-of-concept.
# These are illustrative only and must not be treated as authoritative data.
_SAMPLE_ROWS: list[tuple[str, str, str, float, float, float]] = [
    ("Benzene", "c1ccccc1", "C1CCCCC1", 255, 285, 0.05),
    ("Naphthalene", "c1ccc2ccccc2c1", "C1CCCCC1", 275, 322, 0.23),
    ("Anthracene", "c1ccc2cc3ccccc3cc2c1", "Cc1ccccc1", 375, 400, 0.36),
    ("Tetracene", "c1ccc2cc3cc4ccccc4cc3cc2c1", "Cc1ccccc1", 474, 490, 0.21),
    ("Pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", "C1CCCCC1", 334, 375, 0.65),
    ("Perylene", "c1cc2cccc3c2c2c1cccc2c1cccc23", "ClCCl", 438, 447, 0.94),
    ("Stilbene (trans)", "C(=C/c1ccccc1)\\c1ccccc1", "Cc1ccccc1", 295, 350, 0.05),
    ("Biphenyl", "c1ccc(-c2ccccc2)cc1", "C1CCCCC1", 250, 315, 0.18),
    ("Azobenzene", "c1ccc(/N=N/c2ccccc2)cc1", "CCO", 320, 0, 0.01),
    ("Methyl yellow", "CN(C)c1ccc(/N=N/c2ccccc2)cc1", "CCO", 410, 0, 0.01),
    ("Methyl orange", "CN(C)c1ccc(/N=N/c2ccc(S(=O)(=O)O)cc2)cc1", "O", 465, 0, 0.01),
    ("Methyl red", "CN(C)c1ccc(/N=N/c2ccccc2C(=O)O)cc1", "CCO", 430, 0, 0.01),
    ("4-Nitroaniline", "Nc1ccc([N+](=O)[O-])cc1", "CCO", 380, 0, 0.02),
    ("N,N-dimethyl-4-nitroaniline", "CN(C)c1ccc([N+](=O)[O-])cc1", "CCO", 420, 0, 0.02),
    ("Nitrobenzene", "O=[N+]([O-])c1ccccc1", "CCO", 268, 0, 0.01),
    ("Aniline", "Nc1ccccc1", "CCO", 280, 340, 0.10),
    ("Phenol", "Oc1ccccc1", "CCO", 270, 300, 0.10),
    ("Coumarin", "O=c1ccc2ccccc2o1", "CCO", 310, 360, 0.05),
    ("7-Hydroxycoumarin", "O=c1ccc2cc(O)ccc2o1", "CCO", 325, 450, 0.60),
    ("Coumarin 153", "CCN1CCCc2cc3ccc(=O)oc3c(C(F)(F)F)c21", "CCO", 420, 530, 0.53),
    ("Fluorescein", "O=C1OC2(c3ccccc31)c1ccc(O)cc1Oc1cc(O)ccc12", "O", 490, 514, 0.79),
    ("Rhodamine B", "CCN(CC)c1ccc2c(c1)Oc1cc(=[N+](CC)CC)ccc1C2c1ccccc1C(=O)O", "CCO", 554, 576, 0.65),
    ("Rhodamine 6G", "CCNc1cc2oc3cc(=[NH+]CC)c(C)cc3c(-c3ccccc3C(=O)OCC)c2cc1C", "CCO", 530, 552, 0.95),
    ("Nile red", "CCN(CC)c1ccc2c(c1)oc1cc(=O)c3ccccc3c1c2", "CCO", 553, 636, 0.70),
    ("Methylene blue", "CN(C)c1ccc2nc3ccc(=[N+](C)C)cc3sc2c1", "O", 665, 686, 0.04),
    ("Acridine orange", "CN(C)c1ccc2nc3ccc(=[N+](C)C)cc3cc2c1", "O", 490, 530, 0.20),
    ("BODIPY core", "Cc1cc2n(c1)[B-](F)(F)n1c(C)cc(C)c1C2", "ClCCl", 505, 516, 0.90),
    ("Quinoline", "c1ccc2ncccc2c1", "CCO", 275, 360, 0.10),
    ("Carbazole", "c1ccc2c(c1)[nH]c1ccccc12", "C1CCOC1", 290, 350, 0.40),
    ("Fluorene", "C1c2ccccc2-c2ccccc21", "C1CCCCC1", 265, 305, 0.68),
    ("Dibenzofuran", "c1ccc2c(c1)oc1ccccc12", "C1CCCCC1", 280, 310, 0.30),
    ("Indole", "c1ccc2[nH]ccc2c1", "CCO", 270, 350, 0.30),
    ("Porphyrin (free base)", "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1[nH]2)n5)[nH]4)n3", "ClC(Cl)Cl", 620, 720, 0.10),
    ("Phthalocyanine core", "c1ccc2c(c1)C1=Nc3nc(nc4[nH]c(nc5nc(nc(n2)c2ccccc21)c1ccccc15)c1ccccc341)c1ccccc31", "ClC(Cl)Cl", 680, 700, 0.60),
    ("Merocyanine 540", "CCCCN1C(=O)N(CCCC)C(=O)/C1=C/C=C/C=C1\\Sc2ccccc2[N+]1(CC)CCCS(=O)(=O)[O-]", "CO", 555, 578, 0.35),
    ("DCM dye", "CC1=CC(=CC(C)(C)N1)/C=C/c1ccc(N(C)C)cc1", "CS(=O)C", 470, 610, 0.44),
    ("Cyanine (Cy3-like)", "CC1(C)c2ccccc2N(CC)/C1=C/C=C/C1=[N+](CC)c2ccccc2C1(C)C", "CO", 550, 565, 0.10),
    ("Cyanine (Cy5-like)", "CC1(C)c2ccccc2N(CC)/C1=C/C=C/C=C/C1=[N+](CC)c2ccccc2C1(C)C", "CO", 650, 670, 0.28),
    ("Crystal violet", "CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccc(N(C)C)cc2)cc1", "O", 590, 0, 0.01),
    ("Malachite green", "CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccccc2)cc1", "O", 620, 0, 0.01),
    ("Alizarin", "O=C1c2ccccc2C(=O)c2c1ccc(O)c2O", "CCO", 430, 0, 0.02),
    ("Anthraquinone", "O=C1c2ccccc2C(=O)c2ccccc21", "CCO", 325, 0, 0.01),
    ("Indigo", "O=C1/C(=C2\\Nc3ccccc3C2=O)Nc2ccccc21", "CS(=O)C", 610, 0, 0.01),
    ("Curcumin", "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O", "CO", 425, 500, 0.05),
    ("beta-Carotene", "CC(=CC=CC=C(C)C=CC=C(C)C=CC1=C(C)CCCC1(C)C)C=CC=C(C)C=CC1=C(C)CCCC1(C)C", "C1CCCCC1", 450, 0, 0.01),
    ("Dansyl", "CN(C)c1cccc2c(S(=O)(=O)N)cccc12", "CC#N", 335, 520, 0.30),
    ("NBD amine", "Nc1ccc([N+](=O)[O-])c2nonc12", "CCO", 470, 540, 0.30),
    ("Pyrromethene", "Cc1cc(C)n(c1)[B-](F)(F)n1c(C)cc(C)c1", "CCO", 490, 510, 0.85),
    ("Eosin Y", "O=C1OC2(c3cc(Br)c(O)c(Br)c3Oc3c(Br)c(O)c(Br)cc32)c2ccccc21", "CCO", 524, 543, 0.65),
    ("Perylene diimide", "O=C1N(C)C(=O)c2ccc3c4c(ccc1c24)C(=O)N(C)C3=O", "ClC(Cl)Cl", 525, 535, 0.99),
    ("Naphthalimide", "O=C1N(C)C(=O)c2cccc3cccc1c23", "CCO", 340, 400, 0.30),
    ("Squaraine", "CCN(CC)c1ccc(cc1)C1=CC(=O)C(=C1[O-])c1ccc(cc1)N(CC)CC", "ClCCl", 630, 645, 0.55),
    ("Terthiophene", "c1csc(-c2ccc(-c3cccs3)s2)c1", "C1CCOC1", 350, 430, 0.10),
    ("Fluoranthene", "c1ccc-2c(c1)-c1cccc3cccc-2c13", "Cc1ccccc1", 358, 462, 0.30),
    ("Acridone", "O=c1c2ccccc2[nH]c2ccccc12", "CCO", 400, 430, 0.80),
    ("Xanthone", "O=c1c2ccccc2oc2ccccc12", "CCO", 340, 0, 0.02),
    ("Thioflavin T", "Cc1ccc(cc1)-c1sc2cc(N(C)C)ccc2[n+]1C", "O", 412, 480, 0.02),
]


def build_sample_dataframe() -> pd.DataFrame:
    """Construct the built-in sample dataset as a canonical-schema DataFrame."""
    rows = []
    for name, smi, solv, abs_nm, emi_nm, qy in _SAMPLE_ROWS:
        rows.append(
            {
                "Name": name,
                "Chromophore": smi,
                "Solvent": solv,
                "Absorption max (nm)": abs_nm,
                "Emission max (nm)": emi_nm if emi_nm else pd.NA,
                "log(e/mol-1 dm3 cm-1)": pd.NA,
                "Quantum yield": qy,
            }
        )
    return pd.DataFrame(rows)


def ensure_sample_dataset(path: Path = config.SAMPLE_CSV) -> pd.DataFrame:
    """Write the sample dataset to ``path`` if missing and return it."""
    df = build_sample_dataframe()
    if not path.exists():
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Wrote sample dataset to %s (%d rows)", path, len(df))
    return df


def read_uploaded_csv(file_like: io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Read a Streamlit-uploaded file object into a DataFrame."""
    try:
        return pd.read_csv(file_like)
    except Exception:
        file_like.seek(0)
        return pd.read_excel(file_like)
