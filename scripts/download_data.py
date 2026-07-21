"""Download the 'DB for chromophore' dataset from Figshare.

Usage
-----
    python -m scripts.download_data
    python scripts/download_data.py

On any network failure the script prints instructions for placing a CSV in
data/raw/ manually and always guarantees a usable sample dataset exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import (  # noqa: E402
    download_dataset,
    ensure_sample_dataset,
    fetch_figshare_files,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("download_data")


def main() -> int:
    logger.info("Attempting Figshare download for article %d", config.FIGSHARE_ARTICLE_ID)
    try:
        files = fetch_figshare_files()
        logger.info("Files advertised by Figshare:")
        for f in files:
            logger.info("  - %s (%s bytes)", f.get("name"), f.get("size"))
        saved = download_dataset()
        if saved:
            logger.info("Downloaded %d file(s) to %s", len(saved), config.RAW_DIR)
        else:
            logger.warning("No tabular files were downloaded.")
    except Exception as exc:  # network / API failure
        logger.error("Figshare download failed: %s", exc)
        print(
            "\n[Fallback] Could not download automatically.\n"
            f"1. Open https://figshare.com/articles/dataset/_/{config.FIGSHARE_ARTICLE_ID}\n"
            "2. Download the chromophore CSV/Excel file.\n"
            f"3. Copy it into: {config.RAW_DIR}\n"
            "4. Re-run training: python -m scripts.train_model\n"
        )

    sample = ensure_sample_dataset()
    logger.info("Sample dataset available (%d rows) at %s", len(sample), config.SAMPLE_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
