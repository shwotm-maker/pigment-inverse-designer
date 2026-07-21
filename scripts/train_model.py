"""End-to-end training pipeline: load -> clean -> split -> train -> save.

Usage
-----
    python -m scripts.train_model
    python -m scripts.train_model --split random
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import load_and_map  # noqa: E402
from src.model import save_bundle, train_model  # noqa: E402
from src.preprocessing import clean_dataframe, split_dataset  # noqa: E402
from src.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger("train_model")


def run(split_method: str = config.SPLIT_METHOD_DEFAULT) -> None:
    """Run the full training pipeline and persist the model bundle."""
    set_global_seed()

    mapped, mapping = load_and_map()
    logger.info("Column mapping resolved to: %s", mapping)

    clean, report = clean_dataframe(mapped)
    logger.info("Cleaning report: %s", report.as_dict())
    if clean.empty:
        raise SystemExit("No clean rows to train on. Check data/raw/ contents.")

    clean.to_csv(config.PROCESSED_CSV, index=False, encoding="utf-8")
    logger.info("Wrote processed data to %s", config.PROCESSED_CSV)

    train_df, val_df, test_df = split_dataset(clean, method=split_method)
    bundle = train_model(train_df, val_df, test_df, split_method=split_method)
    save_bundle(bundle)

    print("\n=== Training complete ===")
    for part, m in bundle.metrics.items():
        print(f"{part:>5}: MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}  R2={m['r2']:.3f}  n={m['n']}")
    print(f"Model saved to: {config.MODEL_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the absorption regressor.")
    parser.add_argument(
        "--split",
        choices=["scaffold", "random"],
        default=config.SPLIT_METHOD_DEFAULT,
        help="Train/val/test split strategy.",
    )
    args = parser.parse_args()
    run(split_method=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
