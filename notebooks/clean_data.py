"""Build ``data/clean_credit_risk.csv`` from the raw dataset.

Run from the project root:

    python notebooks/clean_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_prep import clean_data, load_raw_data  # noqa: E402


def main() -> None:
    raw = load_raw_data()
    initial_rows = len(raw)

    clean_df = clean_data(raw)
    clean_df.to_csv(config.CLEAN_DATASET_PATH, index=False)

    print(f"Rows before cleaning: {initial_rows}")
    print(f"Rows after removing invalid ages: {len(clean_df)}")
    print(f"Removed rows: {initial_rows - len(clean_df)}")
    print(f"Saved cleaned data to: {config.CLEAN_DATASET_PATH}")
    print(f"Final shape: {clean_df.shape}")
    print("Missing values in final data:")
    print(clean_df.isnull().sum())


if __name__ == "__main__":
    main()
