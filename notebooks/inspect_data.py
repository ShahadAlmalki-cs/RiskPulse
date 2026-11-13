"""Quick profile of the raw credit-risk dataset.

Run from the project root:

    python notebooks/inspect_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import load_raw_data  # noqa: E402


def main() -> None:
    df = load_raw_data()

    print("First 5 rows:")
    print(df.head())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nColumn info and data types:")
    df.info()

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nSummary statistics for numeric columns:")
    print(df.describe())


if __name__ == "__main__":
    main()
