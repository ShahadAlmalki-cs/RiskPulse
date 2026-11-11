"""Reusable data-preparation helpers shared by the cleaning and training steps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def load_raw_data(path: Path = config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Read the raw credit-risk dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bound ages, impute numeric gaps and one-hot encode categorical columns."""
    cleaned = df[df["person_age"].between(18, 100)].copy()

    # Median is robust to unusually long employment spells and extreme rates.
    for column in ["person_emp_length", "loan_int_rate"]:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    categorical_columns = cleaned.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    return pd.get_dummies(cleaned, columns=list(categorical_columns), dtype=int)


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the default flag."""
    features = df.drop(columns=config.TARGET_COLUMN)
    target = df[config.TARGET_COLUMN]
    return features, target
