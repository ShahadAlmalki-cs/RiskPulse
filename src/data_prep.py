"""Load and clean the raw credit-risk dataset.

This module only prepares *rows*: it filters impossible ages, removes exact
duplicates and recomputes the loan-to-income ratio so it is consistent between
training and inference. All learned transformations (imputation, encoding)
live in the sklearn pipeline so they are fitted on the training split only.
"""

from __future__ import annotations

import pandas as pd

from src import config


def load_raw_data(path=config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Read the raw credit-risk dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning only: age filter, dedup, LTI recomputation.

    Deliberately does NOT impute or encode -- learned transforms belong to the
    fitted pipeline so they are fitted on the training split only.
    """
    cleaned = df[df["person_age"].between(18, 100)].copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    # Recompute rather than trusting the stored column: the dataset's own
    # values disagree with loan_amnt / person_income for ~1.2% of rows.
    cleaned["loan_percent_income"] = (
        cleaned["loan_amnt"] / cleaned["person_income"]
    )
    return cleaned


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the default flag (Set C columns only)."""
    features = df.drop(columns=list(config.MODEL_EXCLUDED_FEATURES))
    target = df[config.TARGET_COLUMN]
    return features, target


def write_reference_profiles(df: pd.DataFrame) -> None:
    """Persist the deduplicated raw frame for the app's portfolio view."""
    df.to_csv(config.REFERENCE_DATASET_PATH, index=False)
