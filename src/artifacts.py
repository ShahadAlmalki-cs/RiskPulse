"""Loaders for the trained artifacts kept in ``models/``.

``load_*`` functions are plain Python (no Streamlit) so they can be reused by
scripts and tests; the app wraps them in ``st.cache_resource``/``st.cache_data``.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from src import config
from src.features import create_model_input


def load_model(path=config.MODEL_PATH):
    """Load the trained classifier."""
    return joblib.load(path)


def load_scaler(path=config.SCALER_PATH):
    """Load the fitted feature scaler saved during training."""
    return joblib.load(path)


def load_metrics(path=config.METRICS_PATH) -> dict[str, Any]:
    """Load the evaluation report produced by ``train_model.py``."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def load_reference_profiles(path=config.CLEAN_DATASET_PATH) -> pd.DataFrame:
    """Load the dataset behind the benchmark tab (no target column).

    Returns an empty frame when the cleaned data is unavailable so the app can
    still score individual applicants.
    """
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, usecols=lambda column: column != config.TARGET_COLUMN)


def default_probability_index(model) -> int:
    """Column of ``predict_proba`` that corresponds to the default class."""
    return list(model.classes_).index(1)


def predict_default_probability(model, values: dict[str, object]) -> float:
    """Score one applicant and return the raw probability of default."""
    model_input = create_model_input(model, values)
    return float(model.predict_proba(model_input)[0][default_probability_index(model)])


def scale_features(scaler, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the training scaler without changing column names or order."""
    if scaler is None or frame.empty:
        return frame
    scaled = scaler.transform(frame)
    return pd.DataFrame(scaled, columns=frame.columns, index=frame.index)
