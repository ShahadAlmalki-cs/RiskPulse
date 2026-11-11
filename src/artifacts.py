"""Loaders for the trained artifacts kept in ``models/``.

``load_*`` functions are plain Python (no Streamlit) so they can be reused by
scripts and tests; the app wraps them in ``st.cache_resource``/``st.cache_data``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import joblib
import pandas as pd

from src import config
from src.features import assert_schema_compatible, build_model_frame, load_schema


def load_pipeline(path=config.PIPELINE_PATH):
    """Load the fitted preprocessing + classifier pipeline."""
    return joblib.load(path)


def load_metrics(path=config.METRICS_PATH) -> dict[str, Any]:
    """Load the evaluation report produced by ``train_model.py``."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def load_reference_profiles(path=config.REFERENCE_DATASET_PATH) -> pd.DataFrame:
    """Load the deduplicated raw frame behind the portfolio view.

    Returns an empty frame when the reference data is unavailable so the app
    can still score individual applicants.
    """
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame.drop(columns=config.TARGET_COLUMN, errors="ignore")


def default_class_index(pipeline) -> int:
    """Column of ``predict_proba`` that corresponds to the default class."""
    classifier = pipeline.named_steps["model"]
    return list(classifier.classes_).index(1)


def predict_default_probability(pipeline, values: Mapping[str, object]) -> float:
    """Score one applicant and return the probability of default.

    Raises ``ValueError`` on unknown categories (the pipeline's encoder is
    configured with ``handle_unknown="error"``) so callers can surface a clear
    application error instead of a silent zero.
    """
    schema = load_schema()
    frame = build_model_frame(values, schema=schema)
    assert_schema_compatible(frame, schema)
    proba = pipeline.predict_proba(frame)
    return float(proba[0][default_class_index(pipeline)])
