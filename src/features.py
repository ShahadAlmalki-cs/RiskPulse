"""Validate applicant inputs and build a Set C model-ready row.

The model expects the nine raw application-time columns listed in
``config.MODEL_NUMERIC_FEATURES`` / ``MODEL_CATEGORICAL_FEATURES``; all
imputation and encoding is done by the fitted pipeline, so this module only
validates and assembles the raw frame.
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from src import config
from src.risk import calculate_loan_percent_income, determine_loan_grade

# The exact raw columns the pipeline was trained on (Set C).
RAW_MODEL_COLUMNS = (*config.MODEL_NUMERIC_FEATURES, *config.MODEL_CATEGORICAL_FEATURES)

# Fields a caller must supply; loan_percent_income is derived, not entered.
REQUIRED_FIELDS = (
    *config.RAW_INPUT_NUMERIC_FIELDS,
    *config.RAW_INPUT_CATEGORICAL_FIELDS,
)

INPUT_LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Employment length (years)",
    "person_home_ownership": "Home ownership",
    "cb_person_cred_hist_length": "Credit history length (years)",
    "cb_person_default_on_file": "Previous default on record",
    "loan_amnt": "Loan amount",
    "loan_intent": "Loan purpose",
}


def _as_finite_float(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{INPUT_LABELS[field]} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{INPUT_LABELS[field]} must be a finite number.")
    return number


def validate_inputs(values: Mapping[str, object]) -> None:
    """Raise ``ValueError`` when a field is missing, malformed or impossible."""
    missing = [field for field in REQUIRED_FIELDS if field not in values]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    numeric = {
        field: _as_finite_float(values[field], field)
        for field in config.RAW_INPUT_NUMERIC_FIELDS
    }

    if not 18 <= numeric["person_age"] <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if numeric["person_income"] <= 0:
        raise ValueError("Annual income must be greater than zero.")
    if numeric["loan_amnt"] <= 0:
        raise ValueError("Loan amount must be greater than zero.")
    if numeric["person_emp_length"] < 0:
        raise ValueError("Employment length cannot be negative.")
    if numeric["cb_person_cred_hist_length"] < 0:
        raise ValueError("Credit history length cannot be negative.")
    if numeric["person_emp_length"] > numeric["person_age"]:
        raise ValueError("Employment length cannot exceed age.")
    if numeric["cb_person_cred_hist_length"] > numeric["person_age"]:
        raise ValueError("Credit history length cannot exceed age.")

    allowed = {
        "person_home_ownership": config.HOME_OWNERSHIP_OPTIONS,
        "loan_intent": config.LOAN_INTENT_OPTIONS,
        "cb_person_default_on_file": config.DEFAULT_ON_FILE_OPTIONS,
    }
    for field, options in allowed.items():
        if values[field] not in options:
            raise ValueError(
                f"{INPUT_LABELS[field]} must be one of: {', '.join(options)}."
            )


def with_policy_fields(values: Mapping[str, object]) -> dict[str, object]:
    """Add derived LTI and the *policy* loan grade (display only, never a model input)."""
    supplied = tuple(c for c in RAW_MODEL_COLUMNS if c != "loan_percent_income")
    prepared = {field: values[field] for field in supplied}
    loan_to_income = calculate_loan_percent_income(
        float(prepared["loan_amnt"]), float(prepared["person_income"])
    )
    grade, reason = determine_loan_grade(
        loan_to_income, str(prepared["cb_person_default_on_file"])
    )
    prepared["loan_percent_income"] = loan_to_income
    prepared["loan_grade"] = grade
    prepared["loan_grade_reason"] = reason
    return prepared


def build_model_frame(values: Mapping[str, object], schema: dict | None = None) -> pd.DataFrame:
    """One-row raw frame in the exact trained column order."""
    prepared = with_policy_fields(values)
    columns = list(schema["raw_input_columns"]) if schema else list(RAW_MODEL_COLUMNS)
    frame = pd.DataFrame(
        [{column: prepared[column] for column in columns}],
        columns=columns,
    )
    for field in config.MODEL_NUMERIC_FEATURES:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype(float)
    for field in config.MODEL_CATEGORICAL_FEATURES:
        frame[field] = frame[field].astype(object)
    return frame


def load_schema(path=config.SCHEMA_PATH) -> dict:
    """Load the persisted feature schema."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def assert_schema_compatible(frame: pd.DataFrame, schema: dict) -> None:
    """Fail loudly if the frame does not match the persisted training schema."""
    expected = schema["raw_input_columns"]
    actual = list(frame.columns)
    if actual != expected:
        raise ValueError(
            "Model input schema mismatch. Expected columns "
            f"{expected}, got {actual}."
        )
