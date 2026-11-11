"""Turn raw applicant inputs into a model-ready feature row.

The training data was one-hot encoded with ``pd.get_dummies``, so the model
expects every possible category as its own 0/1 column. This module rebuilds
that exact schema from the user's choices and validates the input first.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from src import config
from src.risk import calculate_loan_percent_income, determine_loan_grade

NUMERIC_FEATURES = (
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
)

CATEGORICAL_FEATURES = (
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
)

# ``loan_grade`` is not required: it is a policy output derived in
# ``with_derived_fields``, never entered by the user.
REQUIRED_FIELDS = (
    *NUMERIC_FEATURES,
    *[feature for feature in CATEGORICAL_FEATURES if feature != "loan_grade"],
)

INPUT_LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Employment length (years)",
    "person_home_ownership": "Home ownership",
    "cb_person_cred_hist_length": "Credit history length (years)",
    "cb_person_default_on_file": "Previous default on record",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate (%)",
    "loan_intent": "Loan purpose",
}


def validate_inputs(values: Mapping[str, object]) -> None:
    """Raise ``ValueError`` when a field is missing or out of a sane range."""
    missing = [field for field in REQUIRED_FIELDS if field not in values]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    if not 18 <= float(values["person_age"]) <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if float(values["person_income"]) <= 0:
        raise ValueError("Annual income must be greater than zero.")
    if float(values["loan_amnt"]) < 0:
        raise ValueError("Loan amount cannot be negative.")
    if not 0.0 <= float(values["loan_int_rate"]) <= 100.0:
        raise ValueError("Interest rate must be between 0 and 100.")
    if float(values["person_emp_length"]) < 0:
        raise ValueError("Employment length cannot be negative.")
    if int(values["cb_person_cred_hist_length"]) < 0:
        raise ValueError("Credit history length cannot be negative.")

    allowed = {
        "person_home_ownership": config.HOME_OWNERSHIP_OPTIONS,
        "loan_intent": config.LOAN_INTENT_OPTIONS,
        "cb_person_default_on_file": config.DEFAULT_ON_FILE_OPTIONS,
    }
    # Accepted only when a caller supplies a pre-computed grade explicitly.
    if "loan_grade" in values and values["loan_grade"] not in config.LOAN_GRADES:
        raise ValueError(f"Loan grade must be one of {config.LOAN_GRADES}.")
    for field, options in allowed.items():
        if values[field] not in options:
            raise ValueError(f"{INPUT_LABELS[field]} must be one of {options}.")


def with_derived_fields(values: Mapping[str, object]) -> dict[str, object]:
    """Add the policy loan grade and loan-to-income derived from the inputs."""
    prepared = dict(values)
    loan_to_income = calculate_loan_percent_income(
        float(prepared["loan_amnt"]), float(prepared["person_income"])
    )
    grade, _ = determine_loan_grade(
        loan_to_income, str(prepared["cb_person_default_on_file"])
    )
    prepared["loan_grade"] = grade
    prepared["loan_percent_income"] = loan_to_income
    return prepared


def create_model_input(model, values: Mapping[str, object]) -> pd.DataFrame:
    """Build a one-row frame matching the model's training schema."""
    prepared = with_derived_fields(values)

    input_df = pd.DataFrame(0.0, index=[0], columns=model.feature_names_in_)

    for feature in NUMERIC_FEATURES:
        input_df.loc[0, feature] = float(prepared[feature])
    input_df.loc[0, "loan_percent_income"] = float(prepared["loan_percent_income"])

    for feature in CATEGORICAL_FEATURES:
        encoded_column = f"{feature}_{prepared[feature]}"
        if encoded_column in input_df.columns:
            input_df.loc[0, encoded_column] = 1

    return input_df
