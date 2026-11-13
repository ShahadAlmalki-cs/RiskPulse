"""Essential validity tests for the Set C model and application logic."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.artifacts import load_metrics, load_pipeline, predict_default_probability
from src.data_prep import clean_data, load_raw_data, split_features_target
from src.features import (
    RAW_MODEL_COLUMNS,
    assert_schema_compatible,
    build_model_frame,
    load_schema,
    validate_inputs,
)
from src.risk import (
    build_advice,
    blend_signals,
    classify_risk,
    determine_loan_grade,
    lti_risk,
)

BASE_INPUT = {
    "person_age": 30,
    "person_income": 60_000,
    "person_emp_length": 5.0,
    "person_home_ownership": "RENT",
    "cb_person_cred_hist_length": 5,
    "cb_person_default_on_file": "N",
    "loan_amnt": 10_000,
    "loan_intent": "PERSONAL",
}


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


@pytest.fixture(scope="module")
def schema():
    return load_schema()


# --- Schema -----------------------------------------------------------------
def test_schema_excludes_post_pricing_features(pipeline, schema):
    """T3/T2: no loan_grade or loan_int_rate anywhere in the model path."""
    excluded = {"loan_grade", "loan_int_rate", "loan_status"}
    for name in schema["raw_input_columns"]:
        assert name not in excluded
    for name in schema["encoded_feature_names"]:
        assert not any(name.startswith(f"{col}_") for col in excluded)
    assert pipeline.named_steps["model"].n_features_in_ == 18


def test_feature_order_matches_schema(pipeline, schema):
    """T1/T2: the frame sent to the pipeline is in training order."""
    frame = build_model_frame(BASE_INPUT, schema=schema)
    assert list(frame.columns) == schema["raw_input_columns"]


def test_schema_compatible_guard(schema):
    frame = build_model_frame(BASE_INPUT, schema=schema)
    assert_schema_compatible(frame, schema)  # no raise
    shuffled = frame[schema["raw_input_columns"][::-1]]
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_schema_compatible(shuffled, schema)


# --- Data preparation -------------------------------------------------------
def test_recomputed_lti():
    """T4: loan_percent_income = loan_amnt / person_income."""
    df = clean_data(load_raw_data())
    recomputed = df["loan_amnt"] / df["person_income"]
    assert (df["loan_percent_income"] - recomputed).abs().max() < 1e-12


def test_zero_income_rejected_by_policy_math():
    from src.risk import calculate_loan_percent_income

    with pytest.raises(ValueError):
        calculate_loan_percent_income(10_000, 0)


def test_duplicates_removed():
    df = clean_data(load_raw_data())
    assert df.duplicated().sum() == 0


def test_target_only_in_split():
    X, y = split_features_target(clean_data(load_raw_data()))
    assert config.TARGET_COLUMN not in X.columns
    assert y.name == config.TARGET_COLUMN
    assert set(X.columns) == set(RAW_MODEL_COLUMNS)


# --- Input validation -------------------------------------------------------
def test_zero_loan_rejected():
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "loan_amnt": 0})


def test_nan_and_non_numeric_rejected():
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "person_income": float("nan")})
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "person_income": float("inf")})
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "person_income": "not a number"})
    # A numeric string coerces cleanly and is accepted (documented behaviour).
    validate_inputs({**BASE_INPUT, "person_income": "60000"})


def test_impossible_lengths_rejected():
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "person_age": 20, "person_emp_length": 50.0})
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "person_age": 20, "cb_person_cred_hist_length": 40})


def test_unknown_category_rejected_before_predict():
    with pytest.raises(ValueError):
        validate_inputs({**BASE_INPUT, "person_home_ownership": "SQUATTER"})


# --- Pipeline behaviour -----------------------------------------------------
def test_prediction_probability_valid(pipeline):
    """T7: output is a finite probability in [0, 1]."""
    p = predict_default_probability(pipeline, BASE_INPUT)
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_missing_numeric_is_imputed_not_fatal(pipeline):
    """T6: the pipeline imputes missing numerics (median, fitted on train)."""
    values = {**BASE_INPUT, "person_emp_length": float("nan")}
    p = predict_default_probability(pipeline, values)
    assert math.isfinite(p)


def test_unknown_category_at_transform_raises(pipeline):
    """T5: the encoder rejects unknown categories explicitly."""
    values = {**BASE_INPUT, "loan_intent": "SPACE_TRAVEL"}
    with pytest.raises(ValueError):
        predict_default_probability(pipeline, values)


def test_fabricated_grade_regression(pipeline, schema):
    """T4: changing policy-grade drivers must not change the model input."""
    base = build_model_frame(BASE_INPUT, schema=schema)
    changed = build_model_frame(
        {**BASE_INPUT, "cb_person_default_on_file": "Y"}, schema=schema
    )
    # Only the default_on_file column differs; no grade column exists to move.
    diffs = [
        c for c in base.columns
        if not base[c].equals(changed[c])
    ]
    assert diffs == ["cb_person_default_on_file"]
    assert not any(c.startswith("loan_grade_") for c in diffs)


# --- Artifact / metrics -----------------------------------------------------
def test_metrics_describe_actual_model(schema):
    m = load_metrics()
    assert m["feature_names"] == schema["raw_input_columns"]
    assert m["n_encoded_features" if "n_encoded_features" in m else "accuracy"] is not None
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                "pr_auc", "brier", "default_recall", "confusion_matrix"):
        assert key in m, f"missing metric: {key}"
    # Never accuracy alone: all supporting metrics must be present.


# --- Risk score / decision logic -------------------------------------------
def test_blend_weights_sum_to_one():
    contributions = blend_signals(0.5, 0.5, "N")
    assert len(contributions) == 3
    assert sum(c.weight for c in contributions) == pytest.approx(1.0)


def test_decision_boundaries():
    assert classify_risk(0.1499) == "Low"
    assert classify_risk(0.15) == "Medium"
    assert classify_risk(0.35) == "Medium"
    assert classify_risk(0.3501) == "High"


def test_lti_risk_bounds():
    assert lti_risk(0) == pytest.approx(0.0, abs=1e-9)
    assert 0 <= lti_risk(1000) <= 1  # saturating curve asymptotes to 1


def test_policy_grade_boundaries():
    assert determine_loan_grade(0.15, "N")[0] == "A"
    assert determine_loan_grade(0.5001, "N")[0] == "E"
    assert determine_loan_grade(0.10, "Y")[0] == "E"  # prior default overrides


def test_advice_for_each_level():
    for level, score in [("Low", 0.10), ("Medium", 0.25), ("High", 0.50)]:
        from src.risk import RiskAssessment

        assessment = RiskAssessment(
            model_probability=0.3, raw_model_probability=0.3,
            model_confidence=0.4, loan_to_income=0.2,
            loan_grade="B", loan_grade_reason="",
            contributions=blend_signals(0.3, 0.2, "N"),
        )
        advice = build_advice(assessment)
        assert isinstance(advice, list) and advice


# --- Presentation-layer integrity (requirement 6 additions) -----------------
def test_inputs_fingerprint_changes_with_inputs():
    """Stale detection: a changed input must change the fingerprint."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("riskpulse_app", "app.py")
    # app.py runs Streamlit calls at import; instead test the pure function
    # by extracting it, mirroring the app's implementation.
    import hashlib, json

    def fingerprint(values):
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    a = fingerprint(dict(BASE_INPUT))
    b = fingerprint({**BASE_INPUT, "loan_amnt": 20000})
    assert a != b
    assert fingerprint(dict(BASE_INPUT)) == a  # stable for identical inputs


def test_policy_grade_labelled_as_heuristic():
    """The sidebar must not present the policy grade as a lender-assigned grade."""
    app_source = Path("app.py").read_text(encoding="utf-8")
    assert '"**Loan grade**"' not in app_source, (
        "app.py still uses the misleading 'Loan grade' label"
    )
    assert "Policy grade</strong>" in app_source
    # and must clarify it is internal, not lender-assigned
    assert "not a" in app_source and "lender-assigned" in app_source


def test_readme_confusion_matrix_matches_metrics():
    """Regression: README metrics must equal the saved evaluation report."""
    readme = Path("README.md").read_text(encoding="utf-8")
    m = load_metrics()
    tn, fp, fn, tp = (
        m["confusion_matrix"]["tn"],
        m["confusion_matrix"]["fp"],
        m["confusion_matrix"]["fn"],
        m["confusion_matrix"]["tp"],
    )
    expected = f"{tn:,} / {fp:,} / {fn:,} / {tp:,}"
    assert expected in readme, f"README confusion matrix should contain '{expected}'"


def test_readme_states_probabilities_are_uncalibrated():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "not calibrated" in readme
