import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json

# Executing app.py here is safe: with no assessment in session it ends at
# ``st.stop()`` before rendering anything.
# Session-state keys used by app.py (duplicated so the app module is never
# executed at import: under pytest its bare-mode session state is shared with
# the AppTest sessions below, and a real assessment there leaks into them).
ASSESSMENT_SESSION_KEY = "assessment"
INPUTS_SESSION_KEY = "assessment_inputs"
HISTORY_SESSION_KEY = "assessment_history"


def _fingerprint(values):
    """Mirror of app.inputs_fingerprint (importing app.py requires Streamlit runtime)."""
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def test_fingerprint_changes_when_loan_changes():
    base = {
        "person_age": 30, "person_income": 60000, "person_emp_length": 5.0,
        "person_home_ownership": "RENT", "cb_person_cred_hist_length": 5,
        "cb_person_default_on_file": "N", "loan_amnt": 10000, "loan_intent": "PERSONAL",
        "customer_name": "Sara",
    }
    changed = {**base, "loan_amnt": 30000}
    assert _fingerprint(base) != _fingerprint(changed)
    assert _fingerprint(base) == _fingerprint(dict(base))  # stable for identical inputs


def test_fingerprint_is_order_insensitive():
    a = {"loan_amnt": 10000, "person_age": 30}
    b = {"person_age": 30, "loan_amnt": 10000}
    assert _fingerprint(a) == _fingerprint(b)


# --- Input validation: zero / negative income blocks assessment --------------
import pytest
from streamlit.testing.v1 import AppTest

from src.features import validate_inputs, with_policy_fields
from src.risk import RiskAssessment

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

VALID_INPUT = {
    "person_age": 30,
    "person_income": 60_000,
    "person_emp_length": 5.0,
    "person_home_ownership": "RENT",
    "cb_person_cred_hist_length": 5,
    "cb_person_default_on_file": "N",
    "loan_amnt": 10_000,
    "loan_intent": "PERSONAL",
}


# Only rendered when an assessment is displayed (not the static methodology text).
ASSESSMENT_WIDGETS = (
    "Blended Risk Score",
    "Model Default Probability",
    "Model Margin",
    ":material/balance: LTI",
    "Risk score breakdown",
    "Recommendation",
)


def _field(app: AppTest, label: str):
    return next(widget for widget in app.sidebar.text_input if widget.label == label)


def _set_income(app: AppTest, income_text: str) -> None:
    _field(app, "Annual income").set_value(income_text)


def _assess(app: AppTest, **changes) -> AppTest:
    """Set text fields, press Run assessment, return the run app."""
    for label, value in changes.items():
        _field(app, label.replace("_", " ")).set_value(str(value))
    app.main.button[0].click()
    return _run(app)


def _new_app() -> AppTest:
    """A fresh, isolated app session."""
    return AppTest.from_file(APP_PATH, default_timeout=60)


def _run(app: AppTest) -> AppTest:
    return app.run(timeout=60)


def _assessment_state(app: AppTest) -> RiskAssessment | None:
    return app.session_state.get(ASSESSMENT_SESSION_KEY)


def _sidebar_errors(app: AppTest) -> str:
    """The inline error blocks rendered in the sidebar."""
    return "\n".join(
        value for value in (m.value for m in app.sidebar.markdown) if "rp-field-error" in value
    )


def _rendered_text(app: AppTest) -> str:
    parts = []
    for attribute in ("title", "header", "subheader", "markdown", "caption", "text"):
        for element in getattr(app, attribute, []):
            parts.append(str(getattr(element, "value", element)))
    return "\n".join(parts)


@pytest.mark.parametrize("income_text", ["0", "0.0", "-1", "-50000"])
def test_non_positive_income_blocks_assessment(income_text):
    """Zero or negative income: inline message, no score, no model output."""
    app = _new_app()
    _run(app)
    assert app.exception == []
    valid_assessment = _assessment_state(app)

    _set_income(app, income_text)
    _run(app)

    # No traceback / ValueError surfaced to the user.
    assert app.exception == []

    # Message shown right under the field (inline markdown directly below it).
    assert "Annual income must be greater than 0." in _sidebar_errors(app)
    assert not any(
        "Annual income" in caption.value for caption in app.sidebar.caption
    )

    # Blocked assessment: no new result is produced, nothing is appended.
    assert _assessment_state(app) is valid_assessment
    assert app.session_state.get(HISTORY_SESSION_KEY, []) == []

    # No Model Probability / Blended Risk Score / Policy Decision / LTI rendered.
    rendered = _rendered_text(app)
    for label in ASSESSMENT_WIDGETS:
        assert label not in rendered
    assert app.button[0].disabled is True


@pytest.mark.parametrize("income_text", ["0", "-1"])
def test_non_positive_income_never_computes_lti(income_text):
    """Validation rejects the input before any division by income."""
    values = {**VALID_INPUT, "person_income": float(income_text)}
    with pytest.raises(ValueError, match="Annual income must be greater than zero"):
        validate_inputs(values)
    with pytest.raises(ValueError, match="Annual income must be greater than zero"):
        with_policy_fields(values)


def test_zero_income_leaves_no_stale_outputs():
    """After a valid run, zeroing the income must hide the previous result."""
    app = _new_app()
    _run(app)
    app.button[0].click()
    _run(app)
    assert _assessment_state(app) is not None
    assert "Risk score breakdown" in _rendered_text(app)

    _set_income(app, "0")
    _run(app)

    assert app.exception == []
    rendered = _rendered_text(app)
    for label in ASSESSMENT_WIDGETS:
        assert label not in rendered
    assert "Annual income must be greater than 0." in _sidebar_errors(app)


def test_positive_income_still_assesses():
    """Correct inputs keep producing identical outputs (no behaviour change)."""
    app = _new_app()
    _run(app)
    app.button[0].click()
    _run(app)
    first = _assessment_state(app)
    assert first is not None

    app.button[0].click()
    _run(app)
    second = _assessment_state(app)

    assert app.exception == []
    assert second.model_probability == first.model_probability
    assert second.loan_to_income == first.loan_to_income
    assert second.loan_to_income == pytest.approx(10_000 / 60_000)
    assert second.decision in {"Approve", "Manual review", "Decline"}
    assert "Risk score breakdown" in _rendered_text(app)



# --- Redesign: decision hierarchy, all three outcomes, parity ---------------
from src.artifacts import load_pipeline, predict_default_probability  # noqa: E402
from src.risk import (  # noqa: E402
    blend_signals,
    calculate_model_confidence,
    correct_model_probability,
)


def _reference(values: dict[str, object]) -> RiskAssessment:
    """The score exactly as the production code path computes it (no UI)."""
    raw = predict_default_probability(load_pipeline(), values)
    probability = correct_model_probability(raw)
    lti = float(with_policy_fields(values)["loan_percent_income"])
    contributions = blend_signals(probability, lti, str(values["cb_person_default_on_file"]))
    return RiskAssessment(
        model_probability=probability,
        raw_model_probability=raw,
        model_confidence=calculate_model_confidence(probability),
        loan_to_income=lti,
        loan_grade="?",
        loan_grade_reason="",
        contributions=contributions,
    )


OUTCOME_CASES = {
    # income, loan, prior default -> expected policy decision
    "Approve": (80_000, 5_000, "N"),
    "Manual review": (60_000, 10_000, "N"),
    "Decline": (30_000, 200_000, "Y"),
}


@pytest.mark.parametrize("expected", list(OUTCOME_CASES))
def test_each_policy_outcome_renders(expected):
    """Approve, Manual review and Decline each display with their own styling."""
    income, loan, prior = OUTCOME_CASES[expected]
    app = _new_app()
    _run(app)
    _field(app, "Annual income").set_value(str(income))
    _field(app, "Loan amount").set_value(str(loan))
    app.sidebar.selectbox[1].set_value(prior)  # "Previous default on record"
    app.main.button[0].click()
    _run(app)

    assert app.exception == []
    assessment = _assessment_state(app)
    assert assessment is not None
    assert assessment.decision == expected

    # The decision is the headline, rendered in the result hero.
    hero = [m.value for m in app.main.markdown if 'class="rp-hero"' in m.value]
    assert len(hero) == 1
    assert f'class="rp-hero-title">{expected}<' in hero[0]
    hero = [hero[0]]

    # Only the band that matches this outcome is highlighted.
    band_class = {"Approve": "ok", "Manual review": "review", "Decline": "bad"}[expected]
    assert f"rp-ladder-band--{band_class} band-on" in hero[0]
    assert hero[0].count("band-on") == 1


@pytest.mark.parametrize("expected", list(OUTCOME_CASES))
def test_displayed_scores_match_model_and_policy(expected):
    """The redesign changed presentation only: every number is unchanged."""
    income, loan, prior = OUTCOME_CASES[expected]
    app = _new_app()
    _run(app)
    _field(app, "Annual income").set_value(str(income))
    _field(app, "Loan amount").set_value(str(loan))
    app.sidebar.selectbox[1].set_value(prior)
    app.main.button[0].click()
    _run(app)

    shown = _assessment_state(app)
    values = {**VALID_INPUT, "person_income": income, "loan_amnt": loan,
              "cb_person_default_on_file": prior}
    expected_assessment = _reference(values)

    assert shown.model_probability == pytest.approx(expected_assessment.model_probability)
    assert shown.raw_model_probability == pytest.approx(expected_assessment.raw_model_probability)
    assert shown.loan_to_income == pytest.approx(expected_assessment.loan_to_income)
    assert shown.model_confidence == pytest.approx(expected_assessment.model_confidence)
    assert [c.points for c in shown.contributions] == pytest.approx(
        [c.points for c in expected_assessment.contributions]
    )
    assert shown.score == pytest.approx(expected_assessment.score)
    assert shown.decision == expected


def test_model_and_blended_scores_are_visually_distinct():
    """Model probability and blended policy score carry different labels/tags."""
    app = _new_app()
    _run(app)
    app.main.button[0].click()
    _run(app)

    cards = [
        m.value
        for m in app.main.markdown
        if 'class="rp-signal rp-signal--' in m.value
    ]
    assert len(cards) == 2
    model_card, blend_card = cards
    assert "Model Default Probability" in model_card and "Model output" in model_card
    assert "Blended Risk Score" in blend_card and "Policy overlay" in blend_card
    assert "rp-signal--model" in model_card and "rp-signal--blend" in blend_card
