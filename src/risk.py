"""Risk calculations, risk blending and decision advice.

Everything here is pure math: no Streamlit, no model. That keeps the policy
easy to unit-test and reusable outside the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src import config

# Weights in the order they appear in the contribution chart.
WEIGHTS: dict[str, float] = {
    "Model": config.MODEL_WEIGHT,
    "LTI": config.LTI_WEIGHT,
    "History": config.HISTORY_WEIGHT,
}

SIGNAL_LABELS = {
    "Model": "Model probability",
    "LTI": "Loan-to-income",
    "History": "Prior default",
}

RISK_LEVELS = ("Low", "Medium", "High")


# --- Input ratios ----------------------------------------------------------
def calculate_loan_percent_income(loan_amount: float, annual_income: float) -> float:
    """Return the loan-to-income ratio and reject impossible inputs."""
    if annual_income <= 0:
        raise ValueError("Annual income must be greater than zero.")
    if loan_amount < 0:
        raise ValueError("Loan amount cannot be negative.")
    return loan_amount / annual_income


def determine_loan_grade(
    loan_to_income: float, previous_default: str
) -> tuple[str, str]:
    """Assign a policy grade from the loan-to-income ratio and default history."""
    if previous_default == config.PRIOR_DEFAULT_FLAG:
        return config.DEFAULT_LOAN_GRADE, (
            "Grade E: a prior default is on record."
        )
    for threshold, grade in config.GRADE_THRESHOLDS:
        if loan_to_income <= threshold:
            return grade, (
                f"Grade {grade}: loan-to-income is at most {threshold:.0%} "
                "and there is no prior default."
            )
    return config.DEFAULT_LOAN_GRADE, (
        f"Grade {config.DEFAULT_LOAN_GRADE}: loan-to-income exceeds "
        f"{config.GRADE_THRESHOLDS[-1][0]:.0%}."
    )


# --- Model outputs ---------------------------------------------------------
def correct_model_probability(model_probability: float) -> float:
    """Keep the model probability inside the [2%, 98%] guardrail."""
    if not math.isfinite(model_probability) or not 0.0 <= model_probability <= 1.0:
        raise ValueError("Model probability must be a finite value between 0 and 1.")
    return min(
        max(model_probability, config.MODEL_PROBABILITY_MIN),
        config.MODEL_PROBABILITY_MAX,
    )


def calculate_model_confidence(default_probability: float) -> float:
    """Measure model certainty as the gap between its two class probabilities."""
    return abs(default_probability - (1.0 - default_probability))


def lti_risk(loan_to_income: float) -> float:
    """Map loan-to-income to 0-1 with a smooth saturating curve.

    Risk starts near zero for tiny loans and approaches 1 as the loan dwarfs
    the income, without ever jumping between hard buckets.
    """
    return -math.expm1(-max(loan_to_income, 0.0) / config.LTI_RISK_SCALE)


def history_risk(previous_default: str) -> float:
    """1.0 when a prior default exists on file, otherwise 0.0."""
    return 1.0 if previous_default == config.PRIOR_DEFAULT_FLAG else 0.0


# --- Blending --------------------------------------------------------------
@dataclass(frozen=True)
class Contribution:
    """One weighted risk signal inside the blended score."""

    signal: str
    weight: float
    value: float  # the signal's own 0-1 risk
    points: float  # weight * value, i.e. contribution to the final score

    @property
    def label(self) -> str:
        return SIGNAL_LABELS[self.signal]


@dataclass(frozen=True)
class RiskAssessment:
    """Everything the dashboard needs to describe one applicant."""

    model_probability: float  # clipped model probability
    raw_model_probability: float
    model_confidence: float
    loan_to_income: float
    loan_grade: str
    loan_grade_reason: str
    contributions: list[Contribution] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(item.points for item in self.contributions)

    @property
    def level(self) -> str:
        return classify_risk(self.score)

    @property
    def decision(self) -> str:
        return DECISION_BY_LEVEL[self.level]


def blend_signals(
    model_probability: float,
    loan_to_income: float,
    previous_default: str,
) -> list[Contribution]:
    """Split the blended score into its three weighted contributions."""
    signals = {
        "Model": (model_probability, WEIGHTS["Model"]),
        "LTI": (lti_risk(loan_to_income), WEIGHTS["LTI"]),
        "History": (history_risk(previous_default), WEIGHTS["History"]),
    }
    return [
        Contribution(signal=name, weight=weight, value=value, points=weight * value)
        for name, (value, weight) in signals.items()
    ]


def calculate_blended_risk(
    model_probability: float,
    loan_to_income: float,
    previous_default: str,
) -> float:
    """Blend the model, loan-to-income and history signals into one score."""
    score = sum(item.points for item in blend_signals(
        model_probability, loan_to_income, previous_default
    ))
    return max(0.0, min(1.0, score))


def classify_risk(score: float) -> str:
    """Bucket a blended score into the policy risk levels."""
    if score < config.LOW_RISK_MAX:
        return "Low"
    if score <= config.MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


DECISION_BY_LEVEL = {
    "Low": "Approve",
    "Medium": "Manual review",
    "High": "Decline",
}


# --- Recommendations -------------------------------------------------------
_DECISION_ADVICE = {
    "Low": [
        "Proceed with a standard credit decision and document the approval.",
        "Offer the requested repayment schedule; no extra conditions required.",
        "Re-run the assessment if the loan amount or income changes.",
    ],
    "Medium": [
        "Route the file to a credit analyst before any commitment is made.",
        "Request proof of income (payslips or bank statements) for the last 3 months.",
        "Consider a smaller loan amount or a shorter tenor to lower the ratio.",
        "Add a guarantor or a higher first instalment to offset the risk.",
    ],
    "High": [
        "Decline the current request or offer a supervised alternative.",
        "Reduce exposure: a much smaller amount over a shorter tenor.",
        "Require full documentation and a co-signer before reconsidering.",
        "If BNPL is the goal, restrict the applicant to low-value instalments.",
    ],
}

_DRIVER_ADVICE = {
    "LTI": "The loan is large relative to income; a smaller amount or a longer "
    "tenor would improve the score the most.",
    "History": "A recorded prior default weighs heavily; only settlement history "
    "and a co-signer can mitigate it.",
    "Model": "The model sees a profile similar to past defaults; review income "
    "stability, employment length and interest rate.",
}


def build_advice(assessment: RiskAssessment) -> list[str]:
    """Return automatic next steps for the decision plus the top risk driver."""
    advice = list(_DECISION_ADVICE[assessment.level])
    top_driver = max(assessment.contributions, key=lambda item: item.points)
    if top_driver.points > 0:
        advice.append(f"Top driver: {_DRIVER_ADVICE[top_driver.signal]}")
    return advice
