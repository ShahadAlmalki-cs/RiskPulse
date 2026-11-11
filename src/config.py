"""Central configuration for RiskPulse.

Every path is resolved from the project root so the app behaves the same no
matter which directory ``streamlit run`` is launched from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Directories -----------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATASET_PATH = DATA_DIR / "credit_risk_dataset.csv"
REFERENCE_DATASET_PATH = DATA_DIR / "reference_profiles.csv"
PIPELINE_PATH = MODELS_DIR / "risk_pipeline.pkl"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- Model training --------------------------------------------------------
TARGET_COLUMN = "loan_status"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# --- Set C feature schema --------------------------------------------------
# Pre-pricing, pre-approval credit risk assessment.
#
# Deliberately excluded:
#   loan_grade    -- post-decision pricing label assigned by the lender;
#                    98% predictable from loan_int_rate (not an applicant input).
#   loan_int_rate -- set by the lender at pricing, after the assessment
#                    this product models. Excluding it keeps the model valid
#                    for a decision made before any rate is quoted.
#   loan_status   -- the target; never a predictor.
MODEL_NUMERIC_FEATURES = (
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
)
MODEL_CATEGORICAL_FEATURES = (
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
)
MODEL_EXCLUDED_FEATURES = ("loan_grade", "loan_int_rate", TARGET_COLUMN)

# Raw application inputs the UI collects (subset of the numeric features;
# loan_percent_income is derived, not entered).
RAW_INPUT_NUMERIC_FIELDS = (
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "cb_person_cred_hist_length",
)
RAW_INPUT_CATEGORICAL_FIELDS = (
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
)

# --- Risk blending ---------------------------------------------------------
# NOTE: these weights are policy choices, not statistically optimised, and they
# are documented as such in the README. LTI and prior default already enter the
# model as features, so the blend is an overlay, not an independent signal.
MODEL_WEIGHT = 0.40
LTI_WEIGHT = 0.40
HISTORY_WEIGHT = 0.20

# Smoothing scale of the loan-to-income risk curve (larger = more forgiving).
LTI_RISK_SCALE = 0.35
# Keep model probabilities away from absolute certainty before blending.
MODEL_PROBABILITY_MIN = 0.02
MODEL_PROBABILITY_MAX = 0.98
# Only a recorded prior default pushes the history signal.
PRIOR_DEFAULT_FLAG = "Y"

# --- Decision policy -------------------------------------------------------
LOW_RISK_MAX = 0.15  # score < 15%        -> approve
MEDIUM_RISK_MAX = 0.35  # 15% <= score <= 35% -> manual review
# Above MEDIUM_RISK_MAX -> decline / route to collections-grade terms.

# --- Policy loan grades ----------------------------------------------------
LOAN_GRADES = ("A", "B", "C", "D", "E")
DEFAULT_LOAN_GRADE = "E"

# Grade thresholds on the loan-to-income ratio, applied in order.
GRADE_THRESHOLDS = (
    (0.15, "A"),
    (0.30, "B"),
    (0.40, "C"),
    (0.50, "D"),
)

# --- Input domains offered in the UI --------------------------------------
HOME_OWNERSHIP_OPTIONS = ("RENT", "MORTGAGE", "OWN", "OTHER")
LOAN_INTENT_OPTIONS = (
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
)
DEFAULT_ON_FILE_OPTIONS = ("N", "Y")
