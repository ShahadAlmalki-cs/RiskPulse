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
CLEAN_DATASET_PATH = DATA_DIR / "clean_credit_risk.csv"
MODEL_PATH = MODELS_DIR / "risk_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- Model training --------------------------------------------------------
TARGET_COLUMN = "loan_status"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# --- Risk blending ---------------------------------------------------------
# The screening score is a weighted sum of three independent risk signals.
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
