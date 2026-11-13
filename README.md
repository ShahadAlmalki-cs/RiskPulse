# RiskPulse — Pre-Pricing Credit Risk Assessment Dashboard

RiskPulse is a credit risk screening dashboard designed to evaluate loan and Buy Now, Pay Later (BNPL) applications at pre-approval time—**before an interest rate is quoted or an approval decision is finalized**.

The application combines a supervised machine learning model (Random Forest) trained on application-time applicant features with an explicit, rule-based policy overlay to produce an auditable recommendation: **Approve**, **Manual Review**, or **Decline**.

> **Educational & Demonstration Notice:** RiskPulse is an educational demonstrator trained on a public personal-loan dataset. It is **not** trained on Tamara's internal data, actual commercial BNPL transaction records, or any private lender's customer portfolio. It is not intended for real-world automated credit decisions without calibration, fairness audits, and regulatory review.

---

## Key Capabilities

- **Application-Time Screening:** Restricted exclusively to features available when an applicant applies (age, income, employment duration, home ownership, credit history length, prior default, requested amount, and loan purpose). Post-decision variables such as lender-assigned loan grades and quoted interest rates are strictly excluded.
- **Dual-Signal Architecture:** Clear architectural separation between the machine learning model's default probability estimate and the rule-based policy score.
- **Auditable Policy Overlay:** A transparent, deterministic 40/40/20 weighted blend combining the model's clipped probability, a continuous loan-to-income (LTI) risk curve, and past default records.
- **Transparent Decisioning:** Deterministic mapping to Approve (< 15%), Manual Review (15%–35%), or Decline (> 35%), accompanied by risk driver identification and actionable next steps.
- **Empirical Model Evaluation:** Evaluated on a hold-out test set (6,483 applicants) across class-imbalance metrics: PR-AUC, default recall, ROC AUC, and Brier score.
- **Dual-Theme Dashboard:** High-contrast Streamlit interface supporting light and dark modes with WCAG AA-compliant contrast ratios.

---

## Technology Stack

| Layer | Technologies |
| --- | --- |
| **Language & Runtime** | Python 3.10+ (tested on Python 3.12) |
| **Machine Learning** | scikit-learn (RandomForestClassifier, Pipeline, ColumnTransformer) |
| **Data Processing** | pandas, NumPy |
| **Artifact Persistence** | joblib, JSON |
| **User Interface** | Streamlit, Apache ECharts (streamlit-echarts) |
| **Testing & QA** | pytest, Playwright / Patchright (headless browser QA) |

---

## Architecture & Project Structure

RiskPulse follows a strict separation of concerns: `app.py` acts solely as a presentation layer, while all risk logic, feature validation, and persistence live in `src/`. Learned transformations are encapsulated inside the persisted scikit-learn pipeline, ensuring training and inference preprocessing never diverge.

```
RiskPulse/
├── app.py                      # Streamlit presentation dashboard
├── train_model.py              # End-to-end training and evaluation pipeline
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development and testing dependencies
├── .gitignore                  # Git ignore rules for Python, logs, and artifacts
├── .streamlit/
│   └── config.toml             # Theme configuration (light & dark palettes)
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, feature schemas, blend weights, thresholds
│   ├── features.py             # Input validation and raw model frame assembly
│   ├── risk.py                 # Risk calculations, signal blending, and decision rules
│   ├── data_prep.py            # Row-level cleaning, deduplication, and LTI recomputation
│   └── artifacts.py            # Loaders for pipeline, schema, metrics, and profiles
├── models/
│   ├── risk_pipeline.pkl       # Fitted Pipeline (imputer + one-hot encoder + RandomForest)
│   ├── feature_schema.json     # Feature names, categories, and preprocessing contract
│   ├── metrics.json            # Verified hold-out test set evaluation report
│   └── backup_pre_setc/        # Reference model artifacts from earlier iterations
├── data/
│   ├── credit_risk_dataset.csv # Raw credit risk dataset
│   ├── reference_profiles.csv  # Deduplicated historical dataset for portfolio benchmarking
│   └── clean_credit_risk.csv   # Cleaned dataset copy
├── notebooks/
│   ├── inspect_data.py         # Summary profiling script for the raw dataset
│   └── clean_data.py           # Rebuilds reference profiles from raw data
└── tests/
    ├── test_model_validity.py  # Pipeline integrity, schema compatibility, boundary tests
    ├── test_stale_flow.py      # Input validation, UI state flow, and zero-income guards
    └── test_theme_contrast.py  # WCAG AA contrast regression suite for light and dark modes
```

---

## Dataset Description & Preprocessing

The model is trained on a public personal-loan credit risk dataset containing applicant demographics, credit bureau variables, and loan characteristics.

### Preprocessing & Data Hygiene (`src/data_prep.py`)

1. **Age Filtering:** Outlier and erroneous records outside the range of 18 to 100 years are excluded.
2. **Deduplication:** Duplicate applicant records are removed prior to splitting.
3. **LTI Recomputation:** Loan-to-income (`loan_percent_income = loan_amnt / person_income`) is recomputed directly from raw values to resolve data entry inconsistencies in the source data.
4. **Train / Test Split:** A stratified 80/20 split (`random_state=42`) is performed **before** fitting any imputers or encoders to prevent data leakage.
5. **In-Pipeline Imputation & Encoding:**
   - Missing numeric values: Median imputation fitted on training data only.
   - Missing categorical values: Mode (`most_frequent`) imputation fitted on training data only.
   - Categorical encoding: `OneHotEncoder` with strict known categories and `handle_unknown="error"`.

### Feature Schema

| Feature Name | Type | Description |
| --- | --- | --- |
| `person_age` | Numeric | Applicant age in years (18–100) |
| `person_income` | Numeric | Annual gross income in USD |
| `person_emp_length` | Numeric | Employment length in years |
| `person_home_ownership` | Categorical | Housing status: `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `loan_amnt` | Numeric | Requested loan amount in USD |
| `loan_intent` | Categorical | Purpose: `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_percent_income` | Numeric | Derived ratio: `loan_amnt / person_income` |
| `cb_person_default_on_file` | Categorical | Bureau record of historical default: `Y` / `N` |
| `cb_person_cred_hist_length` | Numeric | Length of credit history in years |

**Target Variable:** `loan_status` (Binary: `0` = non-default / repaid, `1` = default). Base default rate is 21.87%.

### Explicitly Excluded Features

- `loan_grade`: A post-decision credit grade assigned by the lender, which is 98% collinear with the quoted interest rate.
- `loan_int_rate`: The interest rate quoted by the lender at pricing, i.e., after pre-approval screening.
- `loan_status`: The ground-truth target variable.

---

## Model Evaluation & Performance

The classifier is a `RandomForestClassifier` (100 estimators, default parameters, `random_state=42`) evaluated on a held-out test split of 6,483 applicants.

### Hold-Out Test Metrics (`models/metrics.json`)

| Metric | Verified Value | Interpretation |
| --- | --- | --- |
| **Accuracy** | **88.43%** | Overall correct rate. In an unbalanced dataset (78.13% non-default), accuracy alone is misleading; a naive majority-class classifier reaches 78.13%. |
| **Precision** | **87.95%** | When the model predicts default, it is correct ~88% of the time. |
| **Recall (Default Class)** | **54.58%** | Proportion of actual defaulting borrowers captured at the standard 0.5 decision threshold. |
| **F1 Score** | **67.36%** | Harmonic mean of precision and recall for the default class. |
| **ROC AUC** | **0.8687** | Discrimination ability across all classification thresholds. |
| **PR-AUC (Average Precision)** | **0.7738** | Primary ranking quality metric under class imbalance (baseline = 0.2187). |
| **Brier Score** | **0.0918** | Mean squared probability error (lower is better). |
| **Test Set Size** | **6,483** | 5,065 non-defaults (`0`) and 1,418 defaults (`1`). |

### Confusion Matrix (Test Split)

- **TN / FP / FN / TP:** `4,959 / 106 / 644 / 774`

| | Predicted Non-Default | Predicted Default |
| --- | --- | --- |
| **Actual Non-Default (`0`)** | 4,959 (TN) | 106 (FP) |
| **Actual Default (`1`)** | 644 (FN) | 774 (TP) |

### Note on Probability Calibration

Random Forest probability outputs (`predict_proba`) represent tree ensemble vote shares and are **not calibrated** probabilities. The model outputs are bounded to `[0.02, 0.98]` as a guardrail. No post-hoc calibration technique (such as Platt scaling or isotonic regression) has been applied in this release.

---

## Decision Architecture: Model Probability vs. Policy Risk Score

RiskPulse distinguishes between statistical estimates and credit policy decisions:

```
[ Applicant Inputs ]
         │
         ├──► [ scikit-learn Pipeline ] ────► Model Default Probability (2% – 98%) [Weight: 40%]
         │
         ├──► [ LTI Calculation ]       ────► LTI Risk Curve (0% – 100%)         [Weight: 40%] ──► Blended Score ──► Decision Rule
         │
         └──► [ Bureau Default Check ]  ────► History Flag (0% or 100%)          [Weight: 20%]
```

1. **Model Default Probability:** The raw tree vote share from the Random Forest model, clipped to `[0.02, 0.98]`. This is the sole statistically trained and evaluated metric.
2. **Blended Risk Score (Policy Overlay):** An illustrative, rule-based screening score combining three factors:
   $$\text{Blended Score} = 0.40 \times P(\text{Default}) + 0.40 \times \text{Risk}_{\text{LTI}} + 0.20 \times \text{Flag}_{\text{Default}}$$
   - $\text{Risk}_{\text{LTI}} = 1 - \exp\left(-\frac{\text{LTI}}{0.35}\right)$: A continuous saturating function that penalizes excessive loan-to-income ratios.
   - $\text{Flag}_{\text{Default}} \in \{0.0, 1.0\}$: Binary indicator of past credit default on file.
   - *Design note:* LTI and prior default are already features within the Random Forest model. The blended score intentionally re-weights them to enforce underwriting policy priorities (affordability and repayment history). It is an illustrative policy overlay, not an independent statistical model.
3. **Recommendation:** A deterministic mapping from the blended score to an underwriting action:

| Blended Risk Score | Risk Level | Policy Decision | Operational Action |
| --- | --- | --- | --- |
| `< 15.0%` | **Low** | **Approve** | Eligible for automated standard approval. |
| `15.0% – 35.0%` | **Medium** | **Manual Review** | Route to credit analyst; verify income and bank statements. |
| `> 35.0%` | **High** | **Decline** | Request outside policy; suggest smaller amount, shorter tenor, or guarantor. |

---

## Installation & Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (developed and verified on Python 3.12)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/RiskPulse.git
cd RiskPulse
```

### 2. Set Up a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install development and test dependencies
pip install -r requirements-dev.txt
```

---

## Running the Application

Pre-trained model artifacts are included in `models/`, allowing immediate execution without a retraining step:

```bash
streamlit run app.py
```

Once started, navigate to `http://localhost:8501` in your browser.

### Re-training the Model (Optional)

To retrain the Random Forest pipeline and regenerate `models/risk_pipeline.pkl`, `models/feature_schema.json`, and `models/metrics.json`:

```bash
python train_model.py
```

---

## Running the Test Suite

The test suite consists of 47 automated tests covering input validation, pipeline schema adherence, risk blend calculations, stale session detection, and UI theme contrast:

```bash
python -m pytest tests -q
```

### Test Scope

- **`tests/test_model_validity.py` (22 tests):** Verifies feature order, exclusion of post-pricing features (`loan_grade`, `loan_int_rate`), LTI calculation, unknown-category handling, missing-value imputation, decision thresholds, and advice generation.
- **`tests/test_stale_flow.py` (21 tests):** Verifies input validation (zero/negative income rejection, age bounds), fingerprint change detection, and UI session state management.
- **`tests/test_theme_contrast.py` (4 tests):** Validates WCAG AA contrast compliance across all foreground/background color combinations in both light and dark mode palettes.

---

## Known Limitations

- **Dataset Origin:** The model is trained on an open-source personal loan dataset. It does not reflect BNPL micro-instalment behaviors, specific merchant risk categories, or regional economic conditions.
- **Uncalibrated Probabilities:** Output probabilities reflect tree ensemble vote distributions and have not undergone isotonic calibration.
- **Policy Weights:** The 40/40/20 weights and 15%/35% decision bands represent illustrative underwriting heuristics rather than mathematically loss-optimized business boundaries.
- **Session-Only Storage:** Assessment history is stored in Streamlit session memory and resets when the browser session ends.

---

## Disclaimer

RiskPulse is an academic and portfolio demonstrator developed for educational purposes in machine learning and credit risk engineering. It is not licensed lending software, does not constitute financial advice, and should never be used as the sole decision mechanism in commercial lending operations without extensive domain adaptation, bias/fairness auditing, and regulatory compliance validation.
