# RiskPulse

**AI-powered credit risk & BNPL screening dashboard.**

RiskPulse scores a loan or buy-now-pay-later applicant with a trained
scikit-learn model, blends that estimate with affordability (loan-to-income)
and credit history, and turns the result into a transparent decision:
**approve**, **manual review** or **decline** — with the factors behind the
score and the recommended next steps.

---

## Highlights

| | |
|---|---|
| **Blended score** | 40% model default probability + 40% loan-to-income risk + 20% prior-default history |
| **Transparency** | Per-factor contribution chart — every point in the final score is attributable |
| **Automatic advice** | Decision policy plus actionable next steps and the dominant risk driver |
| **Policy loan grade** | Grade A–E derived from loan-to-income and default history (no manual grading) |
| **Model quality** | Hold-out accuracy **93.5%**, ROC AUC **0.929** (26,060 train / 6,516 test rows) |

## Screens

- **Metric cards** — blended risk score (sparkline), model default probability (clipped), model confidence, loan-to-income; all colored by risk level
- **Score contribution** — interactive ECharts bar showing the points each signal adds, plus a weights/risk table
- **Decision card** — colored status badge, score-band indicator and automatic next steps
- **Recent assessments** — session log of every assessment (customer, LTI, score, status)
- **Model benchmark** — accuracy, ROC AUC, applicant vs. portfolio benchmark
- **Portfolio position** & **Methodology** — LTI percentile versus the historical portfolio, weights and decision policy
- **Methodology** — blend weights and decision bands

## Project structure

```
RiskPulse/
├── app.py                  # Streamlit entry point — the dashboard UI only
├── train_model.py          # Trains the model, fits the scaler, writes metrics
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Brand theme (light + dark), dense KPI layout
├── models/                 # Generated artifacts (committed for convenience)
│   ├── risk_model.pkl      # RandomForestClassifier
│   ├── scaler.pkl          # StandardScaler fitted on the training split
│   └── metrics.json        # Hold-out evaluation report
├── data/
│   ├── credit_risk_dataset.csv   # Raw dataset
│   └── clean_credit_risk.csv     # Cleaned + one-hot encoded dataset
├── notebooks/
│   ├── inspect_data.py     # Quick profile of the raw data
│   └── clean_data.py       # Rebuilds data/clean_credit_risk.csv
└── src/
    ├── config.py           # Paths, weights, thresholds, option lists
    ├── risk.py             # Risk math, blending, decision policy, advice
    ├── features.py         # Input validation + model feature schema
    ├── artifacts.py        # Model / scaler / metrics / reference loaders
    └── data_prep.py        # Cleaning and train/test split helpers
```

**Design principle:** `app.py` is a presentation layer. All calculations live
in `src/`, so the policy can be unit-tested and reused by batch jobs or an API.

## How the score works

1. **Model probability** — the RandomForest estimates the chance of default;
   the value is bounded to 2%–98% to avoid absolute certainty.
2. **Loan-to-income risk** — `1 − e^(−LTI / 0.35)`, a smooth saturating curve so
   scores do not jump between hard buckets.
3. **Prior-default signal** — `1.0` if a default is on file, otherwise `0.0`.

```
score = 0.40 × model + 0.40 × LTI_risk + 0.20 × history
```

| Score band | Level | Decision |
|---|---|---|
| `< 15%` | Low | Approve |
| `15% – 35%` | Medium | Manual review |
| `> 35%` | High | Decline |

The blended score is a **screening indicator**, not a calibrated probability of
default, and it complements — never replaces — credit policy decisions.

## Getting started

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd RiskPulse

# 2. Create an isolated environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The app opens at <http://localhost:8501>. The trained artifacts in `models/`
are committed, so no training step is required to start.

## Retraining

```bash
python notebooks/clean_data.py   # only if data/clean_credit_risk.csv is missing
python train_model.py            # rewrites models/risk_model.pkl, scaler.pkl, metrics.json
```

`train_model.py` uses a stratified 80/20 split (`random_state=42`) and fits the
scaler on the training split only, keeping the hold-out evaluation leak-free.

## Tech stack

- **Streamlit** — dashboard, KPI cards, native ECharts charts, theming via TOML
- **scikit-learn** — `RandomForestClassifier`, `StandardScaler`, evaluation metrics
- **pandas / NumPy** — cleaning, one-hot encoding, aggregation
- **joblib** — artifact persistence

## Disclaimer

RiskPulse is an educational credit-risk demonstrator trained on a public
credit dataset. Do not use it for real lending decisions without calibration,
fairness testing and regulatory review.
