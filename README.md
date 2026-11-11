# RiskPulse

**Pre-pricing, pre-approval credit risk screening.**

RiskPulse scores a loan or buy-now-pay-later application using only the
information available **before a rate is quoted and a decision is made**, and
turns the estimate into a transparent recommendation: **approve**,
**manual review** or **decline** — with the factors behind the score and the
recommended next steps.

This is an educational demonstrator built on a public credit dataset. It is not
trained on, and does not represent, any real lender's customers.

---

## Scope: what the model does and does not use

The model is deliberately restricted to **application-time features**.

| Used | Excluded (and why) |
|---|---|
| Age, income, employment length, home ownership, credit history length, prior default on file, loan amount, loan purpose | `loan_grade` — a post-decision pricing label assigned by the lender; 98% predictable from the interest rate |
| `loan_percent_income` (= `loan_amnt / person_income`, recomputed consistently everywhere) | `loan_int_rate` — set by the lender at pricing, i.e. **after** the moment in the process this product models |
| | `loan_status` — the target; never a predictor |

**Product definition:** a pre-pricing, pre-approval assessment. If you need a
post-pricing model, the interest rate is a strong predictor and a different
product scope.

## Model evaluation (hold-out test set, 6,483 applicants)

| Metric | Value |
|---|---|
| Accuracy | 88.4% |
| Precision | 87.5% |
| Recall (default class) | 54.6% |
| F1 | 66.6% |
| ROC AUC | 0.869 |
| PR AUC | 0.774 |
| Brier score | 0.092 |
| Default recall | 54.6% |
| Confusion matrix (TN/FP/FN/TP) | 4,968 / 124 / 620 / 771 |
| Base rate | 21.8% of applicants default |

**Read this correctly:** 78% of applicants repay, so accuracy alone is weak
evidence — a majority-guesser already reaches 78%. The metrics that matter for
a risk screener are **default recall (how many defaults are caught)**,
**PR-AUC (ranking quality under class imbalance)** and the **Brier score
(probability calibration)**. Precision/recall trade-off is set at the default
0.5 threshold and was not tuned.

Training and inference share one fitted sklearn `Pipeline` (median imputation +
one-hot encoding + RandomForest), so the preprocessing applied to an applicant
in the app is exactly the preprocessing the model was trained on. Deduplication
and the train/test split both happen before any preprocessing is fitted;
imputers and the encoder see training data only.

## The three layers of the decision

RiskPulse produces **three distinct things**, which should not be confused:

1. **Model probability of default** — the RandomForest's calibrated-ish
   estimate, clipped to the 2%–98% range. This is the only statistically
   evaluated quantity (the metrics table above).
2. **Blended risk score** — a *policy overlay*: `0.40 × model + 0.40 × LTI risk
   + 0.20 × prior default`. It intentionally re-weights affordability and
   credit history, even though both also enter the model, to make policy
   emphasis explicit and auditable. **It is not a model output, and none of the
   evaluation metrics apply to it.**
3. **Recommendation** — a deterministic mapping from the blended score to
   Approve / Manual review / Decline.

## Decision policy (a policy choice, not an optimised threshold)

| Blended score | Level | Recommendation |
|---|---|---|
| `< 15%` | Low | Approve |
| `15% – 35%` | Medium | Manual review |
| `> 35%` | High | Decline |

The 40/40/20 weights and the 15%/35% bands are **documented policy choices, not
statistically optimised thresholds**. They were not tuned against any loss
function. Changing them changes the product's behaviour; treating them as
empirically derived would be incorrect.

## Project structure

```
RiskPulse/
├── app.py                  # Streamlit entry point — the dashboard UI only
├── train_model.py          # Builds the pipeline, trains, evaluates, persists artifacts
├── requirements.txt
├── requirements-dev.txt    # pytest (development only)
├── tests/
│   └── test_model_validity.py   # 21 tests: schema, leakage regression, validation
├── .streamlit/
│   └── config.toml         # Brand theme (light + dark), dense KPI layout
├── models/                 # Generated artifacts (committed for convenience)
│   ├── risk_pipeline.pkl   # Pipeline(imputer + encoder + RandomForest)
│   ├── feature_schema.json # Feature names, order, categories, missing policy
│   ├── metrics.json        # Hold-out evaluation report
│   └── backup_pre_setc/    # Previous (post-pricing) model, kept for reference
├── data/
│   ├── credit_risk_dataset.csv   # Raw dataset
│   └── reference_profiles.csv    # Deduplicated historical portfolio for the app
├── notebooks/
│   ├── inspect_data.py     # Quick profile of the raw data
│   └── clean_data.py       # Rebuilds data/reference_profiles.csv
└── src/
    ├── config.py           # Paths, feature schema, weights, thresholds
    ├── risk.py             # Risk math, blending, decision policy, advice
    ├── features.py         # Input validation + model frame assembly
    ├── artifacts.py        # Pipeline / metrics / reference loaders
    └── data_prep.py        # Row-level cleaning (no learned transforms)
```

**Design principle:** `app.py` is a presentation layer. All calculation lives
in `src/`, and all learned preprocessing lives inside the persisted pipeline,
so training and inference cannot drift apart.

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
pip install -r requirements-dev.txt   # optional, for the test suite

# 4. Run the dashboard
streamlit run app.py
```

The app opens at <http://localhost:8501>. Trained artifacts in `models/` are
committed, so no training step is needed to start.

## Retraining

```bash
python train_model.py          # rebuilds risk_pipeline.pkl, feature_schema.json, metrics.json
```

`train_model.py` deduplicates the raw data, recomputes loan-to-income, does a
stratified 80/20 split (`random_state=42`) **before** fitting any
preprocessing, then fits a single pipeline on the training split and evaluates
it on the held-out test set.

## Tests

```bash
python -m pytest tests -q
```

21 tests cover: schema/order compatibility, absence of post-pricing features,
target exclusion, LTI recomputation, duplicate removal, unknown-category
rejection, missing-value imputation, invalid numeric inputs, probability
validity, and the decision logic boundaries.

## Known limitations

- **Dataset** — a public lending dataset, not BNPL transactions and not
  representative of any specific market or lender.
- **Thresholds** — the decision bands are policy choices, not optimised.
- **Recall** — the model misses ~45% of defaults at the default threshold; a
  production system would tune this explicitly against a documented cost
  asymmetry.
- **UI** — known cosmetic clipping in the benchmark cards at narrow widths.
- **No persistence** — assessment history is session-only.
- **Not production-ready** — no authentication, no audit trail, no monitoring.
  Do not use for real lending decisions without calibration, fairness testing
  and regulatory review.

## Disclaimer

RiskPulse is an educational credit-risk demonstrator trained on a public
credit dataset. Do not use it for real lending decisions without calibration,
fairness testing and regulatory review.
