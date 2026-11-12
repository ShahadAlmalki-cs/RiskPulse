"""RiskPulse — credit risk & BNPL screening dashboard.

Run from the project root:

    streamlit run app.py

Layout
------
Top bar          "Dashboard" title + subtitle + model badge
Metric row       Model default probability · Model margin · LTI · Blended score
Analytics row    Risk score contribution (bar) | Decision card
Summary row      Applicant summary | LTI position
Bottom row       Recent assessments | Model performance

All risk math lives in ``src/risk.py``, feature preparation in
``src/features.py`` and artifact loading in ``src/artifacts.py``; this module is
purely presentation.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime

import pandas as pd
import streamlit as st

from src import config
from src.artifacts import (
    load_metrics,
    load_pipeline,
    load_reference_profiles,
    predict_default_probability,
)
from src.features import validate_inputs, with_policy_fields
from src.risk import (
    WEIGHTS,
    RiskAssessment,
    blend_signals,
    build_advice,
    calculate_model_confidence,
    correct_model_probability,
)

st.set_page_config(
    page_title="RiskPulse Dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
)

# Visual tokens per risk level, reused by every card, badge and alert.
LEVEL_STYLE = {
    "Low": {"color": "green", "icon": "check_circle", "alert": st.success},
    "Medium": {"color": "orange", "icon": "warning", "alert": st.warning},
    "High": {"color": "red", "icon": "cancel", "alert": st.error},
}

# ECharts bar colors matched to the brand palette.
SIGNAL_COLORS = ["#263045", "#9AB9E4", "#08142A"]
HISTORY_SESSION_KEY = "assessment_history"
ASSESSMENT_SESSION_KEY = "assessment"
INPUTS_SESSION_KEY = "assessment_inputs"

# Custom card styling: white surface, soft shadow, generous padding. Applied
# through st.markdown so the theme file stays the single source of color truth.
CARD_CSS = """
<style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 1px 3px rgba(11, 21, 40, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    /* tighter, more card-like metric tiles */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E3E6F5;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(11, 21, 40, 0.05);
    }
</style>
"""


# ---------------------------------------------------------------------------
# Cached artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return load_metrics()


@st.cache_data(show_spinner=False)
def get_reference_profiles() -> pd.DataFrame:
    return load_reference_profiles()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def assess(pipeline, values: dict[str, object], label: str = "Applicant") -> RiskAssessment:
    """Score one applicant and package every signal the dashboard needs."""
    raw_probability = predict_default_probability(pipeline, values)
    model_probability = correct_model_probability(raw_probability)
    prepared = with_policy_fields(values)
    loan_to_income = float(prepared["loan_percent_income"])
    return RiskAssessment(
        model_probability=model_probability,
        raw_model_probability=raw_probability,
        model_confidence=calculate_model_confidence(model_probability),
        loan_to_income=loan_to_income,
        loan_grade=str(prepared["loan_grade"]),
        loan_grade_reason=str(prepared["loan_grade_reason"]),
        contributions=blend_signals(
            model_probability,
            loan_to_income,
            str(values["cb_person_default_on_file"]),
        ),
    )


def record_assessment(
    label: str, values: dict[str, object], assessment: RiskAssessment
) -> None:
    """Append the assessment to the session history (newest first)."""
    history: list[dict[str, object]] = st.session_state.setdefault(
        HISTORY_SESSION_KEY, []
    )
    history.insert(
        0,
        {
            "Customer": label,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Loan amount": float(values["loan_amnt"]),
            "LTI": assessment.loan_to_income,
            "Score": assessment.score,
            "Status": assessment.decision,
            "Level": assessment.level,
        },
    )
    del history[8:]  # keep the last 8 assessments


def history_frame() -> pd.DataFrame:
    """Session history as a display-ready frame."""
    history = st.session_state.get(HISTORY_SESSION_KEY, [])
    return pd.DataFrame(history)


def inputs_fingerprint(values: dict[str, object]) -> str:
    """Stable fingerprint of the current applicant inputs.

    Used to detect when the sidebar has changed since the last assessment so a
    stale result can be flagged instead of silently displayed.
    """
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def assessment_is_stale(values: dict[str, object]) -> bool:
    """True when the sidebar inputs no longer match the stored assessment."""
    if ASSESSMENT_SESSION_KEY not in st.session_state:
        return False
    return st.session_state.get(INPUTS_SESSION_KEY) != inputs_fingerprint(values)


# Digit translation: any Unicode digit variant -> ASCII (English) digits.
_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def _parse_number(raw_text: str, label: str) -> float:
    """Parse a free-text numeric field using English digits.

    Accepts "12,000", "12 000", "12000.5" and Arabic-Indic digit forms; the
    value is always normalised to a plain float so the model sees identical
    numbers no matter how the user's keyboard or OS locale renders them.
    Raises ``ValueError`` with a friendly message on unparseable input.
    """
    cleaned = raw_text.translate(_DIGIT_MAP).strip().replace(",", "").replace(" ", "")
    cleaned = cleaned.replace("٫", ".").replace("٫", ".")  # Arabic decimal marks
    try:
        number = float(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a number using English digits (e.g. 60000)."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[dict[str, object], str]:
    """Collect applicant inputs in a grouped, compact sidebar."""
    with st.sidebar:
        st.markdown(
            "### :material/monitoring: RiskPulse\n"
            ":small[Pre-approval risk assessment — before pricing]"
        )
        st.divider()

        st.markdown("**Applicant profile**")
        customer_name = st.text_input(
            "Customer name", value="Applicant 1", placeholder="e.g. Sara Ali"
        )
        # Numeric fields are plain text inputs so the browser always shows
        # Latin (English) digits, regardless of the operating-system locale.
        person_age_text = st.text_input(
            "Age", value="30", help="Years (18–100)."
        )
        person_income_text = st.text_input(
            "Annual income",
            value="60000",
            help="Gross annual income in USD; drives the loan-to-income ratio.",
        )
        person_emp_length_text = st.text_input(
            "Employment length (years)", value="5.0", help="Years (0–60)."
        )
        cb_person_cred_hist_text = st.text_input(
            "Credit history length (years)", value="5", help="Years (0–80)."
        )
        person_home_ownership = st.selectbox(
            "Home ownership", config.HOME_OWNERSHIP_OPTIONS
        )
        cb_person_default_on_file = st.selectbox(
            "Previous default on record",
            config.DEFAULT_ON_FILE_OPTIONS,
            format_func=lambda value: (
                "Yes" if value == config.PRIOR_DEFAULT_FLAG else "No"
            ),
        )

        st.markdown("**Loan request**")
        loan_amnt_text = st.text_input(
            "Loan amount", value="10000", help="USD (500–1,000,000)."
        )
        loan_intent = st.selectbox("Loan purpose", config.LOAN_INTENT_OPTIONS)

        values: dict[str, object] = {
            "person_age": _parse_number(person_age_text, "Age"),
            "person_income": _parse_number(person_income_text, "Annual income"),
            "person_emp_length": _parse_number(person_emp_length_text, "Employment length"),
            "person_home_ownership": person_home_ownership,
            "cb_person_cred_hist_length": _parse_number(cb_person_cred_hist_text, "Credit history length"),
            "cb_person_default_on_file": cb_person_default_on_file,
            "loan_amnt": _parse_number(loan_amnt_text, "Loan amount"),
            "loan_intent": loan_intent,
        }

        prepared = with_policy_fields(values)
        st.divider()
        with st.container(horizontal=True):
            st.markdown("**Policy grade**")
            st.badge(
                f"Grade {prepared['loan_grade']}",
                icon=":material/verified:",
                color="blue",
            )
        st.caption(
            "Internal heuristic from this app's policy logic — not a "
            "lender-assigned credit grade."
        )
        st.caption(
            f"LTI {float(prepared['loan_percent_income']):.1%} · derived "
            "automatically from the inputs."
        )

        returned_label = (customer_name or "Applicant").strip()
        return values, returned_label


# ---------------------------------------------------------------------------
# Top metric cards
# ---------------------------------------------------------------------------
def render_metric_cards(assessment: RiskAssessment) -> None:
    """Four top cards. Order matches the reference: model first, policy last."""
    style = LEVEL_STYLE[assessment.level]

    cards = [
        (
            ":material/monitoring: Model Probability",
            f"{assessment.model_probability:.1%}",
            f"≈ {assessment.raw_model_probability:.1%} raw",
            "Model default probability, clipped to the 2%–98% guardrail. "
            "Not a calibrated probability — see Model performance.",
        ),
        (
            ":material/shield: Model Margin",
            f"{assessment.model_confidence:.0%}",
            "|2p − 1|",
            "Gap between the model's default and non-default probabilities. "
            "A restatement of the probability, not an independent confidence measure.",
        ),
        (
            ":material/balance: LTI",
            f"{assessment.loan_to_income:.1%}",
            "Loan / Income",
            "Loan amount divided by annual income. Drives both the model and "
            "the policy overlay.",
        ),
        (
            f":{style['color']}[:material/diamond: Policy Score]",
            f"{assessment.score:.1%}",
            f"{assessment.level} risk",
            "Blended policy score — an illustrative, manually weighted "
            "overlay: 40% model + 40% LTI + 20% prior default. It "
            "double-counts affordability and credit history, which the "
            "model already encodes. Not a model output and not "
            "statistically optimised.",
        ),
    ]
    columns = st.columns(4, gap="small")
    for column, (label, value, sub, tip) in zip(columns, cards):
        with column:
            st.metric(label, value, delta=sub, delta_color="off", border=True, help=tip)


# ---------------------------------------------------------------------------
# Analytics: contribution chart + decision
# ---------------------------------------------------------------------------
def render_contribution_chart(assessment: RiskAssessment) -> None:
    """Interactive ECharts bar: points each signal adds to the final score."""
    rows = [
        {
            "Signal": item.label,
            "Points": round(item.points * 100, 4),
            "Label": f"{item.points:.1%}",
        }
        for item in assessment.contributions
    ]
    frame = pd.DataFrame(rows).sort_values("Points", ascending=True)

    with st.container(border=True, height=420):
        col_header, col_spacer = st.columns([3, 1])
        with col_header:
            st.subheader("Risk score breakdown")
        with col_spacer:
            st.caption("Policy score")
        st.caption(
            "Points each factor adds to the blended policy score (weight × "
            "signal risk). The bars sum to the final score."
        )
        st.echarts_chart(
            {
                "dataset": {"source": frame},
                "grid": {"left": 8, "right": 24, "top": 8, "bottom": 8, "containLabel": True},
                # Axis counts percentage points; the heaviest weight is 40%.
                "xAxis": {"type": "value", "max": 40, "axisLabel": {"formatter": "{value}%"}},
                "yAxis": {"type": "category"},
                "tooltip": {"trigger": "axis"},
                "color": SIGNAL_COLORS,
                "series": [
                    {
                        "type": "bar",
                        "name": "Points added",
                        "encode": {"x": "Points", "y": "Signal"},
                        "itemStyle": {"borderRadius": [0, 4, 4, 0]},
                        "label": {"show": True, "position": "right", "formatter": "{@Label}"},
                    }
                ],
            },
            height=290,
            alt=(
                "Points added to the blended score by factor: "
                + ", ".join(
                    f"{item.label} {item.points:.1%}" for item in assessment.contributions
                )
            ),
        )

    with st.container(border=True):
        st.markdown("**Weights & signal risk**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Signal": item.label,
                        "Weight": item.weight,
                        "Signal risk": item.value,
                        "Points": item.points,
                    }
                    for item in assessment.contributions
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Weight": st.column_config.NumberColumn(format="percent"),
                "Signal risk": st.column_config.NumberColumn(format="percent"),
                "Points": st.column_config.ProgressColumn(
                    "Points added", format="percent", min_value=0.0, max_value=1.0
                ),
            },
            alt="Table of factor weights, signal risk and points added",
        )


def render_decision_card(assessment: RiskAssessment) -> None:
    """Prominent decision state with the threshold pill and next steps."""
    style = LEVEL_STYLE[assessment.level]

    with st.container(border=True):
        st.subheader("Decision")
        st.markdown(
            f":{style['color']}[{style['icon'].replace('_', ' ').title()}] "
            f"**{assessment.decision}** · {assessment.level} risk"
        )

        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Risk score**")
            st.badge(
                f"< {config.LOW_RISK_MAX:.0%}"
                if assessment.level == "Low"
                else (
                    f"{config.LOW_RISK_MAX:.0%} – {config.MEDIUM_RISK_MAX:.0%}"
                    if assessment.level == "Medium"
                    else f"> {config.MEDIUM_RISK_MAX:.0%}"
                ),
                color=style["color"],
            )
        st.metric(
            "Policy score",
            f"{assessment.score:.1%}",
            border=False,
        )

        messages = {
            "Low": (
                "Within policy: the blended score is below "
                f"{config.LOW_RISK_MAX:.0%}, so no manual review is required."
            ),
            "Medium": (
                f"Manual review: the score sits in the {config.LOW_RISK_MAX:.0%}–"
                f"{config.MEDIUM_RISK_MAX:.0%} band. A credit analyst must review "
                "the file before approval."
            ),
            "High": (
                f"Outside policy: the score exceeds {config.MEDIUM_RISK_MAX:.0%}. "
                "The request should not be approved as submitted."
            ),
        }
        style["alert"](
            messages[assessment.level],
            title=f"{assessment.decision} · {assessment.level} risk",
            icon=f":material/{style['icon']}:",
        )

        st.markdown("**Recommended next steps**")
        for step in build_advice(assessment):
            st.markdown(f"- {step}")


def render_applicant_summary(
    label: str, values: dict[str, object], assessment: RiskAssessment
) -> None:
    """Compact key-value strip of the submitted application."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{label}**")
            st.badge(assessment.decision, color=LEVEL_STYLE[assessment.level]["color"])
        summary = {
            "Annual income": f"${float(values['person_income']):,.0f}",
            "Loan amount": f"${float(values['loan_amnt']):,.0f}",
            "Employment": f"{float(values['person_emp_length']):g} yrs",
            "Credit history": f"{int(values['cb_person_cred_hist_length'])} yrs",
            "Home": str(values["person_home_ownership"]),
            "Purpose": str(values["loan_intent"]).replace(
                "DEBTCONSOLIDATION", "DEBT CONSOLIDATION"
            ),
        }
        columns = st.columns(4)
        items = list(summary.items())
        for index, (field, value) in enumerate(items):
            with columns[index % 4]:
                st.caption(field)
                st.markdown(f"**{value}**")


# ---------------------------------------------------------------------------
# Recent assessments
# ---------------------------------------------------------------------------
def render_history() -> None:
    """Session log of assessments, styled like a transaction history table."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Recent assessments")
            if st.button(
                "Clear",
                icon=":material/delete_sweep:",
                type="tertiary",
                disabled=not st.session_state.get(HISTORY_SESSION_KEY),
            ):
                st.session_state[HISTORY_SESSION_KEY] = []
                st.session_state.pop(ASSESSMENT_SESSION_KEY, None)
                st.session_state.pop(INPUTS_SESSION_KEY, None)
                st.rerun()

        frame = history_frame()
        if frame.empty:
            st.info(
                "No assessments yet. Fill in the sidebar and choose "
                "**Run assessment**.",
                icon=":material/history:",
            )
            return

        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Customer": st.column_config.TextColumn("Customer", width="medium"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Loan amount": st.column_config.NumberColumn(
                    "Loan amount", format="$%,.0f"
                ),
                "LTI": st.column_config.NumberColumn("LTI", format="percent"),
                "Score": st.column_config.ProgressColumn(
                    "Blended score", format="percent", min_value=0.0, max_value=1.0
                ),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Level": None,
            },
            alt="Assessments run in this session",
        )


# ---------------------------------------------------------------------------
# Model benchmark
# ---------------------------------------------------------------------------
def render_benchmark(metrics: dict) -> None:
    """Model quality as a compact stat strip: accuracy, PR-AUC, Brier, recall."""
    with st.container(border=True):
        header_column, info_column = st.columns([3, 1])
        with header_column:
            st.subheader("Model performance")
        with info_column:
            st.caption("Hold-out test set")
        if not metrics:
            st.info("Run `python train_model.py` to generate `models/metrics.json`.")
            return

        # Four inline stats; the row is wide enough not to clip at 1280px.
        stat_columns = st.columns(4, gap="small")
        stats = [
            ("Accuracy", f"{metrics['accuracy']:.1%}",
             "Share of correct classifications. Not evidence of quality on "
             "its own: 78% of applicants repay, so a majority-guesser "
             "already reaches that."),
            ("PR-AUC", f"{metrics['pr_auc']:.3f}",
             "Ranking quality under class imbalance — the headline metric "
             "for this 21.8% default base rate."),
            ("Brier", f"{metrics['brier']:.3f}",
             "Probability calibration. Consistent with imperfect calibration "
             "(RandomForest vote shares, not calibrated probabilities)."),
            ("Default Recall", f"{metrics['default_recall']:.1%}",
             "Share of actual defaults the model flags. The business-critical "
             "metric for a risk screener."),
        ]
        for column, (label, value, tip) in zip(stat_columns, stats):
            with column:
                st.metric(label, value, border=True, help=tip)
        st.caption(
            f"ROC AUC {metrics['roc_auc']:.3f} · precision "
            f"{metrics['precision']:.1%} · F1 {metrics['f1']:.1%} · "
            f"trained on {metrics['n_train']:,} applicants"
        )


def render_portfolio(reference: pd.DataFrame, assessment: RiskAssessment) -> None:
    """Where the applicant's LTI sits inside the historical portfolio."""
    with st.container(border=True, height=420):
        header_column, spacer_column = st.columns([3, 1])
        with header_column:
            st.subheader("LTI position")
        with spacer_column:
            st.caption("Portfolio")
        if reference.empty or "loan_percent_income" not in reference:
            st.info(
                "Reference profiles unavailable. Run "
                "`python notebooks/clean_data.py` to rebuild "
                "`data/reference_profiles.csv`."
            )
            return

        portfolio_lti = reference["loan_percent_income"].dropna()
        percentile = (portfolio_lti <= assessment.loan_to_income).mean()
        st.caption(
            f"{portfolio_lti.size:,} historical applicants · median LTI "
            f"{portfolio_lti.median():.1%}"
        )
        with st.container(horizontal=True):
            st.metric("Your LTI", f"{assessment.loan_to_income:.1%}")
            st.metric("Portfolio median", f"{portfolio_lti.median():.1%}")
            st.metric("Percentile", f"{percentile:.0%}")

        buckets = pd.cut(
            portfolio_lti,
            bins=[-0.01, 0.15, 0.30, 0.40, 0.50, float("inf")],
            labels=["≤15%", "15–30%", "30–40%", "40–50%", ">50%"],
        )
        shares = (
            buckets.value_counts(normalize=True, sort=False)
            .rename_axis("LTI band")
            .reset_index(name="Share of applicants")
        )
        st.bar_chart(
            shares,
            x="LTI band",
            y="Share of applicants",
            height=190,
            alt="Share of historical applicants in each loan-to-income band.",
        )


def render_methodology(metrics: dict) -> None:
    """Explain weights, thresholds and the model behind the score."""
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Blend weights")
            st.dataframe(
                pd.DataFrame({"Signal": list(WEIGHTS), "Weight": list(WEIGHTS.values())}),
                hide_index=True,
                width="stretch",
                column_config={
                    "Weight": st.column_config.ProgressColumn(
                        "Weight", format="percent", min_value=0.0, max_value=1.0
                    )
                },
                alt="Table of blend weights per risk signal",
            )
            st.caption(
                "Model probability is bounded to 2%–98%; LTI risk follows a "
                "smooth saturating curve, so scores never jump between buckets."
            )

    with col2:
        with st.container(border=True):
            st.subheader("Decision policy")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Score band": [
                            f"below {config.LOW_RISK_MAX:.0%}",
                            f"{config.LOW_RISK_MAX:.0%} – {config.MEDIUM_RISK_MAX:.0%}",
                            f"above {config.MEDIUM_RISK_MAX:.0%}",
                        ],
                        "Decision": ["Approve", "Manual review", "Decline"],
                    }
                ),
                hide_index=True,
                width="stretch",
                alt="Table of score bands and decisions",
            )
            st.caption(
                "The blended score is a screening indicator, not a calibrated "
                "probability of default."
            )

    if metrics:
        feature_count = len(metrics.get("feature_names", []))
        st.caption(
            f"Model: RandomForestClassifier · {feature_count} raw features · "
            f"trained on {metrics['n_train']:,} applicants."
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
try:
    pipeline = get_pipeline()
except FileNotFoundError:
    st.error(
        "Run `python train_model.py` first to create `models/risk_pipeline.pkl`.",
        title="Model not found",
        icon=":material/error:",
    )
    st.stop()

metrics = get_metrics()
values, customer_label = render_sidebar()

try:
    validate_inputs(values)
except ValueError as exc:
    st.error(str(exc), title="Check the inputs", icon=":material/error:")
    st.stop()

# Header bar ---------------------------------------------------------------
st.markdown(CARD_CSS, unsafe_allow_html=True)
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    header_column, badge_column = st.columns([4, 1], gap="medium")
    with header_column:
        st.markdown("# Dashboard")
        st.caption("Pre-approval risk assessment — before pricing.")
    with badge_column:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("Model live", icon=":material/bolt:", color="green")

if st.button(
    "Run assessment",
    type="primary",
    icon=":material/play_arrow:",
    width="stretch",
):
    current = assess(pipeline, values, customer_label)
    st.session_state[ASSESSMENT_SESSION_KEY] = current
    st.session_state[INPUTS_SESSION_KEY] = inputs_fingerprint(values)
    record_assessment(customer_label, values, current)

assessment: RiskAssessment | None = st.session_state.get(ASSESSMENT_SESSION_KEY)
stale = assessment_is_stale(values)

if assessment is None:
    st.info(
        "Fill in the applicant profile and loan request in the sidebar, then "
        "choose **Run assessment** to populate the dashboard.",
        title="No assessment yet",
        icon=":material/info:",
    )
    st.subheader("Recent assessments")
    render_history()
    st.stop()

if stale:
    st.warning(
        "Inputs have changed since this assessment was run. The figures below "
        "describe the **previous** inputs — press **Run assessment** to refresh.",
        title="Outdated assessment",
        icon=":material/history:",
    )

# Metric cards -------------------------------------------------------------
render_metric_cards(assessment)
st.space("small")

# Applicant strip ----------------------------------------------------------
render_applicant_summary(customer_label, values, assessment)
st.space("small")

# Analytics row ------------------------------------------------------------
chart_column, decision_column = st.columns([3, 2], gap="medium")
with chart_column:
    render_contribution_chart(assessment)
with decision_column:
    render_decision_card(assessment)

st.space("small")

# Summary row: applicant strip next to the portfolio LTI position -----------
summary_column, lti_column = st.columns([3, 2], gap="medium")
with summary_column:
    render_applicant_summary(customer_label, values, assessment)
with lti_column:
    render_portfolio(get_reference_profiles(), assessment)

st.space("small")

# Model performance — full width so the four stat cards never clip ----------
render_benchmark(metrics)

st.space("small")

# Recent assessments — full width so seven columns fit without clipping ------
render_history()

with st.container(horizontal=True):
    with st.popover(":material/menu_book: Methodology"):
        render_methodology(metrics)
    with st.popover(":material/receipt_long: Assessment details"):
        st.markdown(
            f"**Raw model probability:** {assessment.raw_model_probability:.2%}  \n"
            f"**Clipped probability:** {assessment.model_probability:.2%}  \n"
            f"**Model margin (|2p-1|):** {assessment.model_confidence:.1%}  \n"
            f"**Policy grade:** {assessment.loan_grade}"
        )
        for item in assessment.contributions:
            st.markdown(
                f"- **{item.label}** — weight {item.weight:.0%} × risk "
                f"{item.value:.1%} = **{item.points:.1%} points**"
            )
        if not metrics:
            st.caption("No `models/metrics.json` found; run `python train_model.py`.")

st.divider()
st.caption(
    "RiskPulse supports — it does not replace — credit policy decisions. "
    "Always pair automated screening with human review."
)
