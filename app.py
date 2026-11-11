"""RiskPulse — credit risk & BNPL screening dashboard.

Run from the project root:

    streamlit run app.py

Layout
------
Top bar          "RiskPulse Dashboard" + policy indicator
Metric row       Blended risk score · Model default probability · Model
                 confidence · Loan-to-income (colored by risk level)
Analytics row    Score contribution by factor | Decision card
History row      Recent assessments (this session) | Model benchmark

All risk math lives in ``src/risk.py``, feature preparation in
``src/features.py`` and artifact loading in ``src/artifacts.py``; this module is
purely presentation.
"""

from __future__ import annotations

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
SIGNAL_COLORS = ["#0E7C86", "#1D4ED8", "#7C3AED"]
HISTORY_SESSION_KEY = "assessment_history"


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[dict[str, object], str]:
    """Collect applicant inputs in a grouped, compact sidebar."""
    with st.sidebar:
        st.markdown(
            f"### :material/monitoring: RiskPulse\n"
            f":small[Credit risk & BNPL screening]"
        )
        st.divider()

        st.markdown("**Applicant profile**")
        customer_name = st.text_input(
            "Customer name", value="Applicant 1", placeholder="e.g. Sara Ali"
        )
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input(
            "Annual income",
            min_value=1_000,
            max_value=10_000_000,
            value=60_000,
            step=1_000,
            help="Gross annual income; drives the loan-to-income ratio.",
        )
        person_emp_length = st.number_input(
            "Employment length (years)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5,
        )
        cb_person_cred_hist_length = st.number_input(
            "Credit history length (years)", min_value=0, max_value=80, value=5
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
        loan_amnt = st.number_input(
            "Loan amount", min_value=500, max_value=1_000_000, value=10_000, step=500
        )
        loan_intent = st.selectbox("Loan purpose", config.LOAN_INTENT_OPTIONS)

        values: dict[str, object] = {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "person_home_ownership": person_home_ownership,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "cb_person_default_on_file": cb_person_default_on_file,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
        }

        prepared = with_policy_fields(values)
        st.divider()
        with st.container(horizontal=True):
            st.markdown("**Loan grade**")
            st.badge(
                f"Grade {prepared['loan_grade']}",
                icon=":material/verified:",
                color="blue",
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
def _sparkline(score: float) -> list[float]:
    """Tiny trend shape for the score card."""
    return [0.0, score * 0.35, score * 0.7, score * 0.95, score]


def render_metric_cards(assessment: RiskAssessment) -> None:
    """Four top cards, each colored by the applicant's risk level."""
    style = LEVEL_STYLE[assessment.level]
    color = style["color"]

    with st.container(horizontal=True):
        st.metric(
            f":{color}[:material/shield: Blended risk score]",
            f"{assessment.score:.1%}",
            delta=f"{assessment.level} risk",
            delta_color="off",
            border=True,
            chart_data=_sparkline(assessment.score),
            chart_type="area",
            help="Weighted blend of the model, LTI and history signals.",
        )
        st.metric(
            "Model default probability (clipped)",
            f"{assessment.model_probability:.1%}",
            delta=f"raw {assessment.raw_model_probability:.1%}",
            delta_color="off",
            border=True,
            help="Model output bounded to the 2%–98% guardrail.",
        )
        st.metric(
            "Model margin",
            f"{assessment.model_confidence:.0%}",
            border=True,
            help="Gap between the model's default and non-default probabilities.",
        )
        st.metric(
            "Loan-to-income (LTI)",
            f"{assessment.loan_to_income:.1%}",
            delta=f"Grade {assessment.loan_grade}",
            delta_color="off",
            border=True,
            help="Loan amount divided by annual income.",
        )


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

    with st.container(border=True):
        st.subheader("Score contribution by factor")
        st.caption(
            "Points each factor adds to the blended score (weight × signal "
            "risk). The bars sum to the final score."
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
            height=260,
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
    """Visual decision state and the automatic next steps."""
    style = LEVEL_STYLE[assessment.level]

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Decision")
            st.badge(
                assessment.decision,
                icon=f":material/{style['icon']}:",
                color=style["color"],
            )
        st.metric(
            f":{style['color']}[Risk level]",
            f"{assessment.level} · {assessment.score:.1%}",
            border=False,
        )

        bands = [
            (f"< {config.LOW_RISK_MAX:.0%}", "Low", "green"),
            (
                f"{config.LOW_RISK_MAX:.0%} – {config.MEDIUM_RISK_MAX:.0%}",
                "Medium",
                "orange",
            ),
            (f"> {config.MEDIUM_RISK_MAX:.0%}", "High", "red"),
        ]
        with st.container(horizontal=True):
            for band, level, band_color in bands:
                active = level == assessment.level
                st.badge(
                    f"{band} · {level}",
                    color=band_color if active else "gray",
                    icon=f":material/{'radio_button_checked' if active else 'radio_button_unchecked'}:",
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
                st.session_state.pop("assessment", None)
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
def render_benchmark(assessment: RiskAssessment, metrics: dict) -> None:
    """Model quality plus the applicant-versus-portfolio comparison."""
    with st.container(border=True):
        st.subheader("Model benchmark")
        if not metrics:
            st.info("Run `python train_model.py` to generate `models/metrics.json`.")
            return

        # Equal columns keep the three benchmark cards on a single row.
        accuracy_column, auc_column, rows_column = st.columns(3)
        accuracy_column.metric("Accuracy", f"{metrics['accuracy']:.1%}", border=True,
            help="Share of correct classifications. Not evidence of quality on its own: "
                 "78% of applicants repay, so a majority-guesser already reaches that.")
        auc_column.metric("ROC AUC", f"{metrics['roc_auc']:.3f}", border=True,
            help="Ranking quality across all thresholds.")
        rows_column.metric("Default recall", f"{metrics['default_recall']:.1%}", border=True,
            help="Share of actual defaults the model flags. The business-critical metric "
                 "for a risk screener.")
        st.caption(
            f"PR-AUC {metrics['pr_auc']:.3f} · Brier {metrics['brier']:.3f} · "
            f"precision {metrics['precision']:.1%} · F1 {metrics['f1']:.1%} · "
            f"trained on {metrics['n_train']:,} applicants"
        )

        comparison = pd.DataFrame(
            {
                "Segment": ["This applicant", "Portfolio average"],
                "Default probability": [
                    assessment.model_probability,
                    metrics.get("train_default_rate", 0.0),
                ],
            }
        )
        st.bar_chart(
            comparison,
            x="Segment",
            y="Default probability",
            height=200,
            alt="Applicant model default probability next to the portfolio average.",
        )


def render_portfolio(reference: pd.DataFrame, assessment: RiskAssessment) -> None:
    """Where the applicant's LTI sits inside the historical portfolio."""
    with st.container(border=True):
        st.subheader("Loan-to-income position")
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
            height=240,
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
        st.caption(
            f"Model: RandomForestClassifier · {len(metrics['features'])} features · "
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
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    st.markdown("## RiskPulse Dashboard")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("Credit risk & BNPL", color="blue")
        st.badge("Model live", icon=":material/bolt:", color="green")

st.caption(
    "Model default probability blended with affordability and credit history "
    "into one screening score."
)

if st.button(
    "Run assessment",
    type="primary",
    icon=":material/play_arrow:",
    width="stretch",
    help="Scores the current sidebar inputs and adds them to the session history.",
):
    current = assess(pipeline, values, customer_label)
    st.session_state["assessment"] = current
    record_assessment(customer_label, values, current)

assessment: RiskAssessment | None = st.session_state.get("assessment")

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

# History + benchmark ------------------------------------------------------
history_column, benchmark_column = st.columns([3, 2], gap="medium")
with history_column:
    render_history()
with benchmark_column:
    render_benchmark(assessment, metrics)

st.space("small")

# Portfolio position --------------------------------------------------------
with st.expander(":material/group: Portfolio position", expanded=False):
    render_portfolio(get_reference_profiles(), assessment)

with st.container(horizontal=True):
    with st.popover(":material/menu_book: Methodology"):
        render_methodology(metrics)
    with st.popover(":material/receipt_long: Assessment details"):
        st.markdown(
            f"**Raw model probability:** {assessment.raw_model_probability:.2%}  \n"
            f"**Clipped probability:** {assessment.model_probability:.2%}  \n"
            f"**Model margin (|2p-1|):** {assessment.model_confidence:.1%}  \n"
            f"**Loan grade:** {assessment.loan_grade}"
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
