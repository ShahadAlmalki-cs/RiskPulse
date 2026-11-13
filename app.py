"""RiskPulse — credit risk & BNPL screening dashboard.

Run from the project root:

    streamlit run app.py

Layout
------
Sidebar          Applicant inputs (profile, loan request, policy grade preview)
Top bar          Title, subtitle, model status badge
Metric row       Blended risk score · Model default probability · Model margin · LTI
Analytics row    Risk score breakdown | Decision card
Summary row      Applicant summary | LTI position
Bottom row       Model performance | Recent assessments

All risk math lives in ``src/risk.py``, feature preparation in
``src/features.py`` and artifact loading in ``src/artifacts.py``; this module is
purely presentation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

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
    page_title="RiskPulse — Credit risk assessment",
    page_icon=":material/monitoring:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
# The lockup is the brand asset itself (riskpulse-logo-horizontal-reversed.svg):
# every path, stroke width and colour is the file's own. Only the colour of the
# "Risk" wordmark and the R monogram is substituted, because this file is the
# reversed (dark-background) version and the app chrome is a light surface.
# "Pulse", the ECG stroke and the coral dot always keep their brand colours.
LOGO_SVG_PATH = Path(__file__).resolve().parent / "riskpulse-logo-horizontal-reversed.svg"


def _load_logo_svg() -> str:
    """The logo as responsive inline SVG (metadata and fixed size stripped)."""
    try:
        raw = LOGO_SVG_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    raw = re.sub(r"<metadata>.*?</metadata>", "", raw, flags=re.S)
    raw = re.sub(r"<title[^>]*>.*?</title>", "", raw, flags=re.S)
    raw = raw.replace(' aria-labelledby="t"', ' aria-label="RiskPulse"')
    raw = raw.replace('width="1148" height="256"', "")
    raw = raw.replace(
        "<svg ", '<svg class="rp-logo" preserveAspectRatio="xMinYMid meet" '
    )
    # Streamlit's markdown sanitiser drops raw newlines, which would glue the
    # self-closing <path/> elements together; join each tag onto one line.
    raw = re.sub(r">\s*\n\s*<", "><", raw)
    return raw.strip()


def _logo(color: str) -> str:
    """Logo recoloured for the given surface: white -> ``color``."""
    svg = _load_logo_svg()
    if not svg:
        return "<strong>RiskPulse</strong>"
    return svg.replace('stroke="#FFFFFF"', f'stroke="{color}"').replace(
        'fill="#FFFFFF"', f'fill="{color}"'
    )


LOGO_SVG = _logo("#FFFFFF")
LOGO_SVG_INK = _logo("#0B1528")
# In dark mode the lockup is used exactly as drawn (white on navy); in light
# mode it is re-inked. Decided from the active theme so the sidebar stays legible.
LOGO_SVG_SIDEBAR = LOGO_SVG if st.context.theme.type == "dark" else LOGO_SVG_INK

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
# One palette for both themes. Every card, badge and chart reads from here, so
# the light and dark themes stay in sync and text always keeps a strong
# contrast against its own surface. Level colours carry a dark-mode variant;
# ``level_style`` picks the pair for the active theme.
LEVEL_STYLE = {
    "Low": {
        "color": "green",
        "icon": "check_circle",
        "alert": st.success,
        "accent": "#1E9E6A",
        "soft": "rgba(30, 158, 106, 0.12)",
        "text": "#0F7A50",
        "dark_accent": "#5BD0A2",
        "dark_soft": "rgba(91, 208, 162, 0.12)",
        "dark_text": "#5BD0A2",
    },
    "Medium": {
        "color": "orange",
        "icon": "warning",
        "alert": st.warning,
        "accent": "#D9930D",
        "soft": "rgba(217, 147, 13, 0.14)",
        "text": "#8A5A00",
        "dark_accent": "#E8B75A",
        "dark_soft": "rgba(232, 183, 90, 0.12)",
        "dark_text": "#E8B75A",
    },
    "High": {
        "color": "red",
        "icon": "cancel",
        "alert": st.error,
        "accent": "#E15241",
        "soft": "rgba(225, 82, 65, 0.14)",
        "text": "#B02A1C",
        "dark_accent": "#F6866E",
        "dark_soft": "rgba(246, 134, 110, 0.12)",
        "dark_text": "#F6866E",
    },
}


def is_dark_theme() -> bool:
    """True when the viewer resolved to the dark theme."""
    return getattr(st.context.theme, "type", "light") == "dark"


def level_style(level: str, dark: bool) -> dict:
    """Outcome colours for the active theme (same three outcomes, both themes)."""
    light = LEVEL_STYLE[level]
    if not dark:
        return light
    return {
        **light,
        "accent": light["dark_accent"],
        "soft": light["dark_soft"],
        "text": light["dark_text"],
    }

# ECharts bar colors matched to the brand palette.
SIGNAL_COLORS = ["#335393", "#7FB0F5", "#0B1528"]

HISTORY_SESSION_KEY = "assessment_history"
ASSESSMENT_SESSION_KEY = "assessment"
INPUTS_SESSION_KEY = "assessment_inputs"

# Card styling. Colors come from CSS variables that Streamlit exposes for the
# active theme, so the same rules work in light and dark mode without a second
# stylesheet. Text colors are always paired with their own surface color.
CARD_CSS = """
<style>
    /* ---- Design tokens ---------------------------------------------------
       Every value is derived from the logo: navy ink, white, periwinkle
       "pulse" and the coral end-point dot. Surfaces, text and borders are read
       from the active Streamlit theme at runtime rather than hard-coded, so the
       custom cards follow light and dark mode without a second stylesheet.
       The hex fallbacks are the light values: they are only used if a token is
       ever missing. Accents come from the palette both themes share. */
    :root {
        --rp-surface: var(--st-secondary-background-color, #FFFFFF);
        --rp-canvas: var(--st-background-color, #F2F5FD);
        --rp-text: var(--st-text-color, #0B1528);
        --rp-muted: var(--st-gray-text-color, #5F6B85);
        --rp-border: var(--st-border-color, #DDE4F4);
        --rp-primary: var(--st-primary-color, #3E6FD6);

        --rp-pulse: #7FA3EE;
        --rp-pulse-strong: var(--st-primary-color, #3E6FD6);
        --rp-coral: #F6866E;
        --rp-ink: #0B1528;
        --rp-on-primary: #FFFFFF;

        /* Nested fills are tinted from the real surface, never a fixed light
           blue: --rp-canvas is the page background the theme bridge reads. */
        --rp-soft: color-mix(in srgb, var(--st-text-color, #0B1528) 7%, var(--rp-canvas, #F2F5FD));
        --rp-soft-strong: color-mix(in srgb, var(--st-text-color, #0B1528) 12%, var(--rp-canvas, #F2F5FD));
        --rp-input: color-mix(in srgb, var(--st-text-color, #0B1528) 3%, var(--rp-canvas, #F2F5FD));
        --rp-shadow: 0 1px 2px rgba(11, 21, 40, 0.05), 0 6px 20px rgba(11, 21, 40, 0.04);
        --rp-radius: 16px;

        /* Outcome colours are set inline per assessment (LEVEL_STYLE), the same
           in both themes; these are only for non-assessment chrome. */
        --rp-ok: #0F7A50;
        --rp-ok-soft: rgba(15, 122, 80, 0.10);
        --rp-warn: #82590A;
        --rp-warn-soft: rgba(154, 106, 0, 0.12);
        --rp-bad: #A8371F;
        --rp-bad-soft: rgba(194, 65, 47, 0.10);
    }
    /* Dark outcome colours for the gauge, ladder bands and decision panel.
       Scoped to a dark colour-scheme preference so light mode is untouched;
       the per-assessment colours render inline from ``level_style``. */
    @media (prefers-color-scheme: dark) {
        :root {
            --rp-ok: #5BD0A2;
            --rp-warn: #E8B75A;
            --rp-bad: #F6866E;
        }
    }

    /* ---- Type ------------------------------------------------------------ */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        letter-spacing: -0.015em;
    }
    .stApp h2 { font-size: 15px; font-weight: 600; }
    .stApp h3 { font-size: 14px; font-weight: 600; }

    /* Section eyebrow above every main block */
    .rp-eyebrow {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2px 0 10px 0;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--rp-muted);
    }
    .rp-eyebrow::before {
        content: "";
        width: 18px;
        height: 2px;
        border-radius: 2px;
        background: var(--rp-pulse);
    }

    /* ---- Metric tiles ---------------------------------------------------- */
    div[data-testid="stMetric"] {
        background: var(--rp-surface) !important;
        border: 1px solid var(--rp-border) !important;
        border-radius: var(--rp-radius);
        padding: 14px 16px;
        box-shadow: var(--rp-shadow);
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {
        color: var(--rp-muted) !important;
        font-weight: 600;
        font-size: 12.5px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--rp-text) !important;
        font-weight: 700;
        letter-spacing: -0.035em;
        font-variant-numeric: tabular-nums;
    }
    div[data-testid="stMetricDelta"] {
        color: var(--rp-muted) !important;
    }

    /* ---- Cards (bordered containers) ------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--rp-surface) !important;
        border: 1px solid var(--rp-border) !important;
        border-radius: var(--rp-radius);
    }

    /* ---- Controls: hover / focus / active / disabled --------------------- */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        background: var(--rp-input) !important;
        border-color: var(--rp-border) !important;
        transition: border-color 120ms ease, box-shadow 120ms ease;
    }
    div[data-testid="stTextInput"] input:hover,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        border-color: var(--rp-pulse) !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--rp-pulse-strong) !important;
        box-shadow: 0 0 0 3px var(--rp-soft-strong) !important;
    }
    div[data-testid="stWidgetLabel"] p {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--rp-text);
    }
    .stButton > button {
        font-weight: 600;
        border-radius: 10px;
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    /* Streamlit labels the primary button white regardless of primary colour;
       on the dark theme's light periwinkle that is 2.2:1. */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        color: var(--rp-on-primary, #FFFFFF) !important;
    }
    .stButton > button:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(11, 21, 40, 0.14);
    }
    .stButton > button:active:not(:disabled) {
        transform: translateY(0);
        box-shadow: none;
    }
    .stButton > button:focus-visible,
    div[data-testid="stTextInput"] input:focus-visible,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        outline: 2px solid var(--rp-pulse-strong);
        outline-offset: 2px;
    }
    .stButton > button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
        box-shadow: none;
    }

    /* ---- Sidebar: brand lockup + numbered sections ----------------------- */
    section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {
        padding-bottom: 0;
    }
    .rp-logo {
        display: block;
        width: auto;
        max-width: 100%;
        height: 30px;
    }
    /* On the light theme the lockup is the reversed (white-ink) file, so its
       "Risk" wordmark is white on a white sidebar and disappears. Invert and
       rotate the hue back so white ink -> navy while the brand periwinkle and
       coral keep their own hues; dark mode is untouched. */
    @media (prefers-color-scheme: light) {
        .rp-logo { filter: invert(1) hue-rotate(180deg); }
    }
    .rp-brand-sub {
        margin: 12px 0 2px 0;
        font-size: 11.5px;
        line-height: 1.45;
        color: var(--rp-muted);
    }
    .rp-step {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 18px 0 8px 0;
    }
    .rp-step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 22px;
        border-radius: 7px;
        background: var(--rp-soft);
        color: var(--rp-pulse-strong);
        font-size: 10.5px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .rp-step-title {
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--rp-text);
        white-space: nowrap;
    }
    .rp-step-rule {
        flex: 1;
        height: 1px;
        background: var(--rp-border);
    }
    .rp-preview {
        border: 1px solid var(--rp-border);
        border-radius: 12px;
        background: var(--rp-soft);
        padding: 2px 12px;
    }
    .rp-preview-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 9px 0;
        font-size: 12.5px;
        color: var(--rp-muted);
    }
    .rp-preview-row + .rp-preview-row {
        border-top: 1px solid var(--rp-border);
    }
    .rp-preview-key {
        color: var(--rp-muted);
        white-space: nowrap;
    }
    .rp-preview-grade {
        white-space: nowrap;
    }
    .rp-preview-grade,
    .rp-preview-value {
        font-size: 13px;
        font-weight: 700;
        color: var(--rp-text);
        font-variant-numeric: tabular-nums;
    }
    .rp-preview-grade {
        padding: 2px 9px;
        border-radius: 999px;
        background: var(--rp-soft-strong);
        color: var(--rp-pulse-strong);
    }
    .rp-preview-note {
        margin-top: 8px;
        font-size: 11px;
        line-height: 1.45;
        color: var(--rp-muted);
    }

    /* ---- Inline field error (Annual income) ------------------------------ */
    .rp-field-error {
        display: flex;
        align-items: flex-start;
        gap: 7px;
        margin: -4px 0 10px 0;
        font-size: 12.5px;
        line-height: 1.4;
        font-weight: 600;
        /* Streamlit's theme-aware error colour: readable in light and dark. */
        color: var(--st-red-text-color, #C2412F);
    }
    .rp-field-error-dot {
        flex: 0 0 auto;
        width: 7px;
        height: 7px;
        margin-top: 5px;
        border-radius: 999px;
        background: currentColor;
    }

    /* ---- Top bar --------------------------------------------------------- */
    .rp-topbar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 0 0 16px 0;
        border-bottom: 1px solid var(--rp-border);
    }
    .rp-topbar-left {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
        min-width: 0;
    }
    .rp-topbar-divider {
        width: 1px;
        height: 34px;
        background: var(--rp-border);
    }
    .rp-topbar-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--rp-text);
        line-height: 1.25;
    }
    .rp-topbar-sub {
        font-size: 12.5px;
        color: var(--rp-muted);
        margin-top: 2px;
    }
    .rp-pills {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .rp-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 600;
        border: 1px solid var(--rp-border);
        background: var(--rp-surface);
        color: var(--rp-muted);
        white-space: nowrap;
    }
    .rp-pill--live {
        border-color: rgba(15, 122, 80, 0.35);
        background: var(--rp-ok-soft);
        color: var(--rp-ok);
    }
    .rp-dot {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: currentColor;
    }

    /* ---- Result hero: decision first, scores beside it ------------------- */
    .rp-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        flex-wrap: wrap;
        align-items: flex-end;
        justify-content: space-between;
        gap: 18px 28px;
        padding: 24px 26px 22px 26px;
        border-radius: 20px;
        border: 1px solid var(--rp-border);
        background:
            radial-gradient(120% 170% at 100% 0%, var(--rp-soft-strong) 0%, transparent 60%),
            var(--rp-surface);
        box-shadow: var(--rp-shadow);
    }
    /* The logo's ECG trace, reused as a single quiet motif. */
    .rp-hero::after {
        content: "";
        position: absolute;
        right: 26px;
        top: 16px;
        width: 132px;
        height: 24px;
        opacity: 0.55;
        background: no-repeat center / contain
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 132 24'%3E%3Cpath d='M2 14L28 14L37 3L48 20L58 9L66 14L124 14' fill='none' stroke='%237FA3EE' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='124' cy='14' r='3.4' fill='%23F6866E'/%3E%3C/svg%3E");
        pointer-events: none;
    }
    .rp-hero-main { position: relative; z-index: 1; }
    .rp-hero-name {
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--rp-muted);
    }
    .rp-hero-decision {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 4px;
    }
    .rp-hero-dot {
        flex: 0 0 auto;
        width: 13px;
        height: 13px;
        border-radius: 999px;
        background: var(--rp-accent);
        box-shadow: 0 0 0 6px var(--rp-accent-soft);
    }
    .rp-hero-title {
        font-size: clamp(30px, 4vw, 40px);
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--rp-text);
        margin: 0;
        padding-right: 120px;
    }
    .rp-hero-sub {
        margin-top: 10px;
        font-size: 13.5px;
        color: var(--rp-muted);
        max-width: 50ch;
        line-height: 1.5;
    }
    .rp-hero-side {
        position: relative;
        z-index: 1;
        display: flex;
        gap: 30px;
        flex-wrap: wrap;
    }
    .rp-hero-kpi-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--rp-muted);
    }
    .rp-hero-kpi-value {
        margin-top: 3px;
        font-size: 23px;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--rp-text);
        font-variant-numeric: tabular-nums;
    }
    .rp-hero-kpi-sub {
        margin-top: 2px;
        font-size: 11.5px;
        color: var(--rp-muted);
    }

    /* ---- Score ladder: Approve | Review | Decline ------------------------ */
    .rp-ladder {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 22px;
        max-width: 560px;
    }
    .rp-ladder-band {
        position: relative;
        height: 6px;
        border-radius: 999px;
        background: var(--rp-soft-strong);
    }
    /* Band widths follow the policy thresholds (15% / 35% of 100%). */
    .rp-ladder-band--ok { flex: 15; }
    .rp-ladder-band--review { flex: 20; }
    .rp-ladder-band--bad { flex: 65; }
    .rp-ladder-marker.is-first { left: 0; }
    .rp-ladder-marker.is-last { left: auto; right: 0; transform: translateX(1.5px); }
    .rp-ladder-band--ok.band-on { background: var(--rp-ok); }
    .rp-ladder-band--review.band-on { background: var(--rp-warn); }
    .rp-ladder-band--bad.band-on { background: var(--rp-bad); }
    /* Only the active band takes on its outcome colour; inactive bands stay
       neutral so the eye lands on the decision being shown. */
    .rp-ladder-marker {
        position: absolute;
        top: -5px;
        width: 3px;
        height: 16px;
        border-radius: 2px;
        background: var(--rp-text);
        box-shadow: 0 0 0 2px var(--rp-surface);
        transform: translateX(-1.5px);
    }
    .rp-ladder-legend {
        display: flex;
        max-width: 560px;
        justify-content: space-between;
        margin-top: 8px;
        font-size: 11px;
        color: var(--rp-muted);
    }

    /* ---- Signal cards (model probability vs blended policy score) -------- */
    .rp-signal {
        height: 100%;
        display: flex;
        flex-shrink: 0;
        flex-direction: column;
        gap: 7px;
        padding: 18px 18px 16px 18px;
        border-radius: var(--rp-radius);
        border: 1px solid var(--rp-border);
        background: var(--rp-surface);
        box-shadow: var(--rp-shadow);
    }
    .rp-signal--model { border-top: 3px solid var(--rp-pulse); }
    .rp-signal--blend { border-top: 3px solid var(--rp-ink, var(--rp-text)); }
    .rp-signal-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }
    .rp-signal-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--rp-muted);
    }
    .rp-signal-tag {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 999px;
        /* Ink-on-tint reads on both themes; brand primary-on-tint is only
           4.3:1 in light mode, so it is not used for body-size text. */
        background: var(--rp-soft);
        color: var(--rp-text);
        white-space: nowrap;
    }
    .rp-signal-value {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.035em;
        line-height: 1.1;
        color: var(--rp-text);
        font-variant-numeric: tabular-nums;
    }
    .rp-signal-meta {
        font-size: 12px;
        color: var(--rp-muted);
        line-height: 1.45;
    }
    /* A fixed-height track: the bar sits in a column flex container, so it
       must not absorb (or be shrunk to) the leftover height. The spacer below
       it keeps the meta text aligned across both cards. */
    .rp-bar {
        flex: 0 0 5px;
        height: 5px;
        border-radius: 999px;
        background: var(--rp-soft-strong);
        overflow: hidden;
    }
    .rp-bar-spacer { flex: 1 1 auto; }
    .rp-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--rp-pulse);
    }
    .rp-bar-fill--blend { background: var(--rp-ink, var(--rp-text)); }

    /* ---- Decision panel -------------------------------------------------- */
    .rp-decision {
        flex-shrink: 0;
        border-radius: var(--rp-radius);
        padding: 16px 18px;
        border: 1px solid var(--rp-border);
        border-left: 4px solid var(--rp-accent);
        background: var(--rp-accent-soft);
    }
    .rp-decision-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--rp-muted);
    }
    .rp-decision-value {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin: 4px 0 6px 0;
        color: var(--rp-accent-text);
    }
    .rp-decision-meta {
        font-size: 13px;
        color: var(--rp-text);
        line-height: 1.5;
    }
    .rp-gauge {
        position: relative;
        height: 8px;
        border-radius: 999px;
        margin: 16px 0 6px 0;
        background: linear-gradient(
            90deg,
            var(--rp-ok) 0%,
            var(--rp-ok) 15%,
            var(--rp-warn) 15%,
            var(--rp-warn) 35%,
            var(--rp-bad) 35%,
            var(--rp-bad) 100%
        );
    }
    .rp-gauge-marker {
        position: absolute;
        top: -4px;
        width: 3px;
        height: 16px;
        border-radius: 2px;
        background: var(--rp-text);
        box-shadow: 0 0 0 2px var(--rp-surface);
        transform: translateX(-1.5px);
    }
    .rp-gauge-scale {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: var(--rp-muted);
        font-variant-numeric: tabular-nums;
    }

    /* ---- Key-value strip ------------------------------------------------- */
    .rp-kv {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px 18px;
    }
    .rp-kv-label {
        font-size: 11.5px;
        color: var(--rp-muted);
        margin-bottom: 3px;
    }
    .rp-kv-value {
        font-size: 14.5px;
        font-weight: 600;
        color: var(--rp-text);
    }

    /* ---- Card headings --------------------------------------------------- */
    .rp-card-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }
    .rp-card-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--rp-text);
        letter-spacing: -0.01em;
    }
    .rp-card-kicker {
        font-size: 11.5px;
        color: var(--rp-muted);
        font-weight: 500;
    }

    /* ---- Footer seal ----------------------------------------------------- */
    .rp-footer {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-top: 14px;
        border-top: 1px solid var(--rp-border);
    }
    .rp-footer .rp-logo { height: 22px; opacity: 0.85; }
    /* In the footer the mark already sits on ink-tinted SVG text for light
       mode, so it is not inverted again. */
    .rp-footer .rp-logo { filter: none; }
    /* The lockup is a markdown image, so it is an <a>: keep the link colour
       from tinting it. */
    a:has(> svg.rp-logo) {
        color: inherit;
        text-decoration: none;
    }
    @media (prefers-color-scheme: dark) {
        a:has(> svg.rp-logo):focus-visible,
        .rp-footer a:focus-visible {
            outline: 2px solid #9DBBF2;
            outline-offset: 3px;
            border-radius: 6px;
        }
    }
    .rp-footer-text {
        font-size: 12px;
        color: var(--rp-muted);
        line-height: 1.5;
    }

    /* Captions: readable, never a washed-out grey. */
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: var(--rp-muted) !important;
    }

    /* ---- Responsive ------------------------------------------------------ */
    @media (max-width: 900px) {
        .rp-hero-title { padding-right: 0; }
        .rp-hero::after { display: none; }
    }
    @media (max-width: 720px) {
        .rp-hero { padding: 20px 16px; border-radius: 16px; }
        .rp-hero-side { gap: 16px 24px; }
        .rp-hero-kpi-value { font-size: 19px; }
        .rp-topbar-divider { display: none; }
        .rp-kv { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .rp-signal-value { font-size: 28px; }
        .rp-decision-value { font-size: 21px; }
        .rp-ladder, .rp-ladder-legend { max-width: 100%; }
    }
    @media (max-width: 420px) {
        .rp-kv { grid-template-columns: 1fr; }
        .rp-topbar { padding-bottom: 12px; }
        .rp-hero-side { gap: 12px 20px; }
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


def _numeric_field(raw_text: str, label: str) -> tuple[float | None, str | None]:
    """Parse a free-text numeric field, returning ``(value, error)`` instead of raising.

    The sidebar uses this so a malformed value shows inline under its own field
    and never surfaces a traceback to the user.
    """
    try:
        return _parse_number(raw_text, label), None
    except ValueError as exc:
        return None, str(exc)


def _income_error(income: float | None, parse_error: str | None) -> str | None:
    """Return the message for an unusable annual income, or ``None`` if it is valid."""
    if parse_error is not None:
        return parse_error
    if income is None or income <= 0:
        return "Annual income must be greater than 0."
    return None


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
    cleaned = cleaned.replace("٫", ".").replace("٬", "")  # Arabic decimal / thousands marks
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
def _section_step(number: str, title: str) -> None:
    """Numbered sidebar section header (01 / 02 / 03)."""
    st.markdown(
        f'<div class="rp-step"><span class="rp-step-num">{number}</span>'
        f'<span class="rp-step-title">{title}</span>'
        f'<span class="rp-step-rule"></span></div>',
        unsafe_allow_html=True,
    )


def _field_note(message: str) -> None:
    """Inline validation message, rendered directly under its input."""
    st.markdown(
        f'<div class="rp-field-error"><span class="rp-field-error-dot"></span>'
        f'{message}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[dict[str, object], str, str | None]:
    """Collect applicant inputs in a grouped sidebar form."""
    with st.sidebar:
        st.markdown(
            f'{LOGO_SVG_SIDEBAR}'
            '<div class="rp-brand-sub">Pre-approval credit risk · before pricing</div>',
            unsafe_allow_html=True,
        )

        _section_step("01", "Applicant")
        customer_name = st.text_input(
            "Customer name", value="Applicant 1", placeholder="e.g. Sara Ali"
        )
        # Numeric fields are plain text inputs so the browser always shows
        # Latin (English) digits, regardless of the operating-system locale.
        person_age_text = st.text_input("Age", value="30", help="Years (18–100).")
        person_age, person_age_error = _numeric_field(person_age_text, "Age")
        if person_age_error:
            _field_note(person_age_error)

        person_income_text = st.text_input(
            "Annual income",
            value="60000",
            help="Gross annual income in USD; drives the loan-to-income ratio.",
        )
        person_income, person_income_parse_error = _numeric_field(
            person_income_text, "Annual income"
        )
        person_income_error = _income_error(person_income, person_income_parse_error)
        if person_income_error:
            # Inline error directly under the field (st.text_input has no error slot).
            _field_note(person_income_error)

        person_emp_length_text = st.text_input(
            "Employment length (years)", value="5.0", help="Years (0–60)."
        )
        person_emp_length, person_emp_length_error = _numeric_field(
            person_emp_length_text, "Employment length"
        )
        if person_emp_length_error:
            _field_note(person_emp_length_error)

        cb_person_cred_hist_text = st.text_input(
            "Credit history length (years)", value="5", help="Years (0–80)."
        )
        cb_person_cred_hist, cb_person_cred_hist_error = _numeric_field(
            cb_person_cred_hist_text, "Credit history length"
        )
        if cb_person_cred_hist_error:
            _field_note(cb_person_cred_hist_error)
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

        _section_step("02", "Loan request")
        loan_amnt_text = st.text_input(
            "Loan amount", value="10000", help="USD (500–1,000,000)."
        )
        loan_amnt, loan_amnt_error = _numeric_field(loan_amnt_text, "Loan amount")
        if loan_amnt_error:
            _field_note(loan_amnt_error)
        loan_intent = st.selectbox("Loan purpose", config.LOAN_INTENT_OPTIONS)

        values: dict[str, object] = {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "person_home_ownership": person_home_ownership,
            "cb_person_cred_hist_length": cb_person_cred_hist,
            "cb_person_default_on_file": cb_person_default_on_file,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
        }

        # Derived LTI is a division by income: never compute it for income <= 0.
        if person_income_error:
            prepared = None
        else:
            try:
                prepared = with_policy_fields(values)
            except ValueError:
                prepared = None

        _section_step("03", "Derived preview")
        if prepared is None:
            st.caption(
                "Policy grade and loan-to-income are unavailable until every "
                "input is valid."
            )
        else:
            st.markdown(
                '<div class="rp-preview">'
                '<div class="rp-preview-row"><span class="rp-preview-key">'
                "<strong>Policy grade</strong>"
                f'</span><span class="rp-preview-grade">Grade {prepared["loan_grade"]}</span>'
                "</div>"
                '<div class="rp-preview-row"><span>Loan-to-income</span>'
                '<span class="rp-preview-value">'
                f'{float(prepared["loan_percent_income"]):.1%}</span></div>'
                "</div>"
                '<div class="rp-preview-note">Internal heuristic from this '
                "app's policy logic — not a lender-assigned credit grade.</div>",
                unsafe_allow_html=True,
            )

        returned_label = (customer_name or "Applicant").strip()
        return values, returned_label, person_income_error


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header(model_meta: str) -> None:
    """Top bar: page title and model status. The brand lockup lives in the sidebar."""
    st.markdown(
        f"""
        <div class="rp-topbar">
            <div class="rp-topbar-left">
                <div>
                    <div class="rp-topbar-title">Assessment dashboard</div>
                    <div class="rp-topbar-sub">Pre-approval screening — before pricing</div>
                </div>
            </div>
            <div class="rp-pills">
                <span class="rp-pill rp-pill--live"><span class="rp-dot"></span>Model live</span>
                <span class="rp-pill">{model_meta}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_hero(customer_label: str, assessment: RiskAssessment) -> None:
    """The single most important block: policy decision, then the two scores.

    Visual ordering encodes the relationship between the three numbers:
    the decision is the headline, the blended policy score is the driver
    (large, dark), the model probability is the reference input (smaller,
    periwinkle) and is explicitly labelled a different kind of quantity.
    """
    style = level_style(assessment.level, is_dark_theme())
    level_text = {
        "Low": f"Blended score is below {config.LOW_RISK_MAX:.0%} — no manual review required.",
        "Medium": (
            f"Blended score is in the {config.LOW_RISK_MAX:.0%}–"
            f"{config.MEDIUM_RISK_MAX:.0%} band — an analyst must review the file."
        ),
        "High": (
            f"Blended score exceeds {config.MEDIUM_RISK_MAX:.0%} — the request "
            "should not be approved as submitted."
        ),
    }[assessment.level]
    band_on = {"ok": "band-on" if assessment.level == "Low" else "",
               "review": "band-on" if assessment.level == "Medium" else "",
               "bad": "band-on" if assessment.level == "High" else ""}
    # The marker sits inside the band the score falls in, positioned by the
    # score's place within that band, so it lines up with its legend label.
    low, medium = config.LOW_RISK_MAX, config.MEDIUM_RISK_MAX
    score = min(max(assessment.score, 0.0), 1.0)
    if score < low:
        band, within = 0, score / low
    elif score <= medium:
        band, within = 1, (score - low) / (medium - low)
    else:
        band, within = 2, (score - medium) / (1.0 - medium)
    markers = ["", "", ""]
    markers[band] = (
        f'<div class="rp-ladder-marker" style="left: {within * 100:.1f}%;"></div>'
    )

    st.markdown(
        f"""
        <div class="rp-hero" style="--rp-accent: {style['accent']};
             --rp-accent-soft: {style['soft']}; --rp-accent-text: {style['text']};">
            <div class="rp-hero-main">
                <div class="rp-hero-name">{customer_label} · policy decision</div>
                <div class="rp-hero-decision">
                    <span class="rp-hero-dot"></span>
                    <h1 class="rp-hero-title">{assessment.decision}</h1>
                </div>
                <div class="rp-hero-sub">{level_text}</div>
                <div class="rp-ladder" role="img"
                     aria-label="Score position on the Approve, review and decline bands">
                    <div class="rp-ladder-band rp-ladder-band--ok {band_on['ok']}">{markers[0]}</div>
                    <div class="rp-ladder-band rp-ladder-band--review {band_on['review']}">{markers[1]}</div>
                    <div class="rp-ladder-band rp-ladder-band--bad {band_on['bad']}">{markers[2]}</div>
                </div>
                <div class="rp-ladder-legend">
                    <span>Approve &lt; {config.LOW_RISK_MAX:.0%}</span>
                    <span>Review ≤ {config.MEDIUM_RISK_MAX:.0%}</span>
                    <span>Decline &gt; {config.MEDIUM_RISK_MAX:.0%}</span>
                </div>
            </div>
            <div class="rp-hero-side">
                <div>
                    <div class="rp-hero-kpi-label">Blended policy score</div>
                    <div class="rp-hero-kpi-value">{assessment.score:.1%}</div>
                    <div class="rp-hero-kpi-sub">{assessment.level} risk</div>
                </div>
                <div>
                    <div class="rp-hero-kpi-label">Model default probability</div>
                    <div class="rp-hero-kpi-value">{assessment.model_probability:.1%}</div>
                    <div class="rp-hero-kpi-sub">clipped · {assessment.raw_model_probability:.1%} raw</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Signal cards
# ---------------------------------------------------------------------------
def render_signal_cards(assessment: RiskAssessment) -> None:
    """Model probability (a model output) beside the blended policy score.

    They are deliberately not styled alike: different borders, tags and bar
    colours, because only one of them is a statistically evaluated quantity.
    """
    model_col, blend_col = st.columns(2, gap="medium")
    with model_col:
        st.markdown(
            f"""
            <div class="rp-signal rp-signal--model">
                <div class="rp-signal-head">
                    <span class="rp-signal-label">Model Default Probability</span>
                    <span class="rp-signal-tag">Model output</span>
                </div>
                <div class="rp-signal-value">{assessment.model_probability:.1%}</div>
                <div class="rp-bar"><div class="rp-bar-fill"
                     style="width: {assessment.model_probability * 100:.1f}%;"></div></div>
                <div class="rp-bar-spacer"></div>
                <div class="rp-signal-meta">
                    RandomForest vote share, clipped to 2%–98%.
                    Raw {assessment.raw_model_probability:.1%} · not a calibrated probability.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with blend_col:
        st.markdown(
            f"""
            <div class="rp-signal rp-signal--blend">
                <div class="rp-signal-head">
                    <span class="rp-signal-label">Blended Risk Score</span>
                    <span class="rp-signal-tag">Policy overlay</span>
                </div>
                <div class="rp-signal-value">{assessment.score:.1%}</div>
                <div class="rp-bar"><div class="rp-bar-fill rp-bar-fill--blend"
                     style="width: {assessment.score * 100:.1f}%;"></div></div>
                <div class="rp-bar-spacer"></div>
                <div class="rp-signal-meta">
                    40% model + 40% LTI + 20% prior default, set by policy.
                    Drives the decision; not a model output.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_inputs_used(assessment: RiskAssessment) -> None:
    """Secondary inputs in one compact strip, visually below the two scores."""
    items = [
        ("Loan-to-income", f"{assessment.loan_to_income:.1%}"),
        ("Model margin", f"{assessment.model_confidence:.0%}"),
        ("Policy grade", f"Grade {assessment.loan_grade}"),
    ]
    columns = st.columns(3, gap="small")
    for column, (label, value) in zip(columns, items):
        with column:
            st.metric(label, value, border=True)


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
        header_column, spacer_column = st.columns([3, 1])
        with header_column:
            st.subheader("Risk score breakdown")
        with spacer_column:
            st.caption("Policy score")
        st.caption(
            "Points each factor adds to the blended policy score (weight × "
            "signal risk). The bars sum to the final score."
        )
        st.echarts_chart(
            {
                "dataset": {"source": frame},
                "grid": {"left": 8, "right": 40, "top": 8, "bottom": 8, "containLabel": True},
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
                        "barMaxWidth": 26,
                        "itemStyle": {"borderRadius": [0, 6, 6, 0]},
                        "label": {"show": True, "position": "right", "formatter": "{@Label}"},
                    }
                ],
            },
            height=250,
            alt=(
                "Points added to the blended score by factor: "
                + ", ".join(
                    f"{item.label} {item.points:.1%}" for item in assessment.contributions
                )
            ),
        )

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
    """Prominent decision state, score gauge and next steps."""
    style = level_style(assessment.level, is_dark_theme())
    marker = min(max(assessment.score, 0.0), 1.0) * 100

    with st.container(border=True):
        st.markdown(
            '<div class="rp-card-head"><span class="rp-card-title">'
            "Policy outcome</span>"
            '<span class="rp-card-kicker">and next steps</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="rp-decision" style="--rp-accent: {style['accent']};
                 --rp-accent-soft: {style['soft']}; --rp-accent-text: {style['text']};">
                <div class="rp-decision-label">Recommended outcome</div>
                <div class="rp-decision-value">
                    <span class="rp-hero-dot" style="width:10px;height:10px;
                          box-shadow:none;"></span>{assessment.decision}
                </div>
                <div class="rp-decision-meta">
                    {assessment.level} risk · blended score {assessment.score:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="rp-gauge">
                <div class="rp-gauge-marker" style="left: {marker:.1f}%;"></div>
            </div>
            <div class="rp-gauge-scale">
                <span>0%</span>
                <span>Approve &lt; {config.LOW_RISK_MAX:.0%}</span>
                <span>Review ≤ {config.MEDIUM_RISK_MAX:.0%}</span>
                <span>100%</span>
            </div>
            """,
            unsafe_allow_html=True,
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
        st.markdown(
            '<div class="rp-card-head"><span class="rp-card-title">'
            f'{label}</span>'
            f'<span class="rp-card-kicker">{assessment.decision} · submitted '
            "application</span></div>",
            unsafe_allow_html=True,
        )
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
        columns = st.columns(3)
        items = list(summary.items())
        for index, (field, value) in enumerate(items):
            with columns[index % 3]:
                st.markdown(
                    f'<div class="rp-kv-label">{field}</div>'
                    f'<div class="rp-kv-value">{value}</div>',
                    unsafe_allow_html=True,
                )


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
                    "Loan amount", format="$%,.0f", width="small"
                ),
                "LTI": st.column_config.NumberColumn("LTI", format="percent", width="small"),
                # Wide enough for the progress bar plus its percentage label.
                "Score": st.column_config.ProgressColumn(
                    "Blended score", format="percent", min_value=0.0, max_value=1.0,
                    width="medium",
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

        stat_columns = st.columns(4, gap="small")
        stats = [
            ("Accuracy", f"{metrics['accuracy']:.1%}",
             "Share of correct classifications. Not sufficient on its own: "
             "78% of applicants repay, which is the majority-class baseline."),
            ("PR-AUC", f"{metrics['pr_auc']:.3f}",
             "Ranking quality under class imbalance; the primary metric "
             "at this 21.8% default base rate."),
            ("Brier", f"{metrics['brier']:.3f}",
             "Probability calibration. Consistent with imperfect calibration "
             "(RandomForest vote shares are not calibrated probabilities)."),
            ("Default Recall", f"{metrics['default_recall']:.1%}",
             "Share of actual defaults the model flags. The most important "
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
    with st.container(border=True):
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
                "smooth saturating curve, so scores change continuously with inputs."
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


def render_footer() -> None:
    """Closing seal: the mark again, and the standing disclaimer."""
    st.markdown(
        f"""
        <div class="rp-footer">
            {LOGO_SVG_SIDEBAR}
            <div class="rp-footer-text">
                RiskPulse supports — it does not replace — credit policy decisions.
                Always pair automated screening with human review. Educational
                demonstrator trained on a public credit dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# The cards and panels below draw with --rp-* tokens. Streamlit expresses the
# active theme as emotion class names (there are no --st-* variables and
# st.context.theme carries only "light"/"dark"), so this script republishes
# the two facts the page resolved for itself: the page and card backgrounds.
# Everything else derives from them in CARD_CSS, so light mode keeps its own
# values and dark mode gets the dark ones. Runs only with
# unsafe_allow_javascript because it reads computed styles (no arguments).
THEME_TOKEN_BRIDGE = """
<script>
if (!window.__rpThemeBridge) {
window.__rpThemeBridge = true;
(function () {
    function channel(v) {
        v = Number(v);
        return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    }
    function rgbOf(value) {
        var m = (value || '').match(/[\\d.]+/g);
        if (!m || m.length < 3) { return null; }
        return { r: Number(m[0]), g: Number(m[1]), b: Number(m[2]), a: m[3] === undefined ? 1 : Number(m[3]) };
    }
    function relativeLuminance(c) {
        return 0.2126 * channel(c.r / 255) + 0.7152 * channel(c.g / 255) + 0.0722 * channel(c.b / 255);
    }
    function luminanceDifference(a, b) {
        var la = relativeLuminance(a), lb = relativeLuminance(b);
        var hi = Math.max(la, lb), lo = Math.min(la, lb);
        return (hi + 0.05) / (lo + 0.05);
    }
    // Mirror of the two standalone theme palettes in .streamlit/config.toml
    // (primary, ink, secondaryBackground, grayText). A custom theme has no
    // "lifted card" setting, so --rp-surface must come from here: inferring it
    // from an element that already uses --rp-surface would feed back into
    // itself and collapse the cards onto the page.
    var RAMP = {
        light: {
            primary: { r: 62, g: 111, b: 214, a: 1 },
            ink: { r: 11, g: 21, b: 40, a: 1 },
            card: { r: 255, g: 255, b: 255, a: 1 },
            muted: { r: 95, g: 107, b: 133, a: 1 },
        },
        dark: {
            primary: { r: 143, g: 176, b: 240, a: 1 },
            ink: { r: 228, g: 234, b: 245, a: 1 },
            card: { r: 24, g: 35, b: 58, a: 1 },
            muted: { r: 168, g: 180, b: 201, a: 1 },
            onPrimary: { r: 11, g: 21, b: 40, a: 1 },   // ink on periwinkle
        },
    };
    RAMP.light.onPrimary = { r: 255, g: 255, b: 255, a: 1 };  // white on #3E6FD6
    function paint() {
        var app = document.querySelector('.stApp');
        var root = document.documentElement;
        if (!app) { return; }
        var page = rgbOf(getComputedStyle(app).backgroundColor);
        if (!page || page.a === 0) { return; }
        var pageDark = page.r + page.g + page.b < 3 * 140;
        var ramp = RAMP[pageDark ? 'dark' : 'light'];
        // Cards lift off the page: white on the light canvas, slate on the
        // dark one. If a theme ever matches the page exactly, keep the pair
        // distinguishable by shifting the card instead of reusing the page.
        var card = ramp.card;
        if (luminanceDifference(card, page) < 1.15) {
            card = pageDark
                ? { r: Math.min(255, page.r + 10), g: Math.min(255, page.g + 10), b: Math.min(255, page.b + 12), a: 1 }
                : { r: Math.max(0, page.r - 8), g: Math.max(0, page.g - 8), b: Math.max(0, page.b - 8), a: 1 };
        }
        var heading = document.querySelector('.stApp h1, .stApp h2, .stApp h3')
            || document.querySelector('[data-testid="stMarkdownContainer"] p');
        var text = heading ? getComputedStyle(heading).color : null;
        // Primary: the resolved button fill, else the ramp's brand accent.
        var button = document.querySelector('.stButton > button');
        var primary = button ? rgbOf(getComputedStyle(button).backgroundColor) : null;
        if (!primary || primary.a === 0) { primary = ramp.primary; }
        // Secondary text: the live caption colour once one exists (the theme
        // hydrates after first paint), otherwise the ramp's mid-grey.
        var captionEl = document.querySelector('[data-testid="stCaptionContainer"] p');
        var muted = captionEl ? rgbOf(getComputedStyle(captionEl).color) : ramp.muted;
        if (muted && (muted.a === 0 || luminanceDifference(muted, card) < 2)) {
            muted = ramp.muted;  // too close to its surface to read as secondary
        }
        function hex(c) {
            return '#' + [c.r, c.g, c.b].map(function (v) {
                return Math.round(v).toString(16).padStart(2, '0');
            }).join('');
        }
        function set(name, c) {
            if (c) { root.style.setProperty(name, hex(c)); }
        }
        set('--rp-surface', card);
        set('--rp-canvas', page);
        set('--rp-primary', primary);
        set('--rp-ink', ramp.ink);   // blend-card top edge in dark mode
        set('--rp-on-primary', ramp.onPrimary);
        if (text) { root.style.setProperty('--rp-text', text); }
        if (muted) { root.style.setProperty('--rp-muted', hex(muted)); }
    }
    // The theme's own stylesheets hydrate after first paint, so repaint on any
    // attribute change; a paint with an unresolved surface is a no-op.
    paint();
    new MutationObserver(paint).observe(document.body, {
        attributes: true, subtree: true, attributeFilter: ['class', 'style'],
    });
})();
}
</script>
"""


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown(CARD_CSS, unsafe_allow_html=True)
st.html(THEME_TOKEN_BRIDGE, unsafe_allow_javascript=True)

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
values, customer_label, income_error = render_sidebar()

# Check the inputs *before* the model or any division runs. An unusable income
# is already reported inline under its field, so it gets no second copy here.
input_error: str | None = None
try:
    validate_inputs(values)
except ValueError as exc:
    input_error = str(exc)
    if income_error is None:
        st.error(str(exc), title="Check the inputs", icon=":material/error:")

model_meta = "RandomForest · application-time features"
render_header(model_meta)

# Primary action sits directly under the top bar, full width, so it is the
# first thing the eye lands on after the brand.
if st.button(
    "Run assessment",
    type="primary",
    icon=":material/play_arrow:",
    width="stretch",
    disabled=input_error is not None,
):
    current = assess(pipeline, values, customer_label)
    st.session_state[ASSESSMENT_SESSION_KEY] = current
    st.session_state[INPUTS_SESSION_KEY] = inputs_fingerprint(values)
    record_assessment(customer_label, values, current)

if input_error is not None:
    # Field-level messages (income) are inline; other inputs fall back to the
    # page-level notice rendered above. No results are shown either way.
    st.space("small")
    render_history()
    st.stop()

assessment: RiskAssessment | None = st.session_state.get(ASSESSMENT_SESSION_KEY)
stale = assessment_is_stale(values)

if assessment is None:
    st.info(
        "Set the applicant profile and loan request in the sidebar, then choose "
        "**Run assessment** to see the decision, model probability and blended "
        "policy score.",
        title="Ready for an assessment",
        icon=":material/info:",
    )
    st.space("small")
    render_history()
    st.stop()

if stale:
    st.warning(
        "Inputs have changed since this assessment was run. The figures below "
        "describe the **previous** inputs — press **Run assessment** to refresh.",
        title="Outdated assessment",
        icon=":material/history:",
    )

st.space("small")

# 1. The decision ----------------------------------------------------------
render_result_hero(customer_label, assessment)

st.space("medium")

# 2. The two scores that drive it: model output vs policy overlay ----------
st.markdown('<div class="rp-eyebrow">Risk signals</div>', unsafe_allow_html=True)
render_signal_cards(assessment)
st.space("small")
render_inputs_used(assessment)

st.space("medium")

# 3. Why: contribution breakdown and recommended next steps ----------------
chart_column, decision_column = st.columns([3, 2], gap="medium")
with chart_column:
    render_contribution_chart(assessment)
with decision_column:
    render_decision_card(assessment)

st.space("medium")

# 4. Applicant strip beside the portfolio LTI position ---------------------
summary_column, lti_column = st.columns([3, 2], gap="medium")
with summary_column:
    render_applicant_summary(customer_label, values, assessment)
with lti_column:
    render_portfolio(get_reference_profiles(), assessment)

st.space("medium")

# 5. Model performance — full width so the four stat cards never clip -------
render_benchmark(metrics)

st.space("medium")

# 6. Recent assessments — full width so seven columns fit without clipping --
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
