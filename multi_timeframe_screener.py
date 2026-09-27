"""Swing Screener AI - входна точка на Streamlit приложението.
Стартиране: streamlit run multi_timeframe_screener.py
Логиката е в модулите: photon (стратегията), universe (универс/данни),
indicators, portfolio_ui, ai_client, ui_common; legacy_strategies пази
неизползваните стари стратегии."""

import streamlit as st

from photon import render_photon_strategy
from portfolio_ui import render_portfolio_section
from ui_common import hide_st_style

st.set_page_config(
    page_title="Swing Screener AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(hide_st_style, unsafe_allow_html=True)


# ============================================================================
# ИНТЕРФЕЙС: избор на стратегия
# ============================================================================

st.markdown(
    '<div class="app-hero"><h1>⚡ Swing Screener AI</h1>'
    '<span class="app-sub">EU/EEA · EUR · Trading 212</span></div>',
    unsafe_allow_html=True,
)

strategy = st.radio(
    "Избери секция:",
    [
        "🧭 Photon Phases (BOS/CHoCH, Phase A/B, long-only)",
        "💼 Портфолио & P&L",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

if strategy.startswith("🧭"):
    render_photon_strategy()
else:
    render_portfolio_section()
