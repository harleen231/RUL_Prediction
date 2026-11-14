"""
AeroGuard Dashboard Cards & UI Component Helpers
Provides structured metric banners, aerospace telemetry badges,
and safety risk indicator cards.
"""

import streamlit as st
from typing import Dict, Any
from dashboard.components.styles import render_kpi_card, render_status_badge, render_hazard_alert


def render_header(subset: str = "FD001", best_model_name: str = "XGBoost"):
    """Renders the top navigation and AeroGuard aerospace header."""
    st.markdown(
        f"""
        <div style="border-bottom: 1px solid rgba(56, 189, 248, 0.2); padding-bottom: 16px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; color: #F8FAFC; letter-spacing: -0.03em;">
                        ✈️ AEROGUARD
                    </h1>
                    <div style="color: #38BDF8; font-weight: 500; font-size: 0.95rem; margin-top: 4px;">
                        Intelligent Aero-Engine Degradation Monitoring & Remaining Useful Life (RUL) Prediction
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(14, 165, 233, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-right: 8px;">
                        DATASET: NASA C-MAPSS ({subset})
                    </span>
                    <span style="background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                        ACTIVE MODEL: {best_model_name.upper()}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_fleet_metrics(stats: Dict[str, Any]):
    """Renders 6 high-level KPI cards across the fleet."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(render_kpi_card("Fleet Analyzed", f"{stats['total_engines']}", "Active Turbofans", "#F8FAFC"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Mean RUL", f"{stats['avg_rul']:.1f}", "Operating Cycles", "#38BDF8"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Healthy (>75)", f"{stats['healthy_count']}", "Nominal Profile", "#10B981"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Attention (31-75)", f"{stats['attention_count']}", "Routine Depot Check", "#F59E0B"), unsafe_allow_html=True)
    with c5:
        st.markdown(render_kpi_card("Warning (11-30)", f"{stats['warning_count']}", "Plan Overhaul", "#F97316"), unsafe_allow_html=True)
    with c6:
        st.markdown(render_kpi_card("Critical (≤10)", f"{stats['critical_count']}", "Ground Engine", "#EF4444"), unsafe_allow_html=True)


def render_engine_diagnostic_card(diagnostic: Dict[str, Any]):
    """Renders detailed diagnostics for an individual selected engine."""
    status = diagnostic["status"]
    is_hazard = status["is_hazard"]

    st.markdown(
        f"""
        <div class="aeroguard-kpi-card" style="margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">
                        ENGINE UNIT #{diagnostic['engine_id']}
                    </span>
                    <span style="margin-left: 12px; color: #94A3B8; font-size: 0.9rem;">
                        Current Observed Cycles: <span class="telemetry-val" style="color: #38BDF8;">{diagnostic['current_cycle']}</span>
                    </span>
                </div>
                <div>
                    {render_status_badge(status['category'])}
                </div>
            </div>
            <div style="display: flex; gap: 30px; margin-top: 8px;">
                <div>
                    <div style="color: #94A3B8; font-size: 0.78rem; text-transform: uppercase;">Estimated Remaining Useful Life</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {status['color']}; font-family: 'JetBrains Mono', monospace;">
                        {diagnostic['predicted_rul']} <span style="font-size: 1.1rem; font-weight: 500;">cycles</span>
                    </div>
                </div>
                <div style="border-left: 1px solid rgba(148, 163, 184, 0.2); padding-left: 20px;">
                    <div style="color: #94A3B8; font-size: 0.78rem; text-transform: uppercase;">Maintenance Protocol Directive</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #E2E8F0; margin-top: 4px;">
                        {status['description']}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if is_hazard:
        st.markdown(render_hazard_alert(status["hazard_message"]), unsafe_allow_html=True)
