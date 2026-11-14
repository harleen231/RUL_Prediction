"""
AeroGuard Interactive Streamlit Dashboard
Focused condition-based turbofan degradation monitoring and interactive
Remaining Useful Life (RUL) simulator featuring:
- XGBoost Regressor
- Random Forest Regressor
- Linear Regression
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="AeroGuard | Aero-Engine RUL Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Append project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import (
    DEFAULT_SUBSET,
    AVAILABLE_SUBSETS,
    DEFAULT_WINDOW_SIZE,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    SENSOR_METADATA,
    MAINTENANCE_LEVELS
)
from src.data_loader import load_raw_dataset
from src.rul_calculation import calculate_test_rul, get_test_last_cycle_records
from src.prediction import (
    AeroGuardPredictor,
    classify_maintenance_status,
    validate_uploaded_csv,
    NOMINAL_HEALTHY_BASELINES,
    FAILURE_ENDPOINTS,
    SIMULATOR_PRESETS
)
from dashboard.components.styles import (
    AEROGUARD_CSS,
    render_kpi_card,
    render_status_badge,
    render_hazard_alert
)

# Apply CSS styling
st.markdown(AEROGUARD_CSS, unsafe_allow_html=True)


# Cached Data & Model Loaders
@st.cache_resource(show_spinner=False)
def load_predictor():
    return AeroGuardPredictor()


@st.cache_data(show_spinner=False)
def load_fleet_data(subset: str = DEFAULT_SUBSET):
    _, df_test, s_rul = load_raw_dataset(subset)
    df_test_rul = calculate_test_rul(df_test, s_rul)
    df_test_last = get_test_last_cycle_records(df_test_rul)
    return df_test, df_test_rul, s_rul, df_test_last


# Load Core Artifacts
try:
    predictor = load_predictor()
    df_test, df_test_rul, s_rul, df_test_last = load_fleet_data()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.info("Please ensure that `train.py` has been executed to generate trained model artifacts.")
    st.stop()


# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 16px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.2);">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">✈️ AEROGUARD</div>
            <div style="font-size: 0.72rem; color: #38BDF8; letter-spacing: 0.1em; text-transform: uppercase;">
                Turbofan Degradation Simulator
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧭 Portal Sections")
    active_tab = st.radio(
        "Navigation",
        [
            "🚀 Interactive Engine Simulator",
            "📊 Fleet Health Overview",
            "📈 3-Model Benchmarks",
            "📁 Custom Telemetry Upload",
            "📖 System Documentation"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🤖 Active ML Models")
    st.markdown(
        """
        - 🟢 **XGBoost Regressor** (Gradient Boosted Trees)
        - 🔵 **Random Forest** (Bootstrap Ensembling)
        - 🟣 **Linear Regression** (Parametric Baseline)
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748B;">
            <b>Dataset:</b> NASA C-MAPSS (FD001)<br>
            <b>Validation:</b> Leakage-free engine-wise split<br>
            <b>Target:</b> Piecewise linear RUL (125-cycle clip)
        </div>
        """,
        unsafe_allow_html=True
    )


# Top Aerospace Banner
st.markdown(
    """
    <div style="border-bottom: 1px solid rgba(56, 189, 248, 0.2); padding-bottom: 14px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 1.95rem; color: #F8FAFC; letter-spacing: -0.02em;">
                    ✈️ AEROGUARD: Intelligent Aero-Engine Degradation & RUL Simulator
                </h1>
                <div style="color: #38BDF8; font-weight: 500; font-size: 0.9rem; margin-top: 3px;">
                    Multivariate Sensor Analytics & Real-Time RUL Prediction Engine
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(14, 165, 233, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                    MODELS: XGBoost | Random Forest | Linear
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# TAB 1: INTERACTIVE ENGINE SIMULATOR (PRIMARY FEATURE)
if active_tab == "🚀 Interactive Engine Simulator":
    st.subheader("Interactive Engine Degradation & What-If RUL Simulator")
    st.write(
        "Adjust flight operating cycles and thermodynamic sensor readings to simulate real-time engine degradation. "
        "Watch all 3 models (XGBoost, Random Forest, Linear Regression) predict remaining useful life simultaneously."
    )

    # Preset / Quick Loader Bar
    st.markdown("#### ⚡ Quick Presets & Test Engine Loader")
    col_pre1, col_pre2 = st.columns([2, 1])

    with col_pre1:
        preset_choice = st.selectbox(
            "Select Operating Scenario Preset",
            list(SIMULATOR_PRESETS.keys()),
            index=0
        )

    with col_pre2:
        load_engine_option = st.selectbox(
            "Or Load from Test Engine #",
            ["None (Use Preset)"] + [f"Test Engine #{i}" for i in range(1, 101)],
            index=0
        )

    # Determine baseline values for sliders
    if load_engine_option != "None (Use Preset)":
        e_id = int(load_engine_option.replace("Test Engine #", ""))
        e_sub = df_test[df_test["engine_id"] == e_id]
        last_snap = e_sub.iloc[-1]
        default_cycle = int(last_snap["cycle"])
        initial_sensors = {c: float(last_snap[c]) for c in e_sub.columns if c.startswith("sensor_") or c.startswith("op_")}
        st.info(f"Loaded actual sensor readings for **Test Engine #{e_id}** at its final observed cycle ({default_cycle}). Ground truth RUL: **{s_rul[e_id]} cycles**.")
    else:
        preset_data = SIMULATOR_PRESETS[preset_choice]
        default_cycle = preset_data["cycle"]
        initial_sensors = preset_data["values"]

    st.markdown("---")
    sim_col1, sim_col2 = st.columns([1, 1])

    # Left Column: Interactive Controls
    with sim_col1:
        st.markdown("### 🎛️ Operational Flight & Sensor Inputs")
        
        sim_cycle = st.slider(
            "Flight Operating Cycles (Age)",
            min_value=1,
            max_value=350,
            value=default_cycle,
            step=1,
            help="Total operational flight cycles completed by the engine."
        )

        st.markdown("#### 🌡️ Critical Thermodynamic Temperatures")
        s4_val = st.slider(
            "T50: Total Temperature at LPT Outlet (°R) [sensor_4]",
            min_value=1380.0,
            max_value=1445.0,
            value=float(np.clip(initial_sensors.get("sensor_4", 1402.7), 1380.0, 1445.0)),
            step=0.2,
            help="Exhaust gas temperature (EGT). Key indicator of thermal distress."
        )
        s3_val = st.slider(
            "T30: Total Temperature at HPC Outlet (°R) [sensor_3]",
            min_value=1570.0,
            max_value=1620.0,
            value=float(np.clip(initial_sensors.get("sensor_3", 1586.8), 1570.0, 1620.0)),
            step=0.5,
            help="Combustor inlet temperature reflecting compressor heating."
        )
        s2_val = st.slider(
            "T24: Total Temperature at LPC Outlet (°R) [sensor_2]",
            min_value=641.0,
            max_value=645.0,
            value=float(np.clip(initial_sensors.get("sensor_2", 642.4), 641.0, 645.0)),
            step=0.05,
            help="Low pressure compressor exit temperature."
        )

        st.markdown("#### 💨 Compressor Pressure Indicators")
        s11_val = st.slider(
            "Ps30: Static Pressure at HPC Outlet (psia) [sensor_11]",
            min_value=46.8,
            max_value=48.6,
            value=float(np.clip(initial_sensors.get("sensor_11", 47.35), 46.8, 48.6)),
            step=0.02,
            help="#1 SHAP Driver! Static pressure drop signals aerodynamic stall/fouling."
        )
        s7_val = st.slider(
            "P30: Total Pressure at HPC Outlet (psia) [sensor_7]",
            min_value=549.5,
            max_value=556.5,
            value=float(np.clip(initial_sensors.get("sensor_7", 553.95), 549.5, 556.5)),
            step=0.1,
            help="High pressure compressor discharge total pressure."
        )

        st.markdown("#### 🔄 Shaft Speeds & Flow Ratios")
        s9_val = st.slider(
            "Nc: Physical Core Speed (rpm) [sensor_9]",
            min_value=9020.0,
            max_value=9245.0,
            value=float(np.clip(initial_sensors.get("sensor_9", 9056.0), 9020.0, 9245.0)),
            step=1.0
        )
        s14_val = st.slider(
            "NRc: Corrected Core Speed (rpm) [sensor_14]",
            min_value=8100.0,
            max_value=8295.0,
            value=float(np.clip(initial_sensors.get("sensor_14", 8137.4), 8100.0, 8295.0)),
            step=0.5
        )

        with st.expander("🛠️ Advanced Auxiliary Sensors & Flow Channels"):
            s8_val = st.number_input("Nf: Fan Speed (rpm) [sensor_8]", value=float(initial_sensors.get("sensor_8", 2388.06)), step=0.05)
            s12_val = st.number_input("phi: Fuel Flow Ratio (pps/psia) [sensor_12]", value=float(initial_sensors.get("sensor_12", 521.91)), step=0.1)
            s15_val = st.number_input("BPR: Bypass Ratio [sensor_15]", value=float(initial_sensors.get("sensor_15", 8.42)), step=0.02)
            s17_val = st.number_input("htBleed: Bleed Enthalpy [sensor_17]", value=float(initial_sensors.get("sensor_17", 392.28)), step=0.5)
            s20_val = st.number_input("W31: HPT Coolant Bleed (lbm/s) [sensor_20]", value=float(initial_sensors.get("sensor_20", 38.93)), step=0.05)
            s21_val = st.number_input("W32: LPT Coolant Bleed (lbm/s) [sensor_21]", value=float(initial_sensors.get("sensor_21", 23.36)), step=0.05)

    # Compile input values
    current_inputs = initial_sensors.copy()
    current_inputs.update({
        "sensor_4": s4_val,
        "sensor_3": s3_val,
        "sensor_2": s2_val,
        "sensor_11": s11_val,
        "sensor_7": s7_val,
        "sensor_9": s9_val,
        "sensor_14": s14_val,
        "sensor_8": s8_val,
        "sensor_12": s12_val,
        "sensor_15": s15_val,
        "sensor_17": s17_val,
        "sensor_20": s20_val,
        "sensor_21": s21_val,
    })

    # Run Prediction across all 3 models
    sim_result = predictor.predict_simulator_inputs(current_inputs, cycle=sim_cycle)
    status_info = sim_result["status"]
    preds = sim_result["model_predictions"]

    # Right Column: Real-time Live Prediction Output
    with sim_col2:
        st.markdown("### 🎯 Live Simulation Results")

        # Master Prediction Card
        st.markdown(
            f"""
            <div class="aeroguard-kpi-card" style="border: 2px solid {status_info['color']}; padding: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">
                        Consensus Engine Health Status
                    </div>
                    <div>
                        {render_status_badge(status_info['category'])}
                    </div>
                </div>
                <div style="margin: 12px 0;">
                    <div style="font-size: 3.0rem; font-weight: 800; color: {status_info['color']}; font-family: 'JetBrains Mono', monospace; line-height: 1;">
                        {sim_result['primary_pred']:.1f} <span style="font-size: 1.3rem; font-weight: 500; color: #CBD5E1;">CYCLES</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                        Remaining Useful Life (RUL) Estimated by Active Model
                    </div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border-radius: 8px; padding: 10px 14px; margin-top: 10px;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Maintenance Protocol Directive</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-top: 2px;">
                        {status_info['description']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if status_info["is_hazard"]:
            st.markdown(render_hazard_alert(status_info["hazard_message"]), unsafe_allow_html=True)

        # 3 Models Comparison Breakdown
        st.markdown("#### 🔬 Predictions Across the 3 ML Models")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            xgb_val = preds.get("XGBoost", 0.0)
            st.markdown(render_kpi_card("XGBoost", f"{xgb_val:.1f}", "Gradient Boosting", "#10B981"), unsafe_allow_html=True)

        with m_col2:
            rf_val = preds.get("Random Forest", 0.0)
            st.markdown(render_kpi_card("Random Forest", f"{rf_val:.1f}", "Tree Ensemble", "#38BDF8"), unsafe_allow_html=True)

        with m_col3:
            lr_val = preds.get("Linear Regression", 0.0)
            st.markdown(render_kpi_card("Linear Reg", f"{lr_val:.1f}", "Parametric Line", "#A855F7"), unsafe_allow_html=True)

        # Degradation Life Consumed Bar
        st.markdown("#### ⏳ Engine Lifespan Degradation Bar")
        total_est_life = sim_cycle + sim_result["primary_pred"]
        pct_consumed = min(100.0, (sim_cycle / max(1.0, total_est_life)) * 100.0)
        
        st.progress(pct_consumed / 100.0)
        st.caption(f"Estimated **{pct_consumed:.1f}%** of total service life consumed ({sim_cycle} cycles elapsed / ~{total_est_life:.0f} total estimated cycles).")

        # Sensor Deviations from Healthy Baseline
        st.markdown("#### 🔍 Sensor Drift from Nominal Healthy Baseline")
        df_dev = pd.DataFrame(sim_result["deviations"])
        
        def format_delta(val):
            color = "#EF4444" if abs(val) > 1.0 else ("#F59E0B" if abs(val) > 0.3 else "#10B981")
            return f"<span style='color: {color}; font-weight: 600;'>{val:+.2f}</span>"

        st.dataframe(
            df_dev[["name", "desc", "current", "nominal", "delta", "percent_drift"]].rename(columns={
                "name": "Channel",
                "desc": "Description",
                "current": "Current Value",
                "nominal": "Healthy Baseline",
                "delta": "Delta (Observed - Baseline)",
                "percent_drift": "Drift %"
            }).style.format({
                "Current Value": "{:.2f}",
                "Healthy Baseline": "{:.2f}",
                "Delta (Observed - Baseline)": "{:+.2f}",
                "Drift %": "{:+.2f}%"
            }),
            use_container_width=True,
            height=240
        )

    # What-If Sensitivity Interactive Curve
    st.markdown("---")
    st.markdown("### 📈 Interactive What-If Sensitivity Analysis")
    st.write("Examine how RUL responds dynamically as a specific sensor channel varies across its physical range.")

    sens_col1, sens_col2 = st.columns([1, 2])
    with sens_col1:
        sens_channel = st.selectbox(
            "Select Sensor to Sweep",
            [
                ("sensor_4", "T50: Total Temperature at LPT Outlet (°R)"),
                ("sensor_11", "Ps30: Static Pressure at HPC Outlet (psia)"),
                ("sensor_3", "T30: Total Temperature at HPC Outlet (°R)"),
                ("sensor_9", "Nc: Physical Core Speed (rpm)")
            ],
            format_func=lambda x: x[1]
        )[0]

    with sens_col2:
        meta_ch = SENSOR_METADATA.get(sens_channel, {})
        h_val = NOMINAL_HEALTHY_BASELINES[sens_channel]
        f_val = FAILURE_ENDPOINTS[sens_channel]
        sweep_range = np.linspace(min(h_val, f_val) * 0.99, max(h_val, f_val) * 1.02, 25)

        sweep_xgb, sweep_rf, sweep_lr = [], [], []
        temp_in = current_inputs.copy()

        for val in sweep_range:
            temp_in[sens_channel] = val
            res_sweep = predictor.predict_simulator_inputs(temp_in, cycle=sim_cycle)
            sweep_xgb.append(res_sweep["model_predictions"].get("XGBoost", 0.0))
            sweep_rf.append(res_sweep["model_predictions"].get("Random Forest", 0.0))
            sweep_lr.append(res_sweep["model_predictions"].get("Linear Regression", 0.0))

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_range, y=sweep_xgb, name="XGBoost", line=dict(color="#10B981", width=2.5)))
        fig_sweep.add_trace(go.Scatter(x=sweep_range, y=sweep_rf, name="Random Forest", line=dict(color="#38BDF8", width=2.5, dash="dash")))
        fig_sweep.add_trace(go.Scatter(x=sweep_range, y=sweep_lr, name="Linear Regression", line=dict(color="#A855F7", width=2, dash="dot")))
        
        # Mark current slider point
        current_val = current_inputs[sens_channel]
        fig_sweep.add_vline(x=current_val, line_dash="solid", line_color="#EF4444", annotation_text="Current Input", annotation_font_color="#EF4444")
        
        fig_sweep.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0A101D",
            font=dict(family="Inter, sans-serif", color="#CBD5E1", size=11),
            xaxis_title=f"{meta_ch.get('name', sens_channel)} ({meta_ch.get('unit', '')})",
            yaxis_title="Predicted RUL (Cycles)",
            margin=dict(l=40, r=20, t=30, b=30)
        )
        st.plotly_chart(fig_sweep, use_container_width=True)


# TAB 2: FLEET HEALTH OVERVIEW
elif active_tab == "📊 Fleet Health Overview":
    st.subheader("NASA C-MAPSS Test Fleet Overview (100 Turbofan Engines)")
    
    # Predict all test engines at final cycle
    preds_all = []
    for eid in sorted(df_test["engine_id"].unique()):
        sub = df_test[df_test["engine_id"] == eid]
        res = predictor.predict_single_engine_current_state(sub)
        t_rul = s_rul.get(eid, np.nan)
        preds_all.append({
            "engine_id": eid,
            "cycle": res["current_cycle"],
            "predicted_rul": res["predicted_rul"],
            "true_rul": t_rul,
            "error": round(res["predicted_rul"] - t_rul, 1) if not np.isnan(t_rul) else 0.0,
            "status": res["status"]["category"]
        })
    df_fleet = pd.DataFrame(preds_all)

    # High-level KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_kpi_card("Fleet Count", f"{len(df_fleet)}", "Turbofan Engines", "#F8FAFC"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("Average RUL", f"{df_fleet['predicted_rul'].mean():.1f}", "Remaining Cycles", "#38BDF8"), unsafe_allow_html=True)
    with k3:
        n_crit = (df_fleet["status"] == "CRITICAL").sum()
        st.markdown(render_kpi_card("Critical Engines", f"{n_crit}", "RUL ≤ 10 cycles", "#EF4444"), unsafe_allow_html=True)
    with k4:
        n_warn = (df_fleet["status"] == "WARNING").sum()
        st.markdown(render_kpi_card("Warning Engines", f"{n_warn}", "RUL 11-30 cycles", "#F97316"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Test Fleet Inventory Table")
    f_stat = st.multiselect("Filter by Status", ["CRITICAL", "WARNING", "ATTENTION", "HEALTHY"], default=["CRITICAL", "WARNING", "ATTENTION", "HEALTHY"])
    df_filtered = df_fleet[df_fleet["status"].isin(f_stat)]
    
    st.dataframe(
        df_filtered.rename(columns={
            "engine_id": "Engine ID",
            "cycle": "Current Cycle",
            "predicted_rul": "Predicted RUL",
            "true_rul": "Ground Truth RUL",
            "error": "Residual Error",
            "status": "Maintenance Status"
        }).style.format({
            "Predicted RUL": "{:.1f}",
            "Ground Truth RUL": "{:.0f}",
            "Residual Error": "{:+.1f}"
        }),
        use_container_width=True
    )


# TAB 3: 3-MODEL BENCHMARKS
elif active_tab == "📈 3-Model Benchmarks":
    st.subheader("Model Benchmarks: XGBoost vs. Random Forest vs. Linear Regression")
    st.write(
        "Empirical performance metrics evaluated on the official NASA C-MAPSS test set ground truth."
    )

    comp_file = REPORTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        
        st.dataframe(
            df_comp.style.format({
                "test_rmse": "{:.3f}",
                "test_mae": "{:.3f}",
                "test_r2": "{:.4f}",
                "test_nasa_score": "{:.2f}",
                "test_overestimate_pct": "{:.1f}%",
                "test_critical_overestimates": "{:d}"
            }),
            use_container_width=True
        )

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_bar = px.bar(
                df_comp,
                x="model_name",
                y="test_rmse",
                color="model_name",
                color_discrete_sequence=["#10B981", "#38BDF8", "#A855F7"],
                title="Root Mean Squared Error (RMSE — Lower is Better)"
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A101D", font=dict(color="#CBD5E1"))
            st.plotly_chart(fig_bar, use_container_width=True)

        with b_col2:
            fig_score = px.bar(
                df_comp,
                x="model_name",
                y="test_nasa_score",
                color="model_name",
                color_discrete_sequence=["#10B981", "#38BDF8", "#A855F7"],
                title="NASA Asymmetric Safety Penalty (Lower is Better)"
            )
            fig_score.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A101D", font=dict(color="#CBD5E1"))
            st.plotly_chart(fig_score, use_container_width=True)


# TAB 4: CUSTOM TELEMETRY UPLOAD
elif active_tab == "📁 Custom Telemetry Upload":
    st.subheader("Upload Custom Engine Flight Telemetry CSV")
    st.write("Upload a flight telemetry record CSV. AeroGuard will validate the schema, extract sliding windows, and run predictions.")

    sample_csv = df_test[df_test["engine_id"] == 1].to_csv(index=False)
    st.download_button("📥 Download Sample Telemetry CSV (Engine #1)", data=sample_csv, file_name="engine_1_telemetry.csv", mime="text/csv")

    up_file = st.file_uploader("Choose CSV file", type=["csv"])
    if up_file is not None:
        try:
            df_up = pd.read_csv(up_file)
            valid, msg, df_clean = validate_uploaded_csv(df_up)
            if not valid:
                st.error(f"Validation Error: {msg}")
            else:
                st.success(f"File Validated: {msg}")
                diag = predictor.predict_single_engine_current_state(df_clean)
                st.markdown(render_kpi_card("Predicted RUL", f"{diag['predicted_rul']:.1f} cycles", f"Engine #{diag['engine_id']} at cycle {diag['current_cycle']}"), unsafe_allow_html=True)
                st.markdown(f"Status: **{diag['status']['category']}** — {diag['status']['description']}")
        except Exception as err:
            st.error(f"Error processing CSV: {err}")


# TAB 5: SYSTEM DOCUMENTATION
elif active_tab == "📖 System Documentation":
    st.subheader("AeroGuard Architecture & Predictive Maintenance Directives")
    st.markdown(
        r"""
        ### 1. The Core ML Models
        - **XGBoost Regressor:** Gradient-boosted decision trees. Optimal for non-linear multivariate interactions with low variance.
        - **Random Forest Regressor:** Bagged ensemble of decision trees. Robust to local sensor noise.
        - **Linear Regression:** Standard benchmark establishing baseline linear trend correlations.

        ### 2. Maintenance Action Thresholds
        - 🟢 **HEALTHY ($RUL > 75$ cycles):** Nominal flight operation.
        - 🟡 **ATTENTION ($31 \le RUL \le 75$ cycles):** Schedule routine depot inspection.
        - 🟠 **WARNING ($11 \le RUL \le 30$ cycles):** Component wear accelerated; plan module overhaul.
        - 🔴 **CRITICAL ($RUL \le 10$ cycles):** Immediate maintenance mandatory; ground engine immediately.

        ### 3. Asymmetric NASA Scoring
        $$d_i = \text{Predicted RUL} - \text{Actual RUL}$$
        - **$d < 0$ (Early Warning / Underestimate):** $s_i = \exp(-d_i / 13) - 1$
        - **$d \ge 0$ (Delayed Maintenance / Overestimate):** $s_i = \exp(d_i / 10) - 1$ (severe exponential penalty!)
        """
    )
