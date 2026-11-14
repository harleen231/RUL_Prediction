"""
AeroGuard Interactive Plotly Visualizations Module
Creates aerospace-styled dark telemetry charts, correlation matrices,
residual error distributions, and SHAP explainability diagrams.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import SENSOR_METADATA

DARK_THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0A101D",
    font=dict(family="Inter, sans-serif", color="#CBD5E1", size=11),
    margin=dict(l=40, r=30, t=50, b=40),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.3)"
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.3)"
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.7)",
        bordercolor="rgba(56, 189, 248, 0.2)",
        borderwidth=1
    )
)


def create_fleet_status_donut(status_counts: Dict[str, int]) -> go.Figure:
    """Creates a sleek donut chart of fleet engine health distribution."""
    colors = {
        "HEALTHY": "#10B981",
        "ATTENTION": "#F59E0B",
        "WARNING": "#F97316",
        "CRITICAL": "#EF4444"
    }
    labels = list(status_counts.keys())
    values = list(status_counts.values())
    pie_colors = [colors.get(l, "#38BDF8") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=pie_colors, line=dict(color="#070B14", width=2)),
        textinfo="label+value",
        hoverinfo="label+value+percent",
        textfont=dict(color="#FFFFFF", size=12, family="Inter")
    )])

    fig.update_layout(
        title=dict(text="Fleet Health Distribution Status", font=dict(size=14, color="#F8FAFC")),
        showlegend=False,
        **DARK_THEME_LAYOUT
    )
    return fig


def create_sensor_trajectory_chart(
    df_engine: pd.DataFrame,
    sensors: List[str]
) -> go.Figure:
    """Visualizes multi-sensor telemetry evolution over engine operating cycles."""
    fig = go.Figure()

    palette = ["#38BDF8", "#F59E0B", "#10B981", "#EC4899", "#8B5CF6", "#06B6D4"]

    for i, s in enumerate(sensors):
        meta = SENSOR_METADATA.get(s, {})
        display_name = f"{meta.get('name', s)} ({meta.get('desc', s)[:20]})"
        color = palette[i % len(palette)]
        
        fig.add_trace(go.Scatter(
            x=df_engine["cycle"],
            y=df_engine[s],
            mode="lines+markers",
            name=display_name,
            line=dict(color=color, width=2.2),
            marker=dict(size=4),
            hovertemplate=f"Cycle: %{{x}}<br>{s}: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Multivariate Sensor Degradation Telemetry", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="Operational Flight Cycles",
        yaxis_title="Normalized Sensor Telemetry",
        **DARK_THEME_LAYOUT
    )
    return fig


def create_rul_trajectory_chart(
    cycles: np.ndarray,
    predicted_rul: np.ndarray,
    actual_rul: Optional[np.ndarray] = None
) -> go.Figure:
    """Plots RUL countdown trajectory comparing model estimate vs actual ground truth."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cycles,
        y=predicted_rul,
        mode="lines",
        name="Predicted RUL",
        line=dict(color="#38BDF8", width=3),
        hovertemplate="Cycle: %{x}<br>Predicted RUL: %{y:.1f} cycles<extra></extra>"
    ))

    if actual_rul is not None:
        fig.add_trace(go.Scatter(
            x=cycles,
            y=actual_rul,
            mode="lines",
            name="Ground Truth RUL",
            line=dict(color="#10B981", width=2, dash="dash"),
            hovertemplate="Cycle: %{x}<br>Actual RUL: %{y:.1f} cycles<extra></extra>"
        ))

    # Add danger threshold line (Critical: RUL <= 10)
    fig.add_hline(y=10, line_dash="dot", line_color="#EF4444", annotation_text="Critical Limit (10 cycles)", annotation_font_color="#EF4444")
    fig.add_hline(y=30, line_dash="dot", line_color="#F97316", annotation_text="Warning Limit (30 cycles)", annotation_font_color="#F97316")

    fig.update_layout(
        title=dict(text="Remaining Useful Life (RUL) Trajectory & Safety Boundaries", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="Operational Flight Cycles",
        yaxis_title="RUL (Remaining Cycles)",
        **DARK_THEME_LAYOUT
    )
    return fig


def create_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: Optional[np.ndarray] = None
) -> go.Figure:
    """Creates interactive Actual vs. Predicted scatter plot with overestimation zone highlight."""
    errors = y_pred - y_true
    max_val = max(np.max(y_true), np.max(y_pred)) + 15

    # Color scale based on error
    fig = go.Figure()

    # Ideal 1:1 Line
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Ideal 1:1 Parity",
        line=dict(color="#EF4444", width=2, dash="dash"),
        hoverinfo="none"
    ))

    # Overestimation zone shading (dangerous area)
    fig.add_trace(go.Scatter(
        x=[0, max_val, 0],
        y=[0, max_val, max_val],
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Overestimation Zone (Risk)",
        hoverinfo="none"
    ))

    # Engine points
    hover_texts = []
    for i in range(len(y_true)):
        eid_str = f"Engine #{engine_ids[i]}<br>" if engine_ids is not None else ""
        hover_texts.append(
            f"{eid_str}Actual: {y_true[i]:.1f} cycles<br>Predicted: {y_pred[i]:.1f} cycles<br>Error: {errors[i]:+.1f} cycles"
        )

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Test Engines",
        marker=dict(
            size=8,
            color=errors,
            colorscale=[[0, "#38BDF8"], [0.5, "#E2E8F0"], [1.0, "#F59E0B"]],
            showscale=True,
            colorbar=dict(title=dict(text="Error (Cycles)", font=dict(color="#CBD5E1")), tickfont=dict(color="#CBD5E1")),
            line=dict(color="#0C1222", width=1)
        ),
        text=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title=dict(text="Actual vs. Predicted RUL (Benchmark Evaluation)", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="Actual Ground Truth RUL (Cycles)",
        yaxis_title="Predicted RUL (Cycles)",
        **DARK_THEME_LAYOUT
    )
    return fig


def create_error_distribution_chart(errors: np.ndarray) -> go.Figure:
    """Plots interactive residual error histogram with risk thresholds."""
    fig = px.histogram(
        x=errors,
        nbins=30,
        color_discrete_sequence=["#0284C7"]
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#EF4444", annotation_text="Zero Error", annotation_font_color="#EF4444")
    fig.add_vline(x=np.mean(errors), line_dash="dot", line_color="#10B981", annotation_text=f"Mean Error: {np.mean(errors):.2f}", annotation_font_color="#10B981")

    fig.update_layout(
        title=dict(text="Prediction Error Residual Distribution (Predicted - Actual)", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="Prediction Error (Cycles)",
        yaxis_title="Engine Count",
        **DARK_THEME_LAYOUT
    )
    return fig


def create_model_comparison_chart(df_comp: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing RMSE, MAE, and NASA Score across models."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_comp["model_name"],
        y=df_comp["test_rmse"],
        name="RMSE (Cycles)",
        marker_color="#38BDF8"
    ))
    fig.add_trace(go.Bar(
        x=df_comp["model_name"],
        y=df_comp["test_mae"],
        name="MAE (Cycles)",
        marker_color="#818CF8"
    ))

    fig.update_layout(
        title=dict(text="Model Accuracy Benchmark (RMSE & MAE — Lower is Better)", font=dict(size=14, color="#F8FAFC")),
        barmode="group",
        xaxis_title="Model Architecture",
        yaxis_title="Error in Operational Cycles",
        **DARK_THEME_LAYOUT
    )
    return fig


def create_shap_sensor_importance_chart(df_sensor_imp: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Creates horizontal bar chart of physical turbofan sensor SHAP importance."""
    df_sub = df_sensor_imp.head(top_n).sort_values("importance", ascending=True)

    labels = [f"{r['name']} ({r['sensor']}): {r['description']}" for _, r in df_sub.iterrows()]

    fig = go.Figure(go.Bar(
        x=df_sub["importance"],
        y=labels,
        orientation='h',
        marker=dict(
            color=df_sub["importance"],
            colorscale="Viridis",
            line=dict(color="#0F172A", width=1)
        ),
        hovertemplate="Impact: %{x:.2f} cycles<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Global Degradation Impact by Physical Sensor (Mean |SHAP|)", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="Mean Absolute SHAP Value (Impact on RUL in Cycles)",
        yaxis_title="Turbofan Sensor Channel",
        **DARK_THEME_LAYOUT
    )
    return fig


def create_local_shap_waterfall(local_drivers: List[dict], base_value: float) -> go.Figure:
    """Plots top positive/negative feature contributions for a specific prediction."""
    features = [d["feature"] for d in local_drivers][::-1]
    impacts = [d["shap_impact"] for d in local_drivers][::-1]
    colors = ["#10B981" if x > 0 else "#EF4444" for x in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
        hovertemplate="Feature: %{y}<br>SHAP Contribution: %{x:+.2f} cycles<extra></extra>"
    ))

    fig.add_vline(x=0, line_dash="solid", line_color="#64748B")

    fig.update_layout(
        title=dict(text=f"Local Sensor Contributions (Baseline Fleet Expectation: {base_value:.1f} Cycles)", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="SHAP Contribution to Predicted RUL (+ Healthy / - Degradation)",
        yaxis_title="Feature Channel",
        **DARK_THEME_LAYOUT
    )
    return fig
