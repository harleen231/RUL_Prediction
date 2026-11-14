"""
AeroGuard Model Evaluation Module
Calculates standard regression metrics (MAE, RMSE, R2) and the official
NASA asymmetric scoring function, with specialized diagnostics for
safety-critical early-failure and overestimation errors.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PLOTS_DIR, REPORTS_DIR, OVERESTIMATION_HAZARD_THRESHOLD, MAINTENANCE_LEVELS


def nasa_scoring_function(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Computes the NASA C-MAPSS PHM 2008 asymmetric prognostic scoring function.
    
    Formula:
      d_i = y_pred_i - y_true_i
      s_i = exp(-d_i / 13) - 1   if d_i < 0  (underestimation: early warning, minor penalty)
      s_i = exp(d_i / 10) - 1    if d_i >= 0 (overestimation: delayed maintenance, severe penalty)
      Total Score S = sum(s_i)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    d = y_pred - y_true
    
    # Asymmetric exponential penalty
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str = "Model"
) -> Dict[str, Union[float, int, str]]:
    """
    Calculates comprehensive performance metrics for RUL regression predictions.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    errors = y_p - y_t  # Positive = Overestimation, Negative = Underestimation

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))
    nasa_score = nasa_scoring_function(y_t, y_p)

    overestimates = errors[errors > 0]
    underestimates = errors[errors < 0]

    # Critical early failure overestimation: Actual RUL <= 30 but model overestimates by >= 10 cycles
    critical_mask = y_t <= 30
    critical_overestimates = (errors[critical_mask] > OVERESTIMATION_HAZARD_THRESHOLD).sum()

    return {
        "model_name": model_name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "nasa_score": round(nasa_score, 2),
        "num_samples": len(y_t),
        "overestimation_pct": round(float(len(overestimates) / len(y_t) * 100), 2),
        "underestimation_pct": round(float(len(underestimates) / len(y_t) * 100), 2),
        "mean_overestimate_error": round(float(np.mean(overestimates)), 3) if len(overestimates) > 0 else 0.0,
        "max_overestimate_error": round(float(np.max(overestimates)), 3) if len(overestimates) > 0 else 0.0,
        "mean_underestimate_error": round(float(np.mean(underestimates)), 3) if len(underestimates) > 0 else 0.0,
        "critical_overestimate_count": int(critical_overestimates),
    }


def analyze_errors_by_rul_range(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> pd.DataFrame:
    """
    Groups prediction errors by operational health categories:
      - Critical (0-10 cycles)
      - Warning (11-30 cycles)
      - Attention (31-75 cycles)
      - Healthy (> 75 cycles)
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    errors = y_p - y_t

    bins = [-np.inf, 10, 30, 75, np.inf]
    labels = ["Critical (0-10)", "Warning (11-30)", "Attention (31-75)", "Healthy (>75)"]
    cat_series = pd.cut(y_t, bins=bins, labels=labels)

    df_err = pd.DataFrame({
        "actual_rul": y_t,
        "predicted_rul": y_p,
        "error": errors,
        "abs_error": np.abs(errors),
        "category": cat_series
    })

    grouped = df_err.groupby("category", observed=False).agg(
        count=("error", "count"),
        mae=("abs_error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else 0.0),
        mean_error=("error", "mean"),
        overestimates=("error", lambda x: (x > 0).sum()),
        underestimates=("error", lambda x: (x < 0).sum()),
        hazardous_overestimates=("error", lambda x: (x > OVERESTIMATION_HAZARD_THRESHOLD).sum())
    ).reset_index()

    return grouped


# Visual Reporting Functions
def plot_actual_vs_predicted(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Generates an Actual vs Predicted RUL scatter diagram with ideal unity line."""
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    max_val = max(np.max(y_t), np.max(y_p)) + 15

    # Scatter points colored by error direction
    errors = y_p - y_t
    colors = np.where(errors > 0, "#F59E0B", "#38BDF8")  # Orange = Overestimate, Cyan = Underestimate

    ax.scatter(y_t, y_p, c=colors, alpha=0.7, edgecolors="none", s=45, label="Test Predictions")
    ax.plot([0, max_val], [0, max_val], color="#EF4444", linestyle="--", linewidth=2, label="Ideal 1:1 Parity")
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], color="#F59E0B", alpha=0.08, label="Overestimation Zone (Risk)")
    ax.fill_between([0, max_val], [0, 0], [0, max_val], color="#38BDF8", alpha=0.08, label="Underestimation Zone (Conservative)")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Ground Truth Remaining Useful Life (Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Remaining Useful Life (Cycles)", fontsize=11, fontweight="bold")
    ax.set_title(f"AeroGuard: Actual vs. Predicted RUL — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_error_distribution(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    model_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plots histogram and KDE of prediction errors (Predicted - Actual)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    errors = np.asarray(y_pred) - np.asarray(y_true)

    sns.histplot(errors, kde=True, ax=ax, color="#0284C7", bins=25, edgecolor="black", alpha=0.6)
    ax.axvline(0, color="#EF4444", linestyle="--", linewidth=2, label="Zero Error Reference")
    ax.axvline(np.mean(errors), color="#10B981", linestyle="-", linewidth=1.8, label=f"Mean Error: {np.mean(errors):.2f}")

    ax.set_xlabel("Prediction Error (Predicted RUL - Actual RUL in Cycles)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency Count", fontsize=11, fontweight="bold")
    ax.set_title(f"Prediction Error Residual Distribution — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Generates comparison bar charts across models for MAE, RMSE, and NASA Score."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    models = comparison_df["model_name"]
    palette = ["#0284C7", "#0EA5E9", "#38BDF8", "#06B6D4", "#10B981"][:len(models)]

    mae_col = "test_mae" if "test_mae" in comparison_df.columns else "mae"
    rmse_col = "test_rmse" if "test_rmse" in comparison_df.columns else "rmse"
    score_col = "test_nasa_score" if "test_nasa_score" in comparison_df.columns else "nasa_score"

    # MAE
    axes[0].bar(models, comparison_df[mae_col], color=palette, edgecolor="black", alpha=0.85)
    axes[0].set_title("Mean Absolute Error (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("MAE (Cycles)")
    axes[0].tick_params(axis='x', rotation=30)

    # RMSE
    axes[1].bar(models, comparison_df[rmse_col], color=palette, edgecolor="black", alpha=0.85)
    axes[1].set_title("Root Mean Squared Error (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("RMSE (Cycles)")
    axes[1].tick_params(axis='x', rotation=30)

    # NASA Score
    axes[2].bar(models, comparison_df[score_col], color=palette, edgecolor="black", alpha=0.85)
    axes[2].set_title("NASA Asymmetric PHM Score (Lower is Better)", fontweight="bold")
    axes[2].set_ylabel("NASA Score")
    axes[2].tick_params(axis='x', rotation=30)

    plt.suptitle("AeroGuard: Multi-Model Performance Benchmark", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
