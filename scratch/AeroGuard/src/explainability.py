"""
AeroGuard Explainable AI Module
Provides global and local model interpretability using SHAP (SHapley Additive exPlanations)
and physics-informed sensor attribution to decode degradation mechanisms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PLOTS_DIR, SENSOR_METADATA


class AeroGuardExplainer:
    """
    Manages SHAP explanations for tree-based, linear, and ensemble regression models,
    aggregating engineered temporal features into physical turbofan sensor contributions.
    """

    def __init__(self, model, background_data: pd.DataFrame):
        self.model = model
        self.feature_names = list(background_data.columns)

        # Use a background sample (e.g., 100 rows) for fast, stable expectation calculation
        bg_sample = shap.sample(background_data, min(100, len(background_data)), random_state=42)
        self.background_sample = bg_sample

        try:
            # TreeExplainer is fast and exact for tree models
            if hasattr(model, "estimators_") or "xgb" in str(type(model)).lower():
                self.explainer = shap.TreeExplainer(model, data=bg_sample)
            else:
                self.explainer = shap.Explainer(model.predict, bg_sample)
        except Exception:
            self.explainer = shap.Explainer(model.predict, bg_sample)

    def explain_global(self, X_sample: pd.DataFrame, max_display: int = 15) -> Dict[str, Union[pd.DataFrame, float]]:
        """
        Computes global feature importances via mean absolute SHAP values across a representative sample.
        """
        sample = X_sample.sample(min(200, len(X_sample)), random_state=42) if len(X_sample) > 200 else X_sample
        shap_values = self.explainer(sample)

        # Mean absolute SHAP per feature
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        # Aggregate back to physical sensor level
        sensor_agg = {}
        for _, row in df_importance.iterrows():
            feat = row["feature"]
            val = row["mean_abs_shap"]
            # Extract root sensor prefix (e.g. sensor_11)
            parts = feat.split("_")
            if len(parts) >= 2 and parts[0] == "sensor":
                root = f"{parts[0]}_{parts[1]}"
            elif parts[0] == "op":
                root = f"{parts[0]}_{parts[1]}_{parts[2]}"
            else:
                root = feat

            sensor_agg[root] = sensor_agg.get(root, 0.0) + val

        df_sensor_importance = pd.DataFrame([
            {
                "sensor": k,
                "importance": v,
                "name": SENSOR_METADATA.get(k, {}).get("name", k),
                "description": SENSOR_METADATA.get(k, {}).get("desc", ""),
                "unit": SENSOR_METADATA.get(k, {}).get("unit", "")
            }
            for k, v in sensor_agg.items()
        ]).sort_values("importance", ascending=False).reset_index(drop=True)

        return {
            "feature_importance": df_importance,
            "sensor_importance": df_sensor_importance,
            "base_value": float(np.mean(shap_values.base_values))
        }

    def explain_local_instance(
        self,
        instance_series: pd.Series,
        top_k: int = 10
    ) -> Dict[str, Union[float, List[dict]]]:
        """
        Generates local explanation for a single operational snapshot:
        shows which features pushed predicted RUL higher or lower.
        """
        X_inst = pd.DataFrame([instance_series[self.feature_names]])
        shap_vals = self.explainer(X_inst)

        base_val = float(shap_vals.base_values[0])
        contributions = shap_vals.values[0]
        actual_vals = instance_series[self.feature_names].values

        features_info = []
        for feat, val, shap_v in zip(self.feature_names, actual_vals, contributions):
            parts = feat.split("_")
            sensor_root = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 and parts[0] == "sensor" else feat
            meta = SENSOR_METADATA.get(sensor_root, {})

            features_info.append({
                "feature": feat,
                "sensor_name": meta.get("name", feat),
                "sensor_desc": meta.get("desc", ""),
                "feature_value": round(float(val), 3),
                "shap_impact": round(float(shap_v), 3),
                "direction": "Pushes RUL Higher (Healthy)" if shap_v > 0 else "Accelerates Degradation (Shortens RUL)"
            })

        # Sort by absolute impact
        features_info = sorted(features_info, key=lambda x: abs(x["shap_impact"]), reverse=True)

        return {
            "base_value": round(base_val, 2),
            "predicted_offset": round(float(np.sum(contributions)), 2),
            "final_predicted_rul": round(base_val + float(np.sum(contributions)), 2),
            "top_drivers": features_info[:top_k]
        }


def plot_global_shap_bar(
    df_sensor_importance: pd.DataFrame,
    save_path: Optional[Path] = None,
    top_n: int = 12
) -> plt.Figure:
    """Generates bar plot of aggregated physical sensor importances."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    df_sub = df_sensor_importance.head(top_n).sort_values("importance", ascending=True)

    labels = [f"{r['name']} ({r['sensor']}): {r['description'][:24]}" for _, r in df_sub.iterrows()]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sub)))

    ax.barh(labels, df_sub["importance"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean Absolute SHAP Impact on RUL (Cycles)", fontsize=11, fontweight="bold")
    ax.set_title("AeroGuard: Global Sensor Degradation Drivers", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
