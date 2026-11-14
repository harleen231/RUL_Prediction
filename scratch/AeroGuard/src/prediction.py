"""
AeroGuard Prediction & Predictive Maintenance Warning Module
Executes end-to-end inference on continuous telemetry streams, uploaded CSVs,
and the interactive real-time Engine Degradation Simulator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import json
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR,
    DEFAULT_WINDOW_SIZE,
    MAINTENANCE_LEVELS,
    OVERESTIMATION_HAZARD_THRESHOLD,
    SETTING_COLS,
    SENSOR_COLS,
    INDEX_COLS,
    ALL_RAW_COLS,
    SENSOR_METADATA
)
from src.preprocessing import CMAPSSPreprocessor
from src.feature_engineering import extract_sliding_window_features

# Empirical nominal healthy baselines (observed in initial cycles 1-15 of C-MAPSS FD001)
NOMINAL_HEALTHY_BASELINES = {
    "op_setting_1": 0.0000,
    "op_setting_2": 0.0000,
    "op_setting_3": 100.0,
    "sensor_1": 518.67,
    "sensor_2": 642.38,
    "sensor_3": 1586.81,
    "sensor_4": 1402.74,
    "sensor_5": 14.62,
    "sensor_6": 21.61,
    "sensor_7": 553.95,
    "sensor_8": 2388.06,
    "sensor_9": 9056.04,
    "sensor_10": 1.30,
    "sensor_11": 47.35,
    "sensor_12": 521.91,
    "sensor_13": 2388.06,
    "sensor_14": 8137.37,
    "sensor_15": 8.42,
    "sensor_16": 0.03,
    "sensor_17": 392.28,
    "sensor_18": 2388.0,
    "sensor_19": 100.0,
    "sensor_20": 38.93,
    "sensor_21": 23.36,
}

# Empirical sensor failure endpoints (cycles within 10 of failure)
FAILURE_ENDPOINTS = {
    "op_setting_1": 0.0000,
    "op_setting_2": 0.0000,
    "op_setting_3": 100.0,
    "sensor_1": 518.67,
    "sensor_2": 643.58,
    "sensor_3": 1601.28,
    "sensor_4": 1427.10,
    "sensor_5": 14.62,
    "sensor_6": 21.61,
    "sensor_7": 551.65,
    "sensor_8": 2388.22,
    "sensor_9": 9095.01,
    "sensor_10": 1.30,
    "sensor_11": 48.09,
    "sensor_12": 519.95,
    "sensor_13": 2388.22,
    "sensor_14": 8164.79,
    "sensor_15": 8.51,
    "sensor_16": 0.03,
    "sensor_17": 396.02,
    "sensor_18": 2388.0,
    "sensor_19": 100.0,
    "sensor_20": 38.48,
    "sensor_21": 23.09,
}

# Simulator presets for one-click testing
SIMULATOR_PRESETS = {
    "🟢 Brand New Engine (Healthy Initial Flight)": {
        "cycle": 15,
        "values": NOMINAL_HEALTHY_BASELINES.copy()
    },
    "🟡 Mid-Life Cruise Wear (Nominal Operational Drift)": {
        "cycle": 110,
        "values": {
            k: NOMINAL_HEALTHY_BASELINES[k] * 0.6 + FAILURE_ENDPOINTS[k] * 0.4
            for k in NOMINAL_HEALTHY_BASELINES
        }
    },
    "🟠 Accelerated Thermal Fatigue (High Vibration & EGT)": {
        "cycle": 175,
        "values": {
            k: NOMINAL_HEALTHY_BASELINES[k] * 0.25 + FAILURE_ENDPOINTS[k] * 0.75
            for k in NOMINAL_HEALTHY_BASELINES
        }
    },
    "🔴 Critical Near-Failure State (Grounding Mandatory)": {
        "cycle": 230,
        "values": FAILURE_ENDPOINTS.copy()
    }
}


def classify_maintenance_status(predicted_rul: float, sensor_roc_warning: bool = False) -> Dict[str, Union[str, float]]:
    """
    Translates numeric predicted RUL into standardized aeronautical maintenance urgency levels.
    Levels:
      - Healthy: RUL > 75
      - Attention: 31 <= RUL <= 75
      - Warning: 11 <= RUL <= 30
      - Critical: RUL <= 10
    """
    p_rul = max(0.0, float(predicted_rul))

    if p_rul > 75:
        category = "HEALTHY"
    elif p_rul >= 31:
        category = "ATTENTION"
    elif p_rul >= 11:
        category = "WARNING"
    else:
        category = "CRITICAL"

    cfg = MAINTENANCE_LEVELS[category]

    hazard_flag = False
    hazard_msg = "Nominal risk profile."
    if sensor_roc_warning and category in ["HEALTHY", "ATTENTION"]:
        hazard_flag = True
        hazard_msg = "CAUTION: Rapid sensor degradation detected. RUL may be overestimated; prioritize inspection."

    return {
        "category": category,
        "badge": cfg["badge"],
        "color": cfg["color"],
        "description": cfg["desc"],
        "predicted_rul": round(p_rul, 1),
        "is_hazard": hazard_flag,
        "hazard_message": hazard_msg,
        "disclaimer": "Decision-support output. Not certified for flight clearance."
    }


class AeroGuardPredictor:
    """
    High-level predictor bundling the trained core models:
    - XGBoost
    - Random Forest
    - Linear Regression
    Supports single-engine historical inference, CSV batch validation, and real-time simulator predictions.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        models_dir: Optional[Path] = None,
        window_size: int = DEFAULT_WINDOW_SIZE
    ):
        self.models_dir = models_dir or MODELS_DIR
        self.window_size = window_size
        self.preprocessor = CMAPSSPreprocessor.load(self.models_dir / "preprocessor.joblib")

        # Load metadata
        meta_file = self.models_dir / "model_metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Model metadata not found at {meta_file}. Please train models first.")
        with open(meta_file, "r") as f:
            self.metadata = json.load(f)

        self.model_name = model_name or self.metadata.get("best_model_name", "XGBoost")
        self.feature_names = self.metadata["feature_names"]

        # Load core models
        self.models = {}
        for key, fname in [("XGBoost", "xgboost.joblib"), ("Random Forest", "random_forest.joblib"), ("Linear Regression", "linear_regression.joblib")]:
            p = self.models_dir / fname
            if p.exists():
                self.models[key] = joblib.load(p)

        # Active model reference
        slug = self.model_name.lower().replace(" ", "_")
        active_path = self.models_dir / f"{slug}.joblib"
        if not active_path.exists():
            active_path = self.models_dir / "best_model.joblib"
        self.model = joblib.load(active_path)

    def predict_engine_trajectory(self, df_engine: pd.DataFrame) -> pd.DataFrame:
        """Takes raw multi-cycle telemetry for an engine and generates RUL predictions across each cycle."""
        df_scaled = self.preprocessor.transform(df_engine)
        df_feat = extract_sliding_window_features(
            df_scaled,
            sensor_cols=self.preprocessor.active_sensor_cols_,
            window_size=self.window_size
        )

        X = df_feat[self.feature_names]
        preds = self.model.predict(X)

        df_out = df_engine.copy()
        df_out["predicted_rul"] = np.clip(preds, 0, None)
        return df_out

    def predict_single_engine_current_state(self, df_engine_history: pd.DataFrame) -> Dict[str, Any]:
        """Predicts RUL for the most recent operational snapshot of an engine based on its history."""
        df_traj = self.predict_engine_trajectory(df_engine_history)
        last_row = df_traj.iloc[-1]
        last_cycle = int(last_row["cycle"])
        engine_id = int(last_row["engine_id"])
        pred_rul = float(last_row["predicted_rul"])

        active_sensors = [s for s in ["sensor_9", "sensor_11", "sensor_4", "sensor_14"] if s in df_engine_history.columns]
        sensor_roc_warning = False
        if len(df_engine_history) >= 5 and active_sensors:
            recent_delta = df_engine_history[active_sensors].iloc[-1] - df_engine_history[active_sensors].iloc[-5]
            if (np.abs(recent_delta) > 2.0).any():
                sensor_roc_warning = True

        status = classify_maintenance_status(pred_rul, sensor_roc_warning)

        return {
            "engine_id": engine_id,
            "current_cycle": last_cycle,
            "predicted_rul": round(pred_rul, 1),
            "status": status,
            "model_used": self.model_name
        }

    def predict_simulator_inputs(
        self,
        sensor_inputs: Dict[str, float],
        cycle: int = 100,
        base_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Takes raw simulator input values and current operational cycle.
        Synthesizes a causal temporal trajectory, transforms it through the C-MAPSS preprocessor,
        and generates RUL predictions simultaneously across XGBoost, Random Forest, and Linear Regression.
        """
        # Determine history window length
        history_len = min(self.window_size, max(2, int(cycle)))
        start_cycle = max(1, int(cycle) - history_len + 1)
        t_span = int(cycle) - start_cycle + 1

        rows = []
        for i, c in enumerate(range(start_cycle, int(cycle) + 1)):
            alpha = (i + 1) / max(1, t_span)
            row = {"engine_id": 999, "cycle": c}
            for k in ALL_RAW_COLS:
                if k in ["engine_id", "cycle"]:
                    continue
                nom = NOMINAL_HEALTHY_BASELINES.get(k, 0.0)
                curr = sensor_inputs.get(k, nom)
                # Small aerodynamic noise for realistic rolling std/diff
                noise = 0.0
                if k.startswith("sensor_") and i < t_span - 1:
                    noise = (np.sin(c * 1.5) * 0.02) * abs(curr - nom + 0.1)
                row[k] = nom + alpha * (curr - nom) + noise
            rows.append(row)

        df_sim = pd.DataFrame(rows)

        # Scale and extract sliding-window features
        df_scaled = self.preprocessor.transform(df_sim)
        df_feat = extract_sliding_window_features(
            df_scaled,
            sensor_cols=self.preprocessor.active_sensor_cols_,
            window_size=self.window_size
        )

        # Extract latest operational point
        X_inst = df_feat.iloc[[-1]][self.feature_names]

        # Predict across all 3 models
        model_predictions = {}
        for m_name in ["XGBoost", "Random Forest", "Linear Regression"]:
            if m_name in self.models:
                p_val = float(self.models[m_name].predict(X_inst)[0])
                model_predictions[m_name] = round(max(0.0, p_val), 1)

        primary_pred = model_predictions.get("XGBoost", float(np.mean(list(model_predictions.values()))))
        consensus_pred = round(float(np.mean(list(model_predictions.values()))), 1)

        # Check rapid wear
        rapid_wear = False
        t50_val = sensor_inputs.get("sensor_4", NOMINAL_HEALTHY_BASELINES["sensor_4"])
        ps30_val = sensor_inputs.get("sensor_11", NOMINAL_HEALTHY_BASELINES["sensor_11"])
        if t50_val > 1420.0 or ps30_val > 48.0:
            rapid_wear = True

        status = classify_maintenance_status(primary_pred, sensor_roc_warning=rapid_wear)

        # Sensor deviations from nominal baseline
        deviations = []
        key_tracked = ["sensor_4", "sensor_11", "sensor_3", "sensor_9", "sensor_14", "sensor_2", "sensor_7", "sensor_12", "sensor_15"]
        for s in key_tracked:
            curr = sensor_inputs.get(s, NOMINAL_HEALTHY_BASELINES[s])
            nom = NOMINAL_HEALTHY_BASELINES[s]
            meta = SENSOR_METADATA.get(s, {})
            delta = curr - nom
            pct = ((delta) / (abs(nom) + 1e-6)) * 100
            deviations.append({
                "sensor": s,
                "name": meta.get("name", s),
                "desc": meta.get("desc", ""),
                "unit": meta.get("unit", ""),
                "current": round(curr, 2),
                "nominal": round(nom, 2),
                "delta": round(delta, 2),
                "percent_drift": round(pct, 2)
            })

        return {
            "cycle": int(cycle),
            "primary_pred": primary_pred,
            "consensus_pred": consensus_pred,
            "model_predictions": model_predictions,
            "status": status,
            "deviations": deviations
        }


def validate_uploaded_csv(df_upload: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
    """Validates user-uploaded telemetry CSV for schema correctness."""
    df = df_upload.copy()
    if df.shape[1] == 26 and set(df.columns) == set(range(26)):
        df.columns = ALL_RAW_COLS
        return True, "Successfully mapped 26 raw C-MAPSS telemetry columns.", df

    req_meta = ["engine_id", "cycle"]
    missing_meta = [c for c in req_meta if c not in df.columns]
    if missing_meta:
        return False, f"Missing required index columns: {missing_meta}. Telemetry must include 'engine_id' and 'cycle'.", df

    sensor_found = [c for c in df.columns if c.startswith("sensor_")]
    if len(sensor_found) < 5:
        return False, f"Insufficient sensor columns detected ({len(sensor_found)} found). Expects at least 14 active C-MAPSS sensors.", df

    return True, f"Valid telemetry file loaded with {len(df)} records and {len(sensor_found)} sensor channels.", df
