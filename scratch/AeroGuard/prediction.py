"""
AeroGuard – prediction.py
Functions for making RUL predictions from engine history or manual simulator inputs.
Used by the dashboard (app.py) — not to be run directly.
"""

import numpy as np
import pandas as pd

from preprocessing import (
    apply_scaler,
    extract_window_features,
    ACTIVE_SENSORS,
    ALL_COLS,
    HEALTHY_DEFAULTS,
    WINDOW_SIZE,
    MAX_RUL,
)

# Maintenance status thresholds
THRESHOLDS = {
    "HEALTHY":   {"min": 76,  "color": "#10B981", "desc": "Normal flight operation"},
    "ATTENTION": {"min": 31,  "color": "#F59E0B", "desc": "Schedule routine inspection"},
    "WARNING":   {"min": 11,  "color": "#F97316", "desc": "Plan overhaul soon"},
    "CRITICAL":  {"min": 0,   "color": "#EF4444", "desc": "Immediate maintenance required"},
}

# Approximate sensor failure endpoints (observed near end-of-life in FD001)
FAILURE_ENDPOINTS = {
    "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
    "sensor_1":  518.67, "sensor_2":  643.58, "sensor_3":  1601.28,
    "sensor_4":  1427.10,"sensor_5":  14.62,  "sensor_6":  21.61,
    "sensor_7":  551.65, "sensor_8":  2388.22,"sensor_9":  9095.01,
    "sensor_10": 1.30,   "sensor_11": 48.09,  "sensor_12": 519.95,
    "sensor_13": 2388.22,"sensor_14": 8164.79,"sensor_15": 8.51,
    "sensor_16": 0.03,   "sensor_17": 396.02, "sensor_18": 2388.0,
    "sensor_19": 100.0,  "sensor_20": 38.48,  "sensor_21": 23.09,
}

# Ready-to-use simulator presets
SIMULATOR_PRESETS = {
    "🟢 Brand New Engine (Healthy)": {
        "cycle": 10,
        "sensors": HEALTHY_DEFAULTS.copy(),
    },
    "🟡 Mid-Life (Attention)": {
        "cycle": 110,
        "sensors": {
            k: HEALTHY_DEFAULTS[k] * 0.6 + FAILURE_ENDPOINTS[k] * 0.4
            for k in HEALTHY_DEFAULTS
        },
    },
    "🟠 Accelerated Wear (Warning)": {
        "cycle": 190,
        "sensors": {
            k: HEALTHY_DEFAULTS[k] * 0.25 + FAILURE_ENDPOINTS[k] * 0.75
            for k in HEALTHY_DEFAULTS
        },
    },
    "🔴 Near-Failure (Critical)": {
        "cycle": 230,
        "sensors": FAILURE_ENDPOINTS.copy(),
    },
}


def classify_rul(predicted_rul):
    """
    Maps a numeric predicted RUL value to a maintenance status category.
    Returns a dict with keys: category, color, description.
    """
    rul = max(0.0, float(predicted_rul))
    if rul > 75:
        cat = "HEALTHY"
    elif rul >= 31:
        cat = "ATTENTION"
    elif rul >= 11:
        cat = "WARNING"
    else:
        cat = "CRITICAL"

    cfg = THRESHOLDS[cat]
    return {
        "category":    cat,
        "color":       cfg["color"],
        "description": cfg["desc"],
        "predicted_rul": round(rul, 1),
    }


def predict_from_history(df_engine_history, model, scaler, feature_names):
    """
    Given a DataFrame of raw sensor readings for one engine over multiple cycles,
    runs the full preprocessing pipeline and returns predicted RUL for each cycle.

    df_engine_history must include columns: engine_id, cycle, all sensor columns.
    Returns a Series of predicted RUL values aligned to df_engine_history.
    """
    feature_cols = ["op_setting_1", "op_setting_2"] + ACTIVE_SENSORS
    df_scaled    = apply_scaler(df_engine_history, scaler, feature_cols)
    df_feat      = extract_window_features(df_scaled, ACTIVE_SENSORS)
    X            = df_feat[feature_names].values
    preds        = model.predict(X).clip(0, MAX_RUL)
    return pd.Series(preds, index=df_engine_history.index, name="predicted_rul")


def predict_from_inputs(sensor_dict, cycle, model, scaler, feature_names, window=WINDOW_SIZE):
    """
    Simulator prediction: takes a dict of raw sensor values and the current cycle,
    synthesises a window of history by interpolating from the nominal healthy baseline
    to the current input values, then runs the full preprocessing pipeline.

    Returns the predicted RUL (float).
    """
    cycle       = int(cycle)
    history_len = min(window, max(2, cycle))
    start_cycle = max(1, cycle - history_len + 1)
    t_span      = cycle - start_cycle + 1

    rows = []
    for i, c in enumerate(range(start_cycle, cycle + 1)):
        # alpha=0 → start of window (healthy baseline), alpha=1 → current input
        alpha = (i + 1) / max(1, t_span)
        row   = {"engine_id": 999, "cycle": c}
        for col in ALL_COLS:
            if col in ("engine_id", "cycle"):
                continue
            nominal = HEALTHY_DEFAULTS.get(col, 0.0)
            current = sensor_dict.get(col, nominal)
            # Small sinusoidal noise to produce realistic rolling std / diff features
            noise = 0.0
            if col.startswith("sensor_") and i < t_span - 1:
                noise = np.sin(c * 1.5) * 0.02 * abs(current - nominal + 0.1)
            row[col] = nominal + alpha * (current - nominal) + noise
        rows.append(row)

    df_sim = pd.DataFrame(rows)

    feature_cols = ["op_setting_1", "op_setting_2"] + ACTIVE_SENSORS
    df_scaled    = apply_scaler(df_sim, scaler, feature_cols)
    df_feat      = extract_window_features(df_scaled, ACTIVE_SENSORS, window=window)

    X_inst = df_feat.iloc[[-1]][feature_names].values
    pred   = float(model.predict(X_inst)[0])
    return max(0.0, pred)
