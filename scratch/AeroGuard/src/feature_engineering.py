"""
AeroGuard Sliding-Window Feature Engineering Module
Extracts temporal degradation statistics (mean, std, min, max, latest,
rate-of-change, consecutive differences) over configurable window sizes
while strictly enforcing engine grouping boundaries and chronological causality.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_WINDOW_SIZE, WINDOW_SIZES_EXPERIMENT, RUL_COL


def extract_sliding_window_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    include_raw: bool = True
) -> pd.DataFrame:
    """
    Extracts rolling temporal degradation features for each sensor per engine.

    Features generated per sensor:
      - `{sensor}_latest`: Most recent snapshot value
      - `{sensor}_mean`: Rolling window mean
      - `{sensor}_std`: Rolling window standard deviation (imputed 0 for single-cycle)
      - `{sensor}_min`: Rolling window minimum
      - `{sensor}_max`: Rolling window maximum
      - `{sensor}_diff`: Consecutive reading difference (t - (t-1))
      - `{sensor}_roc`: Rate of change over the effective window span

    Boundary rules:
      - Rolling operations are grouped strictly by 'engine_id' to prevent cross-engine leakage.
      - min_periods=1 ensures initial cycles (< window_size) are preserved smoothly.
      - No future information is utilized.
    """
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise ValueError("DataFrame must contain 'engine_id' and 'cycle' columns.")

    # Sort strictly chronologically
    df_sorted = df.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)
    grouped = df_sorted.groupby("engine_id")

    feature_blocks = []

    # Preserve index / meta columns
    meta_cols = ["engine_id", "cycle"]
    if RUL_COL in df_sorted.columns:
        meta_cols.append(RUL_COL)
    if "RUL_raw" in df_sorted.columns:
        meta_cols.append("RUL_raw")
    feature_blocks.append(df_sorted[meta_cols])

    for sensor in sensor_cols:
        if sensor not in df_sorted.columns:
            continue

        s_series = df_sorted[sensor]

        # Consecutive difference within engine
        diff = grouped[sensor].diff().fillna(0.0)

        # Rolling window calculations
        roller = grouped[sensor].rolling(window=window_size, min_periods=1)
        mean_val = roller.mean().reset_index(level=0, drop=True)
        std_val = roller.std().reset_index(level=0, drop=True).fillna(0.0)
        min_val = roller.min().reset_index(level=0, drop=True)
        max_val = roller.max().reset_index(level=0, drop=True)

        # Rate of change over the window
        shift_w = grouped[sensor].shift(window_size - 1)
        first_val = grouped[sensor].transform("first")
        window_start = shift_w.fillna(first_val)
        span = df_sorted["cycle"].clip(upper=window_size)
        roc = (s_series - window_start) / np.maximum(1, span - 1)

        sensor_features = {
            f"{sensor}_latest": s_series,
            f"{sensor}_mean": mean_val,
            f"{sensor}_std": std_val,
            f"{sensor}_min": min_val,
            f"{sensor}_max": max_val,
            f"{sensor}_diff": diff,
            f"{sensor}_roc": roc,
        }

        feature_blocks.append(pd.DataFrame(sensor_features, index=df_sorted.index))

    df_features = pd.concat(feature_blocks, axis=1)
    return df_features


def extract_raw_features(
    df: pd.DataFrame,
    sensor_cols: List[str]
) -> pd.DataFrame:
    """
    Extracts raw instantaneous sensor readings without temporal window statistics.
    Used as the scientific baseline to benchmark against sliding-window representations.
    """
    meta_cols = ["engine_id", "cycle"]
    if RUL_COL in df.columns:
        meta_cols.append(RUL_COL)
    if "RUL_raw" in df.columns:
        meta_cols.append("RUL_raw")

    active_sensors = [s for s in sensor_cols if s in df.columns]
    return df[meta_cols + active_sensors].copy()


def prepare_feature_experiments(
    df_train_scaled: pd.DataFrame,
    df_test_scaled: pd.DataFrame,
    sensor_cols: List[str],
    window_sizes: List[int] = WINDOW_SIZES_EXPERIMENT
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generates feature sets for raw sensors and multiple sliding-window sizes
    (e.g., 15, 30, 50 cycles) to evaluate temporal depth effects.
    """
    datasets = {}

    # 1. Baseline: Raw Sensor Features
    df_train_raw = extract_raw_features(df_train_scaled, sensor_cols)
    df_test_raw = extract_raw_features(df_test_scaled, sensor_cols)
    datasets["raw"] = (df_train_raw, df_test_raw)

    # 2. Sliding-Window Sets
    for w in window_sizes:
        df_train_w = extract_sliding_window_features(df_train_scaled, sensor_cols, window_size=w)
        df_test_w = extract_sliding_window_features(df_test_scaled, sensor_cols, window_size=w)
        datasets[f"window_{w}"] = (df_train_w, df_test_w)

    return datasets
