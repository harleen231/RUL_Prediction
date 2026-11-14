"""
Unit tests for Sliding-Window Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import extract_sliding_window_features


def test_sliding_window_engine_boundary_isolation():
    """
    Ensures that sliding window calculations never cross engine boundaries.
    """
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "cycle":     [1, 2, 3, 1, 2, 3],
        "sensor_2":  [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    })

    feat = extract_sliding_window_features(df, sensor_cols=["sensor_2"], window_size=2)

    # For engine 1:
    # cycle 1: mean = 10, std = 0, diff = 0
    # cycle 2: mean = 15, diff = 10
    # cycle 3: mean = 25, diff = 10
    # For engine 2:
    # cycle 1: mean = 100 (MUST NOT include engine 1 values!), diff = 0
    # cycle 2: mean = 150, diff = 100
    # cycle 3: mean = 250, diff = 100
    assert feat.loc[0, "sensor_2_mean"] == 10.0
    assert feat.loc[1, "sensor_2_mean"] == 15.0
    assert feat.loc[2, "sensor_2_mean"] == 25.0

    # Critical check: Engine 2 cycle 1 must be exactly 100.0, not contaminated with 30.0 from Engine 1!
    assert feat.loc[3, "sensor_2_mean"] == 100.0
    assert feat.loc[3, "sensor_2_diff"] == 0.0
    assert feat.loc[4, "sensor_2_mean"] == 150.0
    assert feat.loc[5, "sensor_2_mean"] == 250.0


def test_sliding_window_handling_short_trajectories():
    """
    Ensures engines with fewer cycles than window_size are handled without NaN.
    """
    df = pd.DataFrame({
        "engine_id": [1, 1],
        "cycle":     [1, 2],
        "sensor_3":  [5.0, 7.0]
    })

    # Window size 30 with only 2 rows
    feat = extract_sliding_window_features(df, sensor_cols=["sensor_3"], window_size=30)
    assert len(feat) == 2
    assert feat.isnull().sum().sum() == 0
    assert feat.loc[1, "sensor_3_mean"] == 6.0
    assert feat.loc[1, "sensor_3_latest"] == 7.0
