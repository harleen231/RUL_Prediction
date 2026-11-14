"""
Unit tests for RUL Calculation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.rul_calculation import calculate_train_rul, calculate_test_rul, get_test_last_cycle_records
from config import RUL_COL


def test_calculate_train_rul_formula():
    # Synthetic 2-engine test dataframe
    # Engine 1 has 5 cycles (max=5), Engine 2 has 3 cycles (max=3)
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "cycle":     [1, 2, 3, 4, 5, 1, 2, 3]
    })

    # Unclipped
    res = calculate_train_rul(df, max_rul_clip=None)
    # Expected RUL_raw: [4, 3, 2, 1, 0, 2, 1, 0]
    expected_rul = [4, 3, 2, 1, 0, 2, 1, 0]
    np.testing.assert_array_equal(res[RUL_COL].values, expected_rul)

    # Clipped at 2
    res_clipped = calculate_train_rul(df, max_rul_clip=2)
    expected_clipped = [2, 2, 2, 1, 0, 2, 1, 0]
    np.testing.assert_array_equal(res_clipped[RUL_COL].values, expected_clipped)


def test_calculate_test_rul_with_ground_truth():
    df_test = pd.DataFrame({
        "engine_id": [1, 1, 2, 2, 2],
        "cycle":     [1, 2, 1, 2, 3]
    })
    # Ground truth: at last cycle (2 for engine 1, 3 for engine 2), remaining life is 10 and 20
    s_truth = pd.Series([10, 20], index=[1, 2])

    res = calculate_test_rul(df_test, s_truth, max_rul_clip=None)
    # For engine 1: last cycle is 2, RUL at cycle 2 = 10, RUL at cycle 1 = 10 + (2-1) = 11
    # For engine 2: last cycle is 3, RUL at cycle 3 = 20, cycle 2 = 21, cycle 1 = 22
    expected = [11, 10, 22, 21, 20]
    np.testing.assert_array_equal(res[RUL_COL].values, expected)

    # Check last cycle extraction
    last_records = get_test_last_cycle_records(res)
    assert len(last_records) == 2
    assert last_records.loc[last_records["engine_id"] == 1, RUL_COL].values[0] == 10
    assert last_records.loc[last_records["engine_id"] == 2, RUL_COL].values[0] == 20
