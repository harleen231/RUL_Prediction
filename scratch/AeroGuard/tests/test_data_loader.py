"""
Unit tests for AeroGuard Data Loader
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import locate_dataset_files, load_raw_dataset
from config import ALL_RAW_COLS, DATA_DIR


def test_locate_dataset_files():
    train_p, test_p, rul_p = locate_dataset_files("FD001", DATA_DIR)
    assert train_p.exists(), f"Train file missing at {train_p}"
    assert test_p.exists(), f"Test file missing at {test_p}"
    assert rul_p.exists(), f"RUL file missing at {rul_p}"


def test_load_raw_dataset_schema():
    df_train, df_test, s_rul = load_raw_dataset("FD001", DATA_DIR)
    
    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_test, pd.DataFrame)
    assert isinstance(s_rul, pd.Series)

    assert df_train.shape[1] == len(ALL_RAW_COLS)
    assert df_test.shape[1] == len(ALL_RAW_COLS)

    # Check key columns exist
    assert "engine_id" in df_train.columns
    assert "cycle" in df_train.columns
    assert "sensor_1" in df_train.columns
    assert "sensor_21" in df_train.columns

    # Verify no nulls
    assert df_train.isnull().sum().sum() == 0
    assert df_test.isnull().sum().sum() == 0

    # Engine count check for FD001
    assert df_train["engine_id"].nunique() == 100
    assert df_test["engine_id"].nunique() == 100
    assert len(s_rul) == 100
