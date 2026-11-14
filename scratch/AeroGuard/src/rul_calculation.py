"""
AeroGuard RUL Calculation Module
Computes Remaining Useful Life (RUL) labels for training and test trajectories,
supporting configurable piecewise-linear RUL clipping (default 125 cycles).
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_MAX_RUL, RUL_COL


def calculate_train_rul(
    df_train: pd.DataFrame,
    max_rul_clip: Optional[int] = DEFAULT_MAX_RUL
) -> pd.DataFrame:
    """
    Computes RUL for run-to-failure training engines.
    Formula: RUL(i, t) = max_cycle(i) - current_cycle(t)
    Optionally applies piecewise-linear clipping: min(RUL, max_rul_clip).

    Returns a copy of df_train with added 'RUL' and 'RUL_raw' columns.
    """
    df = df_train.copy()
    
    # Calculate maximum cycle per engine
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL_raw"] = max_cycles - df["cycle"]
    
    if max_rul_clip is not None and max_rul_clip > 0:
        df[RUL_COL] = df["RUL_raw"].clip(upper=max_rul_clip)
    else:
        df[RUL_COL] = df["RUL_raw"]

    return df


def calculate_test_rul(
    df_test: pd.DataFrame,
    s_rul_truth: pd.Series,
    max_rul_clip: Optional[int] = DEFAULT_MAX_RUL
) -> pd.DataFrame:
    """
    Computes RUL for test engines using ground-truth RUL from RUL_FD00x.txt.
    For each test engine i, with last observed cycle T_last(i) and true RUL R_truth(i):
      Failure cycle T_fail(i) = T_last(i) + R_truth(i)
      RUL(i, t) = T_fail(i) - t = R_truth(i) + (T_last(i) - t)

    Returns a copy of df_test with added 'RUL' and 'RUL_raw' columns.
    """
    df = df_test.copy()
    
    # Map last observed cycle per test engine
    last_cycles = df.groupby("engine_id")["cycle"].transform("max")
    
    # Map ground-truth RUL to each row based on engine_id
    rul_mapped = df["engine_id"].map(s_rul_truth)
    if rul_mapped.isnull().any():
        missing_ids = df.loc[rul_mapped.isnull(), "engine_id"].unique()
        raise ValueError(f"Missing ground-truth RUL for engine IDs: {missing_ids}")

    # RUL at cycle t
    df["RUL_raw"] = rul_mapped + (last_cycles - df["cycle"])
    
    if max_rul_clip is not None and max_rul_clip > 0:
        df[RUL_COL] = df["RUL_raw"].clip(upper=max_rul_clip)
    else:
        df[RUL_COL] = df["RUL_raw"]

    return df


def get_test_last_cycle_records(
    df_test_with_rul: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts the final recorded operational snapshot for each test engine.
    This corresponds to the official benchmark evaluation point for NASA C-MAPSS.
    """
    idx_last = df_test_with_rul.groupby("engine_id")["cycle"].idxmax()
    return df_test_with_rul.loc[idx_last].reset_index(drop=True)
