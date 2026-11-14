"""
AeroGuard Preprocessing Pipeline Module
Identifies non-informative/constant sensors, scales multivariate sensor streams,
and prevents data leakage by fitting transformers strictly on training partitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    INDEX_COLS,
    SETTING_COLS,
    SENSOR_COLS,
    VARIANCE_THRESHOLD,
    MODELS_DIR,
)


class CMAPSSPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessor for multivariate C-MAPSS sensor telemetry.
    Learns constant/non-informative features and standardizes continuous telemetry.
    """

    def __init__(
        self,
        scaler_type: str = "standard",  # "standard" or "minmax"
        variance_threshold: float = VARIANCE_THRESHOLD,
        drop_settings: bool = False
    ):
        self.scaler_type = scaler_type
        self.variance_threshold = variance_threshold
        self.drop_settings = drop_settings

        self.constant_columns_: List[str] = []
        self.active_sensor_cols_: List[str] = []
        self.feature_cols_: List[str] = []
        self.scaler_: Union[StandardScaler, MinMaxScaler] = (
            StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        )
        self.feature_stats_: Dict[str, dict] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame, y=None):
        """
        Fits the preprocessor on the training data.
        Determines invariant sensors and computes scaling statistics.
        """
        feature_candidates = (
            SENSOR_COLS if self.drop_settings else SETTING_COLS + SENSOR_COLS
        )
        available_cols = [c for c in feature_candidates if c in df.columns]

        # Compute standard deviations across training set
        stds = df[available_cols].std()
        self.constant_columns_ = stds[stds <= self.variance_threshold].index.tolist()
        self.active_sensor_cols_ = [
            c for c in SENSOR_COLS if c in available_cols and c not in self.constant_columns_
        ]
        self.feature_cols_ = [c for c in available_cols if c not in self.constant_columns_]

        # Fit the scaler on informative features
        self.scaler_.fit(df[self.feature_cols_])

        # Cache descriptive stats for EDA / telemetry inspection
        for col in available_cols:
            self.feature_stats_[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "is_constant": col in self.constant_columns_
            }

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input dataframe using fitted scaler and drops constant features.
        Preserves index and target columns (engine_id, cycle, RUL, etc.) intact.
        """
        if not self.is_fitted_:
            raise RuntimeError("CMAPSSPreprocessor must be fitted before transforming data.")

        df_out = df.copy()

        # Scale features
        scaled_array = self.scaler_.transform(df_out[self.feature_cols_])
        scaled_df = pd.DataFrame(
            scaled_array,
            columns=self.feature_cols_,
            index=df_out.index
        )

        # Update columns in copy
        for col in self.feature_cols_:
            df_out[col] = scaled_df[col]

        # Drop constant columns
        df_out = df_out.drop(columns=[c for c in self.constant_columns_ if c in df_out.columns])
        return df_out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        path = Path(filepath) if filepath else MODELS_DIR / "preprocessor.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "CMAPSSPreprocessor":
        path = Path(filepath) if filepath else MODELS_DIR / "preprocessor.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No saved preprocessor found at '{path}'")
        return joblib.load(path)


def compute_sensor_correlations(
    df: pd.DataFrame,
    target_col: Optional[str] = "RUL"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes Pearson correlation matrix among sensors and correlation with RUL target.
    """
    available_sensors = [c for c in SENSOR_COLS if c in df.columns]
    cols_to_corr = available_sensors + ([target_col] if target_col in df.columns else [])
    
    corr_matrix = df[cols_to_corr].corr()
    target_corr = (
        corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        if target_col in corr_matrix.columns
        else pd.Series(dtype=float)
    )
    return corr_matrix, target_corr
