"""
Generates the 5 research Jupyter notebooks for AeroGuard.
"""

import json
from pathlib import Path

nb_dir = Path(__file__).resolve().parent.parent / "notebooks"
nb_dir.mkdir(parents=True, exist_ok=True)


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.4"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }


# 01_EDA.ipynb
nb1 = create_notebook([
    md_cell("# AeroGuard Notebook 01: Exploratory Data Analysis (EDA)\nInvestigating multivariate turbofan degradation sensor streams in NASA C-MAPSS FD001."),
    code_cell(
"""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_dataset
from config import SENSOR_METADATA

df_train, df_test, s_rul = load_raw_dataset("FD001")
print(f"Train telemetry shape: {df_train.shape}")
print(f"Total run-to-failure engines: {df_train['engine_id'].nunique()}")
df_train.head()"""
    ),
    code_cell(
"""# Operating cycles to failure distribution
engine_lifespans = df_train.groupby("engine_id")["cycle"].max()

plt.figure(figsize=(8, 4), dpi=120)
sns.histplot(engine_lifespans, bins=20, kde=True, color="#0284C7")
plt.axvline(engine_lifespans.mean(), color="#EF4444", linestyle="--", label=f"Mean: {engine_lifespans.mean():.1f}")
plt.title("Turbofan Operational Lifespan Distribution (Cycles to Failure)")
plt.xlabel("Operating Flight Cycles")
plt.ylabel("Engine Count")
plt.legend()
plt.show()"""
    ),
    code_cell(
"""# Sensor statistics summary
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
df_train[sensor_cols].describe().T"""
    )
])
with open(nb_dir / "01_EDA.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)


# 02_Preprocessing.ipynb
nb2 = create_notebook([
    md_cell("# AeroGuard Notebook 02: Preprocessing & Leakage-Free Splitting\nIdentifying constant sensors, engine-wise train/validation partitioning, and scaling."),
    code_cell(
"""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from src.data_loader import load_raw_dataset
from src.rul_calculation import calculate_train_rul
from src.preprocessing import CMAPSSPreprocessor
from src.model_training import split_engine_wise

df_train, _, _ = load_raw_dataset("FD001")
df_train_rul = calculate_train_rul(df_train, max_rul_clip=125)

# Engine-wise split to avoid cross-record data leakage
df_tr_split, df_val_split, tr_ids, val_ids = split_engine_wise(df_train_rul, val_ratio=0.2)
print(f"Train engines: {len(tr_ids)} ({len(df_tr_split)} cycles)")
print(f"Validation engines: {len(val_ids)} ({len(df_val_split)} cycles)")"""
    ),
    code_cell(
"""# Fit preprocessor strictly on training fold
preprocessor = CMAPSSPreprocessor(scaler_type="standard")
preprocessor.fit(df_tr_split)

print("Constant/Invariant Channels Dropped:", preprocessor.constant_columns_)
print("Informative Telemetry Channels Retained:", len(preprocessor.active_sensor_cols_))

df_tr_scaled = preprocessor.transform(df_tr_split)
df_val_scaled = preprocessor.transform(df_val_split)
df_tr_scaled.head()"""
    )
])
with open(nb_dir / "02_Preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)


# 03_Degradation_Analysis.ipynb
nb3 = create_notebook([
    md_cell("# AeroGuard Notebook 03: Turbofan Degradation Trend Analysis\nAnalyzing physics-informed degradation trajectories in High Pressure Compressor (HPC) sensors."),
    code_cell(
"""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import matplotlib.pyplot as plt
from src.data_loader import load_raw_dataset
from config import SENSOR_METADATA

df_train, _, _ = load_raw_dataset("FD001")

key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9", "sensor_11", "sensor_14"]
fig, axes = plt.subplots(len(key_sensors), 1, figsize=(10, 14), sharex=True, dpi=120)

for ax, s in zip(axes, key_sensors):
    meta = SENSOR_METADATA.get(s, {})
    for eid in [1, 2, 3]:
        sub = df_train[df_train["engine_id"] == eid]
        ax.plot(sub["cycle"], sub[s], label=f"Engine {eid}", alpha=0.85)
    ax.set_ylabel(meta.get("name", s), fontweight="bold")
    ax.set_title(f"{s.upper()}: {meta.get('desc', '')} ({meta.get('unit', '')})", fontsize=10, loc="left")

axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Operational Flight Cycles", fontweight="bold")
plt.tight_layout()
plt.show()"""
    )
])
with open(nb_dir / "03_Degradation_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb3, f, indent=2)


# 04_Feature_Engineering.ipynb
nb4 = create_notebook([
    md_cell("# AeroGuard Notebook 04: Sliding-Window Feature Extraction\nRolling statistics (mean, std, min, max, latest, rate-of-change, consecutive difference) isolated per engine."),
    code_cell(
"""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from src.data_loader import load_raw_dataset
from src.preprocessing import CMAPSSPreprocessor
from src.feature_engineering import extract_sliding_window_features, prepare_feature_experiments

df_train, df_test, s_rul = load_raw_dataset("FD001")
preprocessor = CMAPSSPreprocessor().fit(df_train)
df_scaled = preprocessor.transform(df_train)

# Sliding window with 30-cycle temporal horizon
df_features = extract_sliding_window_features(
    df_scaled, 
    sensor_cols=preprocessor.active_sensor_cols_, 
    window_size=30
)
print("Engineered feature matrix shape:", df_features.shape)
df_features.head()"""
    )
])
with open(nb_dir / "04_Feature_Engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=2)


# 05_Model_Comparison.ipynb
nb5 = create_notebook([
    md_cell("# AeroGuard Notebook 05: Multi-Model Regression Benchmarking & Explainability\nBenchmarking Linear, Random Forest, Gradient Boosting, XGBoost, and SVR with NASA Asymmetric Scoring and SHAP."),
    code_cell(
"""import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
from config import REPORTS_DIR

# Empirical benchmark comparison table
df_comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
print("=== MODEL BENCHMARK RESULTS (TEST EVALUATION) ===")
df_comp[["model_name", "test_rmse", "test_mae", "test_r2", "test_nasa_score", "test_overestimate_pct", "test_critical_overestimates"]]"""
    ),
    code_cell(
"""# Window size sensitivity experiment (Raw vs 15, 30, 50 cycles)
df_win = pd.read_csv(REPORTS_DIR / "window_size_experiment.csv")
print("=== SLIDING-WINDOW SENSITIVITY ===")
df_win"""
    ),
    code_cell(
"""# SHAP Physical Sensor Importance
df_shap = pd.read_csv(REPORTS_DIR / "sensor_importance.csv")
print("=== TOP TURBOFAN SENSOR DEGRADATION DRIVERS (SHAP) ===")
df_shap.head(8)"""
    )
])
with open(nb_dir / "05_Model_Comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb5, f, indent=2)

print("All 5 Jupyter research notebooks successfully created.")
