"""
AeroGuard – preprocessing.py
Handles data loading, RUL labeling, engine-wise splitting,
feature scaling, and sliding-window feature extraction.
All transformations are fit ONLY on training data to prevent leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

# ─── Column names for the raw C-MAPSS text files ──────────────────────────────
INDEX_COLS   = ["engine_id", "cycle"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS  = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS     = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors that carry no useful information (nearly constant across all engines)
INVARIANT_SENSORS = ["op_setting_3", "sensor_1", "sensor_5",
                     "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

# Sensors that actually vary and help predict RUL
ACTIVE_SENSORS = [s for s in SENSOR_COLS if s not in INVARIANT_SENSORS]

# Default modeling constants
MAX_RUL      = 125    # piecewise linear RUL clipping limit (literature standard)
WINDOW_SIZE  = 30     # sliding-window length in operational cycles
VAL_RATIO    = 0.20   # fraction of engines used for validation
RANDOM_SEED  = 42

# ─── Sensor metadata (name, description, unit) ────────────────────────────────
SENSOR_META = {
    "sensor_2":  {"name": "T24",     "desc": "Total temp at LPC outlet",      "unit": "°R"},
    "sensor_3":  {"name": "T30",     "desc": "Total temp at HPC outlet",      "unit": "°R"},
    "sensor_4":  {"name": "T50",     "desc": "Total temp at LPT outlet",      "unit": "°R"},
    "sensor_6":  {"name": "P15",     "desc": "Total pressure in bypass-duct", "unit": "psia"},
    "sensor_7":  {"name": "P30",     "desc": "Total pressure at HPC outlet",  "unit": "psia"},
    "sensor_8":  {"name": "Nf",      "desc": "Physical fan speed",            "unit": "rpm"},
    "sensor_9":  {"name": "Nc",      "desc": "Physical core speed",           "unit": "rpm"},
    "sensor_11": {"name": "Ps30",    "desc": "Static pressure at HPC outlet", "unit": "psia"},
    "sensor_12": {"name": "phi",     "desc": "Fuel flow / Ps30 ratio",        "unit": "pps/psia"},
    "sensor_13": {"name": "NRf",     "desc": "Corrected fan speed",           "unit": "rpm"},
    "sensor_14": {"name": "NRc",     "desc": "Corrected core speed",          "unit": "rpm"},
    "sensor_15": {"name": "BPR",     "desc": "Bypass Ratio",                  "unit": "—"},
    "sensor_17": {"name": "htBleed", "desc": "Bleed Enthalpy",                "unit": "—"},
    "sensor_20": {"name": "W31",     "desc": "HPT coolant bleed",             "unit": "lbm/s"},
    "sensor_21": {"name": "W32",     "desc": "LPT coolant bleed",             "unit": "lbm/s"},
}

# Nominal healthy sensor values (mean of first 15 cycles across FD001 engines)
HEALTHY_DEFAULTS = {
    "op_setting_1": 0.0000,
    "op_setting_2": 0.0000,
    "op_setting_3": 100.0,
    "sensor_1":  518.67, "sensor_2":  642.38, "sensor_3":  1586.81,
    "sensor_4":  1402.74,"sensor_5":  14.62,  "sensor_6":  21.61,
    "sensor_7":  553.95, "sensor_8":  2388.06,"sensor_9":  9056.04,
    "sensor_10": 1.30,   "sensor_11": 47.35,  "sensor_12": 521.91,
    "sensor_13": 2388.06,"sensor_14": 8137.37,"sensor_15": 8.42,
    "sensor_16": 0.03,   "sensor_17": 392.28, "sensor_18": 2388.0,
    "sensor_19": 100.0,  "sensor_20": 38.93,  "sensor_21": 23.36,
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(subset="FD001"):
    """
    Loads the train, test, and ground-truth RUL files for a C-MAPSS subset.
    Returns (df_train, df_test, rul_series).
    """
    train_path = DATA_DIR / f"train_{subset}.txt"
    test_path  = DATA_DIR / f"test_{subset}.txt"
    rul_path   = DATA_DIR / f"RUL_{subset}.txt"

    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                "Please place the NASA C-MAPSS text files inside the 'data/' folder.\n"
                "You can download them from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
            )

    def _read(path):
        df = pd.read_csv(path, sep=r"\s+", header=None)
        # Drop any trailing empty columns the dataset sometimes has
        if df.shape[1] > len(ALL_COLS):
            df = df.iloc[:, :len(ALL_COLS)]
        df.columns = ALL_COLS
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"]     = df["cycle"].astype(int)
        return df

    df_train = _read(train_path)
    df_test  = _read(test_path)

    # RUL truth: one value per test engine, indexed 1..N
    rul_raw = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_series = rul_raw.iloc[:, 0].copy()
    rul_series.index = np.arange(1, len(rul_series) + 1)
    rul_series.name = "true_rul"

    return df_train, df_test, rul_series


# ─── RUL calculation ──────────────────────────────────────────────────────────

def calculate_train_rul(df_train, max_rul=MAX_RUL):
    """
    Adds 'RUL' column to training data.
    Formula: RUL = max_cycle_for_engine - current_cycle, clipped at max_rul.
    """
    df = df_train.copy()
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (max_cycles - df["cycle"]).clip(upper=max_rul)
    return df


def calculate_test_rul(df_test, rul_series, max_rul=MAX_RUL):
    """
    Adds 'RUL' column to test data using the official ground-truth RUL file.
    For each engine: RUL at cycle t = ground_truth_RUL + (last_cycle - t)
    This correctly represents how far from failure every test cycle is.
    """
    df = df_test.copy()
    last_cycles = df.groupby("engine_id")["cycle"].transform("max")
    rul_mapped  = df["engine_id"].map(rul_series)
    df["RUL"]   = (rul_mapped + (last_cycles - df["cycle"])).clip(upper=max_rul)
    return df


# ─── Engine-wise train/validation split ───────────────────────────────────────

def split_engines(df_train, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """
    Splits training data into train and validation sets by engine ID.
    Keeps all cycles of each engine together — never splits individual rows randomly.
    Returns (df_tr, df_val).
    """
    rng = np.random.default_rng(seed)
    engine_ids = df_train["engine_id"].unique()
    rng.shuffle(engine_ids)

    n_val = max(1, int(len(engine_ids) * val_ratio))
    val_ids   = set(engine_ids[:n_val])
    train_ids = set(engine_ids[n_val:])

    df_tr  = df_train[df_train["engine_id"].isin(train_ids)].copy()
    df_val = df_train[df_train["engine_id"].isin(val_ids)].copy()
    return df_tr, df_val


# ─── Feature scaling ──────────────────────────────────────────────────────────

def fit_scaler(df_train, feature_cols):
    """
    Fits a StandardScaler on the training data.
    Returns the fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    return scaler


def apply_scaler(df, scaler, feature_cols):
    """
    Applies a pre-fitted scaler to the given dataframe.
    Only scales the specified columns; all other columns are preserved.
    """
    df_out = df.copy()
    df_out[feature_cols] = scaler.transform(df_out[feature_cols])
    return df_out


# ─── Sliding-window feature extraction ────────────────────────────────────────

def extract_window_features(df, sensor_cols, window=WINDOW_SIZE):
    """
    Creates rolling temporal features for each sensor per engine.
    For each sensor, produces 7 features:
      _latest  → current snapshot value
      _mean    → rolling mean over the window
      _std     → rolling standard deviation
      _min     → rolling minimum
      _max     → rolling maximum
      _diff    → difference from previous cycle
      _roc     → rate of change over the window

    Windows never cross engine boundaries.
    Engines with fewer cycles than 'window' still get valid features (min_periods=1).
    """
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise ValueError("DataFrame must have 'engine_id' and 'cycle' columns.")

    # Sort so rolling is always in time order
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("engine_id")

    # Keep metadata columns
    meta_cols = ["engine_id", "cycle"]
    if "RUL" in df.columns:
        meta_cols.append("RUL")

    feature_blocks = [df[meta_cols].copy()]

    for sensor in sensor_cols:
        if sensor not in df.columns:
            continue

        s = df[sensor]

        diff     = grouped[sensor].diff().fillna(0.0)
        roller   = grouped[sensor].rolling(window=window, min_periods=1)
        mean_val = roller.mean().reset_index(level=0, drop=True)
        std_val  = roller.std().reset_index(level=0, drop=True).fillna(0.0)
        min_val  = roller.min().reset_index(level=0, drop=True)
        max_val  = roller.max().reset_index(level=0, drop=True)

        # Rate of change: (current - value at start of window) / window span
        shift_w     = grouped[sensor].shift(window - 1)
        first_val   = grouped[sensor].transform("first")
        window_start = shift_w.fillna(first_val)
        span         = df["cycle"].clip(upper=window)
        roc          = (s - window_start) / np.maximum(1, span - 1)

        feature_blocks.append(pd.DataFrame({
            f"{sensor}_latest": s.values,
            f"{sensor}_mean":   mean_val.values,
            f"{sensor}_std":    std_val.values,
            f"{sensor}_min":    min_val.values,
            f"{sensor}_max":    max_val.values,
            f"{sensor}_diff":   diff.values,
            f"{sensor}_roc":    roc.values,
        }, index=df.index))

    return pd.concat(feature_blocks, axis=1)


# ─── Full preprocessing pipeline (used by train.py) ───────────────────────────

def build_features(df_train_raw, df_val_raw, df_test_raw):
    """
    Runs the full preprocessing pipeline:
      1. Scale active sensors (fit on train only)
      2. Extract sliding-window features
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler, sensor_stats).
    """
    feature_cols = ["op_setting_1", "op_setting_2"] + ACTIVE_SENSORS

    # Fit scaler only on training data
    scaler = fit_scaler(df_train_raw, feature_cols)

    df_tr_sc  = apply_scaler(df_train_raw,  scaler, feature_cols)
    df_val_sc = apply_scaler(df_val_raw,    scaler, feature_cols)
    df_te_sc  = apply_scaler(df_test_raw,   scaler, feature_cols)

    # Extract sliding-window features
    df_tr_feat  = extract_window_features(df_tr_sc,  ACTIVE_SENSORS)
    df_val_feat = extract_window_features(df_val_sc, ACTIVE_SENSORS)
    df_te_feat  = extract_window_features(df_te_sc,  ACTIVE_SENSORS)

    feature_names = [c for c in df_tr_feat.columns
                     if c not in ("engine_id", "cycle", "RUL")]

    X_train = df_tr_feat[feature_names].values
    y_train = df_tr_feat["RUL"].values
    X_val   = df_val_feat[feature_names].values
    y_val   = df_val_feat["RUL"].values
    X_test  = df_te_feat[feature_names].values
    y_test  = df_te_feat["RUL"].values

    # Save per-sensor stats (min/max/mean) from raw training data for simulator bounds
    sensor_stats = {}
    for s in ACTIVE_SENSORS + ["op_setting_1", "op_setting_2"]:
        if s in df_train_raw.columns:
            sensor_stats[s] = {
                "min":  float(df_train_raw[s].min()),
                "max":  float(df_train_raw[s].max()),
                "mean": float(df_train_raw[s].mean()),
                "std":  float(df_train_raw[s].std()),
            }

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler, sensor_stats
