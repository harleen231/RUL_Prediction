"""
AeroGuard Configuration Module
Defines project paths, column mappings, sensor metadata, modeling parameters,
sliding window configurations, and maintenance risk thresholds.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Subsets
AVAILABLE_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_SUBSET = "FD001"

# Column Schema for C-MAPSS
INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Detailed Sensor Metadata for Aeronautics Engineering
SENSOR_METADATA = {
    "sensor_1": {"name": "T2", "desc": "Total temperature at fan inlet", "unit": "°R"},
    "sensor_2": {"name": "T24", "desc": "Total temperature at LPC outlet", "unit": "°R"},
    "sensor_3": {"name": "T30", "desc": "Total temperature at HPC outlet", "unit": "°R"},
    "sensor_4": {"name": "T50", "desc": "Total temperature at LPT outlet", "unit": "°R"},
    "sensor_5": {"name": "P2", "desc": "Pressure at fan inlet", "unit": "psia"},
    "sensor_6": {"name": "P15", "desc": "Total pressure in bypass-duct", "unit": "psia"},
    "sensor_7": {"name": "P30", "desc": "Total pressure at HPC outlet", "unit": "psia"},
    "sensor_8": {"name": "Nf", "desc": "Physical fan speed", "unit": "rpm"},
    "sensor_9": {"name": "Nc", "desc": "Physical core speed", "unit": "rpm"},
    "sensor_10": {"name": "epr", "desc": "Engine pressure ratio (P50/P2)", "unit": "--"},
    "sensor_11": {"name": "Ps30", "desc": "Static pressure at HPC outlet", "unit": "psia"},
    "sensor_12": {"name": "phi", "desc": "Ratio of fuel flow to Ps30", "unit": "pps/psia"},
    "sensor_13": {"name": "NRf", "desc": "Corrected fan speed", "unit": "rpm"},
    "sensor_14": {"name": "NRc", "desc": "Corrected core speed", "unit": "rpm"},
    "sensor_15": {"name": "BPR", "desc": "Bypass Ratio", "unit": "--"},
    "sensor_16": {"name": "farB", "desc": "Burner fuel-air ratio", "unit": "--"},
    "sensor_17": {"name": "htBleed", "desc": "Bleed Enthalpy", "unit": "--"},
    "sensor_18": {"name": "Nf_dmd", "desc": "Demanded fan speed", "unit": "rpm"},
    "sensor_19": {"name": "PCNfR_dmd", "desc": "Demanded corrected fan speed", "unit": "rpm"},
    "sensor_20": {"name": "W31", "desc": "HPT coolant bleed", "unit": "lbm/s"},
    "sensor_21": {"name": "W32", "desc": "LPT coolant bleed", "unit": "lbm/s"},
}

# RUL Labeling Configuration
DEFAULT_MAX_RUL = 125  # Piecewise linear clipping limit commonly adopted in NASA C-MAPSS literature
RUL_COL = "RUL"

# Sliding Window Feature Engineering
DEFAULT_WINDOW_SIZE = 30
WINDOW_SIZES_EXPERIMENT = [15, 30, 50]
WINDOW_STAT_METRICS = ["mean", "std", "min", "max", "latest", "linear_trend", "delta"]

# Constant Sensor Variance Threshold
VARIANCE_THRESHOLD = 1e-4

# Validation Split
VALIDATION_SPLIT_RATIO = 0.20
RANDOM_SEED = 42

# Maintenance Action Levels (Operational Decision Support Thresholds)
MAINTENANCE_LEVELS = {
    "HEALTHY": {"min": 76, "max": 9999, "color": "#10B981", "badge": "HEALTHY", "desc": "Normal flight operation"},
    "ATTENTION": {"min": 31, "max": 75, "color": "#F59E0B", "badge": "ATTENTION", "desc": "Schedule routine depot inspection"},
    "WARNING": {"min": 11, "max": 30, "color": "#F97316", "badge": "WARNING", "desc": "Component wear accelerated; plan overhaul"},
    "CRITICAL": {"min": 0, "max": 10, "color": "#EF4444", "badge": "CRITICAL", "desc": "Immediate maintenance required; ground engine"}
}

# Overestimation Risk Threshold
# When true RUL is critical/warning (<= 30 cycles), an overestimation of > 10 cycles is hazardous
OVERESTIMATION_HAZARD_THRESHOLD = 10.0

# Kagglehub Dataset Reference
KAGGLE_DATASET_ID = "behrad3d/nasa-cmaps"
