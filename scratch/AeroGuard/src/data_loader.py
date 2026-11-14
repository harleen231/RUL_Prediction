"""
AeroGuard Data Loader Module
Handles downloading, caching, file discovery, schema assignment,
and integrity validation for NASA C-MAPSS dataset subsets (FD001 - FD004).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR,
    AVAILABLE_SUBSETS,
    DEFAULT_SUBSET,
    ALL_RAW_COLS,
    KAGGLE_DATASET_ID,
)

logger = logging.getLogger("AeroGuard.DataLoader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def download_dataset_via_kagglehub(target_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Attempts to download the NASA C-MAPSS dataset using kagglehub.
    Gracefully handles authentication or network errors.
    """
    target = target_dir or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        logger.info("Initiating dataset download via kagglehub ('%s')...", KAGGLE_DATASET_ID)
        download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
        logger.info("Kagglehub download succeeded at: %s", download_path)

        # Look for CMaps subfolder or direct txt files
        candidates = list(download_path.rglob("train_FD001.txt"))
        if candidates:
            source_dir = candidates[0].parent
            logger.info("Found dataset text files in: %s", source_dir)
            # Copy into target if different
            for txt_file in source_dir.glob("*.txt"):
                dest_file = target / txt_file.name
                if not dest_file.exists():
                    dest_file.write_bytes(txt_file.read_bytes())
            return target
        return download_path
    except Exception as exc:
        logger.warning(
            "Automatic Kagglehub download failed: %s. AeroGuard will attempt to load local cached files.", exc
        )
        return None


def locate_dataset_files(subset: str = DEFAULT_SUBSET, data_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Locates the train, test, and RUL truth files for a specific subset (e.g., FD001).
    Searches in the configured DATA_DIR and user cache.
    """
    if subset not in AVAILABLE_SUBSETS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {AVAILABLE_SUBSETS}")

    search_dirs = [data_dir or DATA_DIR]
    
    # Check kagglehub cache directory as secondary fallback
    user_home = Path.home()
    kagglehub_cache = user_home / ".cache" / "kagglehub" / "datasets" / "behrad3d" / "nasa-cmaps"
    if kagglehub_cache.exists():
        for cmaps_dir in kagglehub_cache.rglob("CMaps"):
            if cmaps_dir.is_dir():
                search_dirs.append(cmaps_dir)

    train_name = f"train_{subset}.txt"
    test_name = f"test_{subset}.txt"
    rul_name = f"RUL_{subset}.txt"

    train_path, test_path, rul_path = None, None, None

    for directory in search_dirs:
        t_p = directory / train_name
        te_p = directory / test_name
        r_p = directory / rul_name
        if t_p.exists() and te_p.exists() and r_p.exists():
            train_path, test_path, rul_path = t_p, te_p, r_p
            break

    if not (train_path and test_path and rul_path):
        # Try downloading if not found
        logger.info("Dataset files for %s not found in local directories. Attempting download...", subset)
        download_dataset_via_kagglehub(data_dir or DATA_DIR)
        
        # Check target again
        t_p = (data_dir or DATA_DIR) / train_name
        te_p = (data_dir or DATA_DIR) / test_name
        r_p = (data_dir or DATA_DIR) / rul_name
        if t_p.exists() and te_p.exists() and r_p.exists():
            return t_p, te_p, r_p

        raise FileNotFoundError(
            f"Could not locate C-MAPSS dataset files for subset '{subset}'.\n"
            f"Expected files:\n"
            f"  - {train_name}\n"
            f"  - {test_name}\n"
            f"  - {rul_name}\n"
            f"Please ensure network access for kagglehub, or manually download the NASA C-MAPSS dataset "
            f"and place the text files inside: '{data_dir or DATA_DIR}'."
        )

    return train_path, test_path, rul_path


def load_raw_dataset(
    subset: str = DEFAULT_SUBSET,
    data_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads train, test, and ground truth RUL data for the specified C-MAPSS subset.
    Assigns standard domain column names and verifies record integrity.

    Returns:
        (df_train, df_test, s_rul_truth)
    """
    train_path, test_path, rul_path = locate_dataset_files(subset, data_dir)
    logger.info("Loading C-MAPSS %s from:\n  Train: %s\n  Test:  %s\n  RUL:   %s", subset, train_path, test_path, rul_path)

    # Read space-delimited text files
    df_train = pd.read_csv(train_path, sep=r"\s+", header=None)
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None)
    df_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)

    # Trim any possible trailing extra whitespace columns
    if df_train.shape[1] > len(ALL_RAW_COLS):
        df_train = df_train.iloc[:, :len(ALL_RAW_COLS)]
    if df_test.shape[1] > len(ALL_RAW_COLS):
        df_test = df_test.iloc[:, :len(ALL_RAW_COLS)]

    df_train.columns = ALL_RAW_COLS
    df_test.columns = ALL_RAW_COLS

    # Convert engine_id and cycle to integer
    df_train["engine_id"] = df_train["engine_id"].astype(int)
    df_train["cycle"] = df_train["cycle"].astype(int)
    df_test["engine_id"] = df_test["engine_id"].astype(int)
    df_test["cycle"] = df_test["cycle"].astype(int)

    # RUL series: 1-indexed to match engine_id 1..N
    s_rul_truth = df_rul.iloc[:, 0].copy()
    s_rul_truth.index = np.arange(1, len(s_rul_truth) + 1)
    s_rul_truth.name = "true_rul"

    # Integrity verification
    train_nulls = df_train.isnull().sum().sum()
    test_nulls = df_test.isnull().sum().sum()
    train_dups = df_train.duplicated(subset=["engine_id", "cycle"]).sum()
    test_dups = df_test.duplicated(subset=["engine_id", "cycle"]).sum()

    logger.info(
        "Loaded %s: Train shape=%s (%d engines), Test shape=%s (%d engines), Ground-Truth RUL count=%d",
        subset,
        df_train.shape,
        df_train["engine_id"].nunique(),
        df_test.shape,
        df_test["engine_id"].nunique(),
        len(s_rul_truth)
    )

    if train_nulls > 0 or test_nulls > 0:
        logger.warning("Dataset contains missing values: Train nulls=%d, Test nulls=%d", train_nulls, test_nulls)
    if train_dups > 0 or test_dups > 0:
        logger.warning("Dataset contains duplicate engine-cycle records: Train dups=%d, Test dups=%d", train_dups, test_dups)

    return df_train, df_test, s_rul_truth


if __name__ == "__main__":
    train_df, test_df, rul_truth = load_raw_dataset("FD001")
    print(f"Dataset successfully loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
