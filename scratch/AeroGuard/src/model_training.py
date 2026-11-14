"""
AeroGuard Model Training & Benchmarking Pipeline
Focused, streamlined multi-model regression architectures:
1. XGBoost Regressor
2. Random Forest Regressor
3. Linear Regression
Uses leakage-free engine-wise validation splitting and automated model serialization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR,
    REPORTS_DIR,
    VALIDATION_SPLIT_RATIO,
    RANDOM_SEED,
    RUL_COL
)
from src.evaluation import evaluate_predictions, nasa_scoring_function

logger = logging.getLogger("AeroGuard.ModelTraining")


def split_engine_wise(
    df: pd.DataFrame,
    val_ratio: float = VALIDATION_SPLIT_RATIO,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Performs leakage-free train/validation split at the engine level.
    Ensures all cycles for a given engine remain strictly in either train or val.
    """
    unique_engines = np.array(sorted(df["engine_id"].unique()))
    rng = np.random.RandomState(random_seed)
    
    n_val = int(np.round(len(unique_engines) * val_ratio))
    val_engines = sorted(rng.choice(unique_engines, size=n_val, replace=False).tolist())
    train_engines = sorted([e for e in unique_engines if e not in val_engines])

    df_train_split = df[df["engine_id"].isin(train_engines)].copy().reset_index(drop=True)
    df_val_split = df[df["engine_id"].isin(val_engines)].copy().reset_index(drop=True)

    logger.info(
        "Engine-wise split completed: Train=%d engines (%d rows), Val=%d engines (%d rows)",
        len(train_engines), len(df_train_split), len(val_engines), len(df_val_split)
    )
    return df_train_split, df_val_split, train_engines, val_engines


def get_model_registry() -> Dict[str, object]:
    """
    Initializes the 3 core benchmark regression models:
    - XGBoost
    - Random Forest
    - Linear Regression
    """
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "Linear Regression": LinearRegression()
    }


def train_and_benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, object], str]:
    """
    Trains XGBoost, Random Forest, and Linear Regression, benchmarks them,
    identifies the best model, and serializes trained artifacts.
    """
    m_dir = models_dir or MODELS_DIR
    r_dir = reports_dir or REPORTS_DIR
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    registry = get_model_registry()
    results = []
    trained_models = {}

    feature_names = list(X_train.columns)
    logger.info("Training features count: %d across %d core models", len(feature_names), len(registry))

    best_val_rmse = float("inf")
    best_model_name = "XGBoost"

    for name, model in registry.items():
        logger.info("Training architecture: %s ...", name)
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Validation set evaluation
        val_preds = model.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_preds, model_name=name)

        # Test set evaluation
        test_preds = model.predict(X_test)
        test_metrics = evaluate_predictions(y_test, test_preds, model_name=name)

        # Track best model based on validation RMSE
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_model_name = name

        # Serialize individual model
        slug = name.lower().replace(" ", "_")
        model_file = m_dir / f"{slug}.joblib"
        joblib.dump(model, model_file)

        # Record metrics row
        results.append({
            "model_name": name,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_nasa_score": val_metrics["nasa_score"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_nasa_score": test_metrics["nasa_score"],
            "test_overestimate_pct": test_metrics["overestimation_pct"],
            "test_critical_overestimates": test_metrics["critical_overestimate_count"],
        })

    df_comparison = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)
    logger.info("Model benchmarking complete. Best model by validation: %s", best_model_name)

    # Save best model reference
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, m_dir / "best_model.joblib")

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "validation_rmse": float(best_val_rmse),
        "models_benchmarked": list(registry.keys()),
    }
    with open(m_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save comparison CSV
    df_comparison.to_csv(r_dir / "model_comparison.csv", index=False)
    logger.info("Saved model comparison table to %s", r_dir / "model_comparison.csv")

    return df_comparison, trained_models, best_model_name
