"""
Unit tests for Model Evaluation and Inference
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.evaluation import nasa_scoring_function, evaluate_predictions
from src.prediction import classify_maintenance_status, AeroGuardPredictor
from config import MODELS_DIR


def test_nasa_scoring_function_asymmetry():
    """
    NASA Score penalizes late predictions (d > 0) exponentially higher than early (d < 0).
    """
    y_true = np.array([50.0])
    
    # 5 cycles underestimation: d = -5
    # score = exp(5/13) - 1 ≈ 0.469
    y_pred_early = np.array([45.0])
    score_early = nasa_scoring_function(y_true, y_pred_early)

    # 5 cycles overestimation: d = +5
    # score = exp(5/10) - 1 ≈ 0.648
    y_pred_late = np.array([55.0])
    score_late = nasa_scoring_function(y_true, y_pred_late)

    assert score_late > score_early, "NASA score must penalize overestimation more harshly than underestimation."


def test_maintenance_classification_thresholds():
    healthy = classify_maintenance_status(90.0)
    assert healthy["category"] == "HEALTHY"

    attention = classify_maintenance_status(50.0)
    assert attention["category"] == "ATTENTION"

    warning = classify_maintenance_status(20.0)
    assert warning["category"] == "WARNING"

    critical = classify_maintenance_status(5.0)
    assert critical["category"] == "CRITICAL"


def test_predictor_loaded_and_predicts():
    """Verifies that saved models can be loaded and perform inference."""
    if not (MODELS_DIR / "best_model.joblib").exists():
        pytest.skip("Models not yet trained.")

    predictor = AeroGuardPredictor()
    assert predictor.model is not None
    assert len(predictor.feature_names) > 0
    # Verify core 3 models
    assert "XGBoost" in predictor.models
    assert "Random Forest" in predictor.models
    assert "Linear Regression" in predictor.models


def test_simulator_inputs_prediction():
    """Verifies that the interactive simulator predicts RUL across all 3 models from sensor inputs."""
    if not (MODELS_DIR / "best_model.joblib").exists():
        pytest.skip("Models not yet trained.")

    from src.prediction import SIMULATOR_PRESETS, NOMINAL_HEALTHY_BASELINES, FAILURE_ENDPOINTS
    predictor = AeroGuardPredictor()

    # Test brand new engine preset
    res_new = predictor.predict_simulator_inputs(NOMINAL_HEALTHY_BASELINES, cycle=15)
    assert res_new["status"]["category"] == "HEALTHY"
    assert res_new["primary_pred"] > 75.0
    assert "XGBoost" in res_new["model_predictions"]
    assert "Random Forest" in res_new["model_predictions"]
    assert "Linear Regression" in res_new["model_predictions"]

    # Test critical failure preset
    res_fail = predictor.predict_simulator_inputs(FAILURE_ENDPOINTS, cycle=220)
    assert res_fail["status"]["category"] == "CRITICAL"
    assert res_fail["primary_pred"] <= 15.0
