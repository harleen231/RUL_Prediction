"""
AeroGuard – train.py
Trains Random Forest and XGBoost models on the NASA C-MAPSS FD001 dataset,
evaluates them on a held-out test set, and saves all artifacts to models/.

Run this script once before launching the dashboard:
    python train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocessing import (
    load_data,
    calculate_train_rul,
    calculate_test_rul,
    split_engines,
    build_features,
    MODELS_DIR,
    MAX_RUL,
)

MODELS_DIR.mkdir(exist_ok=True)


# ─── NASA Scoring Function ─────────────────────────────────────────────────────
def nasa_score(y_true, y_pred):
    """
    Official NASA C-MAPSS scoring function.
    Penalises late predictions (overestimation) more heavily than early ones.
    Lower is better.
    """
    errors = y_pred - y_true
    scores = np.where(errors < 0,
                      np.exp(-errors / 13) - 1,
                      np.exp( errors / 10) - 1)
    return float(np.sum(scores))


# ─── Evaluation helper ────────────────────────────────────────────────────────
def evaluate(model, X, y, label=""):
    preds = model.predict(X).clip(0, MAX_RUL)
    mae   = mean_absolute_error(y, preds)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    r2    = r2_score(y, preds)
    ns    = nasa_score(y, preds)
    if label:
        print(f"  {label}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  NASA={ns:.2f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "nasa_score": ns, "predictions": preds.tolist()}


# ─── Main training pipeline ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AeroGuard – Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading NASA C-MAPSS FD001 dataset...")
    df_train_raw, df_test_raw, rul_series = load_data("FD001")
    print(f"      Train: {df_train_raw.shape}  |  Test: {df_test_raw.shape}")

    # 2. Compute RUL labels
    print("\n[2/5] Computing RUL labels...")
    df_train_rul = calculate_train_rul(df_train_raw)
    df_test_rul  = calculate_test_rul(df_test_raw, rul_series)

    # 3. Engine-wise train/validation split
    print("\n[3/5] Splitting engines into train / validation sets...")
    df_tr, df_val = split_engines(df_train_rul)
    print(f"      Train engines: {df_tr['engine_id'].nunique()}  |  Val engines: {df_val['engine_id'].nunique()}")

    # 4. Build features (scale + sliding window)
    print("\n[4/5] Building sliding-window features (window=30 cycles)...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler, sensor_stats = \
        build_features(df_tr, df_val, df_test_rul)
    print(f"      Feature count: {len(feature_names)}")
    print(f"      Train rows: {len(X_train)}  |  Val rows: {len(X_val)}  |  Test rows: {len(X_test)}")

    # 5. Train and evaluate models
    print("\n[5/5] Training models...")

    models_to_train = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    metrics = {}

    for name, model in models_to_train.items():
        print(f"\n  ── {name} ──")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"     Trained in {elapsed:.1f}s")

        val_metrics  = evaluate(model, X_val,  y_val,  label="Val ")
        test_metrics = evaluate(model, X_test, y_test, label="Test")

        metrics[name] = {
            "val_mae":        round(val_metrics["mae"],        3),
            "val_rmse":       round(val_metrics["rmse"],       3),
            "val_r2":         round(val_metrics["r2"],         4),
            "val_nasa_score": round(val_metrics["nasa_score"], 2),
            "test_mae":       round(test_metrics["mae"],        3),
            "test_rmse":      round(test_metrics["rmse"],       3),
            "test_r2":        round(test_metrics["r2"],         4),
            "test_nasa_score":round(test_metrics["nasa_score"], 2),
            "test_predictions": test_metrics["predictions"],
            "test_true":        y_test.tolist(),
        }

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"  {'Model':<18} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'NASA Score':>12}")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"  {name:<18} {m['test_mae']:>8.3f} {m['test_rmse']:>8.3f} {m['test_r2']:>8.4f} {m['test_nasa_score']:>12.2f}")
    print("=" * 60)

    # Identify best model by test RMSE
    best_name = min(metrics, key=lambda n: metrics[n]["test_rmse"])
    print(f"\n  Best model (lowest test RMSE): {best_name}")

    # 6. Save artifacts
    print("\n  Saving model artifacts to models/...")

    for name, model in models_to_train.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, MODELS_DIR / fname)
        print(f"    Saved: {fname}")

    joblib.dump(scaler,        MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(sensor_stats,  MODELS_DIR / "sensor_stats.pkl")

    # Also save the best model separately
    best_model = models_to_train[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    # Save metrics (drop raw prediction lists to keep JSON readable)
    metrics_json = {}
    for name, m in metrics.items():
        metrics_json[name] = {k: v for k, v in m.items()
                              if k not in ("test_predictions", "test_true")}
    metrics_json["best_model"]  = best_name
    metrics_json["feature_names"] = feature_names
    metrics_json["num_features"]  = len(feature_names)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print("\n  All artifacts saved successfully.")
    print(f"  → models/random_forest.pkl")
    print(f"  → models/xgboost.pkl")
    print(f"  → models/scaler.pkl")
    print(f"  → models/feature_names.pkl")
    print(f"  → models/sensor_stats.pkl")
    print(f"  → models/metrics.json")
    print("\nDone! You can now run: python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()
