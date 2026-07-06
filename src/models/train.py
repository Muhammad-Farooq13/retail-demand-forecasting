"""
Train and evaluate demand forecasting models.

Handles:
  - Chronological train/test split (last N days held out, no shuffling)
  - Seasonal-naive baseline (predict = same weekday last week) as the bar
    the ML model must clear -- reporting only the ML model's error without
    a baseline is a common way forecasting projects mislead themselves
  - Gradient Boosting Regressor on lag/rolling/calendar features
  - Metrics: MAE, RMSE, MAPE, and WAPE (weighted absolute percentage error,
    more robust than MAPE when some rows have near-zero actual demand)
  - MLflow experiment tracking
  - Persisted model + metadata for the Streamlit app

Usage:
    python -m src.models.train --features data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from src.features.build_features import NUMERIC_FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("models_store")
CATEGORICAL_COLS = ["item_category", "store_id", "item_id"]


def chronological_split(df: pd.DataFrame, test_days: int = 60):
    """Hold out the last `test_days` calendar days as the test set."""
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] > cutoff].copy()
    logger.info(
        "Chronological split -> train=%d rows (through %s), test=%d rows (%s to %s)",
        len(train_df),
        cutoff.date(),
        len(test_df),
        test_df["date"].min().date(),
        test_df["date"].max().date(),
    )
    return train_df, test_df


def seasonal_naive_baseline(df: pd.DataFrame) -> np.ndarray:
    """Predict this week's value using lag_7 (same weekday, prior week)."""
    return df["lag_7"].fillna(df["lag_7"].median()).values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE excluding zero-actual rows (undefined otherwise)
    nonzero_mask = y_true > 0
    mape = float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    )
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)
    return {
        "mae": round(float(mae), 3),
        "rmse": round(rmse, 3),
        "mape_pct": round(mape, 2),
        "wape_pct": round(wape, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train retail demand forecasting models")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlruns.db")
    parser.add_argument("--experiment-name", type=str, default="retail-demand-forecasting")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])

    train_df, test_df = chronological_split(df, test_days=args.test_days)

    # Fill any remaining NaNs (e.g. rolling_std with <2 points) with 0
    train_df[NUMERIC_FEATURES] = train_df[NUMERIC_FEATURES].fillna(0)
    test_df[NUMERIC_FEATURES] = test_df[NUMERIC_FEATURES].fillna(0)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = encoder.fit_transform(train_df[CATEGORICAL_COLS])
    test_cat = encoder.transform(test_df[CATEGORICAL_COLS])

    X_train = np.hstack([train_df[NUMERIC_FEATURES].values, train_cat])
    X_test = np.hstack([test_df[NUMERIC_FEATURES].values, test_cat])
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    with mlflow.start_run(run_name="seasonal_naive_baseline"):
        baseline_pred = seasonal_naive_baseline(test_df)
        baseline_metrics = compute_metrics(y_test, baseline_pred)
        mlflow.log_params({"model": "seasonal_naive"})
        mlflow.log_metrics(baseline_metrics)
        logger.info("Baseline (seasonal naive) -> %s", baseline_metrics)

    with mlflow.start_run(run_name="gradient_boosting_regressor"):
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        gbr_pred = model.predict(X_test)
        gbr_metrics = compute_metrics(y_test, gbr_pred)

        mlflow.log_params(
            {
                "model": "gradient_boosting_regressor",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
            }
        )
        mlflow.log_metrics(gbr_metrics)
        mlflow.sklearn.log_model(model, "gradient_boosting_regressor")
        logger.info("Gradient Boosting -> %s", gbr_metrics)

    improvement_pct = round(
        (baseline_metrics["wape_pct"] - gbr_metrics["wape_pct"])
        / baseline_metrics["wape_pct"]
        * 100,
        1,
    )
    logger.info(
        "Gradient Boosting improves WAPE by %.1f%% over seasonal-naive baseline "
        "(%.2f%% -> %.2f%%)",
        improvement_pct,
        baseline_metrics["wape_pct"],
        gbr_metrics["wape_pct"],
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "demand_model.joblib")
    joblib.dump(encoder, ARTIFACT_DIR / "category_encoder.joblib")
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "gradient_boosting_regressor",
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_COLS,
                "test_days": args.test_days,
                "baseline_metrics": baseline_metrics,
                "model_metrics": gbr_metrics,
                "wape_improvement_pct_vs_baseline": improvement_pct,
            },
            f,
            indent=2,
        )

    logger.info("Saved model artifacts to %s", ARTIFACT_DIR)
    print(
        json.dumps(
            {
                "baseline": baseline_metrics,
                "gradient_boosting": gbr_metrics,
                "improvement_pct": improvement_pct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
