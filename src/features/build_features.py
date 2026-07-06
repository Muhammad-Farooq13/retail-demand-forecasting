"""
Feature engineering for retail demand forecasting.

Builds lag, rolling-window, and calendar features per (store_id, item_id)
series. All lag/rolling features are shifted so that the feature for date T
only uses information available up to and including date T-1 -- otherwise
the model would "see the future" at inference time, which is the single
most common bug in time-series forecasting pipelines.

Usage:
    python -m src.features.build_features --input data/raw/sales.csv \
        --output data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]

TARGET = "units_sold"
CATEGORICAL_FEATURES = ["item_category", "store_id", "item_id", "day_of_week", "month"]
NUMERIC_FEATURES = (
    [f"lag_{lag}" for lag in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + [
        "price",
        "is_promo",
        "is_holiday",
        "day_of_week",
        "month",
        "is_weekend",
        "days_since_start",
    ]
)
FEATURE_COLUMNS = NUMERIC_FEATURES + ["item_category", "store_id", "item_id"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag and rolling features per (store_id, item_id) series.

    IMPORTANT: rolling means/stds are computed on the *shifted* series
    (shift(1) applied before rolling), so the rolling window for date T
    covers [T-window, T-1] -- it never includes day T's own sales.
    """
    df = df.sort_values(["store_id", "item_id", "date"]).copy()
    grp = df.groupby(["store_id", "item_id"])[TARGET]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df.groupby(["store_id", "item_id"])[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby(["store_id", "item_id"])[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
        )
    del shifted

    return df.sort_values("date").reset_index(drop=True)


def build_feature_frame(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["date"])
    logger.info("Loaded raw data: %d rows", len(df))
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    # Drop the initial warm-up rows per series where the longest lag/rolling
    # window has no history yet (can't be used for training or fair eval).
    max_window = max(max(LAG_DAYS), max(ROLLING_WINDOWS))
    df["series_day_index"] = df.groupby(["store_id", "item_id"]).cumcount()
    before = len(df)
    df = df[df["series_day_index"] >= max_window].drop(columns="series_day_index")
    logger.info(
        "Dropped %d warm-up rows (insufficient lag history); %d rows remain",
        before - len(df),
        len(df),
    )
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready forecasting features")
    parser.add_argument("--input", type=str, default="data/raw/sales.csv")
    parser.add_argument("--output", type=str, default="data/processed/features.parquet")
    args = parser.parse_args()

    df = build_feature_frame(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved feature frame to %s (%d rows, %d cols)",
        output_path,
        len(df),
        df.shape[1],
    )


if __name__ == "__main__":
    main()
