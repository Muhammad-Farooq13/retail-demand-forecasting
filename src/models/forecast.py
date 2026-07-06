"""
Recursive multi-step demand forecasting.

Lag/rolling features depend on prior days' actual sales. To forecast N days
ahead, each day's prediction must feed into the lag features for the next
day's prediction -- a naive "predict all N days independently using only
known history" approach silently produces garbage beyond lag_1's horizon.
This module implements that recursive loop explicitly and correctly.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.features.build_features import LAG_DAYS, NUMERIC_FEATURES, ROLLING_WINDOWS

US_FIXED_HOLIDAYS_MMDD = {
    (1, 1): True,
    (7, 4): True,
    (11, 27): True,
    (12, 24): True,
    (12, 25): True,
    (12, 31): True,
}


def _is_holiday(date: pd.Timestamp) -> int:
    return int((date.month, date.day) in US_FIXED_HOLIDAYS_MMDD)


def forecast_series(
    history: pd.DataFrame,
    model,
    encoder,
    horizon_days: int,
    store_id: int,
    item_id: int,
    item_category: str,
    assumed_price: float,
    promo_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """
    Recursively forecast `horizon_days` ahead for a single (store_id, item_id) series.

    Args:
        history: past rows for this exact (store_id, item_id), sorted by date,
            containing at least `date` and `units_sold` columns, with enough
            history to cover the largest lag/rolling window.
        model: trained regressor with .predict(X) -> array
        encoder: fitted OrdinalEncoder for [item_category, store_id, item_id]
        horizon_days: number of future days to forecast
        assumed_price: price to assume for future days (last known price by default)
        promo_dates: set of future dates that are promo days (default: none)

    Returns:
        DataFrame with columns [date, predicted_units_sold]
    """
    promo_dates = promo_dates or set()
    series = history.sort_values("date").reset_index(drop=True).copy()
    max_window = max(max(LAG_DAYS), max(ROLLING_WINDOWS))
    if len(series) < max_window:
        raise ValueError(
            f"Need at least {max_window} days of history to forecast, got {len(series)}"
        )

    last_date = series["date"].max()
    predictions = []

    for step in range(1, horizon_days + 1):
        target_date = last_date + timedelta(days=step)
        units = series["units_sold"].values

        lag_features = {
            f"lag_{lag}": units[-lag] if len(units) >= lag else np.nan for lag in LAG_DAYS
        }
        rolling_features = {}
        for window in ROLLING_WINDOWS:
            window_vals = units[-window:]
            rolling_features[f"rolling_mean_{window}"] = float(np.mean(window_vals))
            rolling_features[f"rolling_std_{window}"] = (
                float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
            )

        is_promo = int(target_date in promo_dates)
        is_holiday = _is_holiday(target_date)
        day_of_week = target_date.dayofweek
        month = target_date.month
        is_weekend = int(day_of_week in [4, 5, 6])
        days_since_start = (target_date - series["date"].min()).days

        row_numeric = {
            **lag_features,
            **rolling_features,
            "price": assumed_price,
            "is_promo": is_promo,
            "is_holiday": is_holiday,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "days_since_start": days_since_start,
        }
        numeric_vector = np.array([[row_numeric[col] for col in NUMERIC_FEATURES]])
        numeric_vector = np.nan_to_num(numeric_vector, nan=0.0)

        cat_df = pd.DataFrame(
            [[item_category, store_id, item_id]],
            columns=["item_category", "store_id", "item_id"],
        )
        cat_vector = encoder.transform(cat_df)
        X = np.hstack([numeric_vector, cat_vector])

        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)
        predictions.append({"date": target_date, "predicted_units_sold": round(pred, 1)})

        # Feed the prediction back into the series so the NEXT step's lag/rolling
        # features can use it -- this is what makes it "recursive" forecasting.
        series = pd.concat(
            [series, pd.DataFrame([{"date": target_date, "units_sold": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)
