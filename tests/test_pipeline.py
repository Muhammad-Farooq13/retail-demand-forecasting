"""Unit and integration tests for the retail demand forecasting platform."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import GenerationConfig, generate_sales
from src.features.build_features import (
    LAG_DAYS,
    ROLLING_WINDOWS,
    add_calendar_features,
    add_lag_and_rolling_features,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    config = GenerationConfig(n_days=200, n_stores=2, n_items=3, random_seed=7)
    return generate_sales(config)


class TestDataGeneration:
    def test_row_count(self, sample_df):
        assert len(sample_df) == 200 * 2 * 3

    def test_no_missing_values(self, sample_df):
        assert sample_df.isna().sum().sum() == 0

    def test_units_sold_non_negative(self, sample_df):
        assert (sample_df["units_sold"] >= 0).all()

    def test_price_positive(self, sample_df):
        assert (sample_df["price"] > 0).all()

    def test_promo_and_holiday_are_binary(self, sample_df):
        assert set(sample_df["is_promo"].unique()).issubset({0, 1})
        assert set(sample_df["is_holiday"].unique()).issubset({0, 1})

    def test_reproducible_with_same_seed(self):
        cfg = GenerationConfig(n_days=50, n_stores=1, n_items=2, random_seed=123)
        df1 = generate_sales(cfg)
        df2 = generate_sales(cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_weekend_effect_present(self, sample_df):
        """Sanity check that the synthetic weekly seasonality was actually applied."""
        df = sample_df.copy()
        df["dow"] = df["date"].dt.dayofweek
        weekday_avg = df[df["dow"].isin([0, 1, 2, 3])]["units_sold"].mean()
        weekend_avg = df[df["dow"].isin([4, 5, 6])]["units_sold"].mean()
        assert weekend_avg > weekday_avg


class TestFeatureEngineering:
    def test_lag_1_equals_previous_day_actual(self, sample_df):
        """lag_1 for day T must equal the actual units_sold on day T-1, no leakage."""
        df = add_calendar_features(sample_df)
        df = add_lag_and_rolling_features(df)
        one_series = (
            df[(df.store_id == 1) & (df.item_id == 1)].sort_values("date").reset_index(drop=True)
        )
        # Check a handful of rows where lag_1 is not NaN
        valid = one_series.dropna(subset=["lag_1"]).reset_index(drop=True)
        for i in range(1, min(5, len(valid))):
            row = valid.iloc[i]
            prev_actual = one_series[one_series["date"] == row["date"] - pd.Timedelta(days=1)]
            if len(prev_actual) == 1:
                assert abs(row["lag_1"] - prev_actual["units_sold"].iloc[0]) < 1e-9

    def test_rolling_mean_excludes_current_day(self, sample_df):
        """The rolling_mean_7 for day T must not include day T's own units_sold."""
        df = add_calendar_features(sample_df)
        df = add_lag_and_rolling_features(df)
        one_series = (
            df[(df.store_id == 1) & (df.item_id == 2)].sort_values("date").reset_index(drop=True)
        )
        row_idx = 10  # arbitrary row with enough history
        target_date = one_series.loc[row_idx, "date"]
        expected_window = one_series[
            (one_series["date"] < target_date)
            & (one_series["date"] >= target_date - pd.Timedelta(days=7))
        ]["units_sold"]
        actual = one_series.loc[row_idx, "rolling_mean_7"]
        assert abs(actual - expected_window.mean()) < 1e-6

    def test_warmup_rows_dropped(self, sample_df):
        from src.features.build_features import build_feature_frame

        sample_df.to_csv("/tmp/_test_sales.csv", index=False)
        features = build_feature_frame("/tmp/_test_sales.csv")
        max_window = max(max(LAG_DAYS), max(ROLLING_WINDOWS))
        n_series = sample_df.groupby(["store_id", "item_id"]).ngroups
        expected_rows = len(sample_df) - n_series * max_window
        assert len(features) == expected_rows

    def test_no_nan_in_lag_28_after_warmup(self, sample_df):
        from src.features.build_features import build_feature_frame

        sample_df.to_csv("/tmp/_test_sales.csv", index=False)
        features = build_feature_frame("/tmp/_test_sales.csv")
        assert features["lag_28"].isna().sum() == 0

    def test_calendar_features_correct(self, sample_df):
        df = add_calendar_features(sample_df)
        known_monday = pd.Timestamp("2024-01-01")  # 2024-01-01 is a Monday
        row = df[df["date"] == known_monday].iloc[0]
        assert row["day_of_week"] == 0
        assert row["is_weekend"] == 0


class TestForecasting:
    def test_recursive_forecast_produces_correct_horizon_length(self, sample_df):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import OrdinalEncoder

        from src.features.build_features import (
            CATEGORICAL_FEATURES,  # noqa: F401
            NUMERIC_FEATURES,
            TARGET,
            build_feature_frame,
        )
        from src.models.forecast import forecast_series

        sample_df.to_csv("/tmp/_test_sales_forecast.csv", index=False)
        features = build_feature_frame("/tmp/_test_sales_forecast.csv")
        features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].fillna(0)

        cat_cols = ["item_category", "store_id", "item_id"]
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoded = encoder.fit_transform(features[cat_cols])
        X = np.hstack([features[NUMERIC_FEATURES].values, cat_encoded])
        y = features[TARGET].values

        model = GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0)
        model.fit(X, y)

        raw = sample_df[(sample_df.store_id == 1) & (sample_df.item_id == 1)].sort_values("date")
        forecast = forecast_series(
            history=raw[["date", "units_sold"]],
            model=model,
            encoder=encoder,
            horizon_days=10,
            store_id=1,
            item_id=1,
            item_category=raw["item_category"].iloc[0],
            assumed_price=float(raw["price"].iloc[-1]),
        )
        assert len(forecast) == 10
        assert (forecast["predicted_units_sold"] >= 0).all()
        assert list(forecast["date"]) == sorted(forecast["date"])

    def test_forecast_raises_with_insufficient_history(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import OrdinalEncoder

        from src.models.forecast import forecast_series

        short_history = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "units_sold": [10, 12, 11, 9, 14],
            }
        )
        model = GradientBoostingRegressor()
        encoder = OrdinalEncoder()
        with pytest.raises(ValueError):
            forecast_series(
                history=short_history,
                model=model,
                encoder=encoder,
                horizon_days=5,
                store_id=1,
                item_id=1,
                item_category="produce",
                assumed_price=5.0,
            )
