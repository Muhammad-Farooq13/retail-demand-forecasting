"""
Synthetic daily retail sales dataset generator.

NOTE ON DATA PROVENANCE
------------------------
This build environment cannot reach Kaggle (e.g. the popular "Store Item
Demand Forecasting" or "M5" competition datasets) or the UCI repository, so
this module generates a realistic synthetic daily sales dataset instead.
Every effect below (weekly seasonality, annual seasonality, holiday spikes,
promotion lift, price elasticity, trend) is a well-documented retail-demand
pattern, not an arbitrary number, but the data itself is synthetic and is
labeled as such throughout this project -- it is not a substitute for a
real, sourced dataset.

Usage:
    python -m src.data.generate_data --n-days 730 --n-stores 5 --n-items 10
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ITEM_CATEGORIES = [
    ("produce", 1.0, 0.35),
    ("dairy", 0.8, 0.20),
    ("bakery", 0.6, 0.25),
    ("beverages", 1.2, 0.15),
    ("snacks", 0.9, 0.10),
]

US_FIXED_HOLIDAYS_MMDD = {
    (1, 1): 1.15,  # New Year's Day
    (7, 4): 1.25,  # July 4th
    (11, 27): 1.9,  # Thanksgiving week proxy (approx, boosts around it)
    (12, 24): 2.4,  # Christmas Eve
    (12, 25): 0.2,  # Christmas Day (stores closed / low)
    (12, 31): 1.6,  # New Year's Eve
}


@dataclass(frozen=True)
class GenerationConfig:
    n_days: int = 730
    n_stores: int = 5
    n_items: int = 10
    start_date: str = "2024-01-01"
    random_seed: int = 42


def _holiday_multiplier(date: pd.Timestamp) -> float:
    return US_FIXED_HOLIDAYS_MMDD.get((date.month, date.day), 1.0)


def generate_sales(config: GenerationConfig) -> pd.DataFrame:
    """Generate a synthetic, but statistically realistic, daily sales dataset."""
    rng = np.random.default_rng(config.random_seed)
    dates = pd.date_range(config.start_date, periods=config.n_days, freq="D")

    n_categories = len(ITEM_CATEGORIES)
    item_category = [ITEM_CATEGORIES[i % n_categories][0] for i in range(config.n_items)]
    item_base_level = rng.uniform(20, 120, size=config.n_items)
    item_annual_amp = np.array(
        [ITEM_CATEGORIES[i % n_categories][2] for i in range(config.n_items)]
    )
    item_base_price = rng.uniform(2.0, 15.0, size=config.n_items)

    store_multiplier = rng.uniform(0.7, 1.5, size=config.n_stores)

    rows = []
    day_index = np.arange(config.n_days)

    for store_id in range(1, config.n_stores + 1):
        for item_id in range(1, config.n_items + 1):
            idx = item_id - 1
            base = item_base_level[idx] * store_multiplier[store_id - 1]
            annual_amp = item_annual_amp[idx]
            price = item_base_price[idx]

            # Slow upward trend (store growth) ~10% over 2 years
            trend = 1.0 + 0.10 * (day_index / config.n_days)

            # Weekly seasonality: weekends higher for most categories
            dow = dates.dayofweek.values  # 0=Mon
            weekly_effect = 1.0 + 0.18 * np.isin(dow, [4, 5, 6])  # Fri/Sat/Sun bump

            # Annual seasonality (sinusoidal, category-dependent amplitude)
            day_of_year = dates.dayofyear.values
            annual_effect = 1.0 + annual_amp * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)

            # Holiday effects
            holiday_effect = np.array([_holiday_multiplier(d) for d in dates])

            # Promotions: random ~8% of days, boosts sales 30-70%
            is_promo = rng.binomial(1, 0.08, size=config.n_days)
            promo_lift = 1.0 + is_promo * rng.uniform(0.3, 0.7, size=config.n_days)

            # Price varies slightly day to day (small random walk around base)
            price_noise = rng.normal(0, 0.05, size=config.n_days).cumsum()
            daily_price = np.clip(price + price_noise, price * 0.7, price * 1.3)
            # Promo days get a discount
            daily_price = np.where(is_promo == 1, daily_price * 0.85, daily_price)

            expected_units = (
                base * trend * weekly_effect * annual_effect * holiday_effect * promo_lift
            )
            noise = rng.normal(1.0, 0.12, size=config.n_days)
            units_sold = np.clip(np.round(expected_units * noise), 0, None).astype(int)

            for i in range(config.n_days):
                rows.append(
                    {
                        "date": dates[i],
                        "store_id": store_id,
                        "item_id": item_id,
                        "item_category": item_category[idx],
                        "units_sold": int(units_sold[i]),
                        "price": round(float(daily_price[i]), 2),
                        "is_promo": int(is_promo[i]),
                        "is_holiday": int(holiday_effect[i] != 1.0),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    logger.info(
        "Generated %d rows: %d days x %d stores x %d items",
        len(df),
        config.n_days,
        config.n_stores,
        config.n_items,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic retail sales dataset")
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--n-stores", type=int, default=5)
    parser.add_argument("--n-items", type=int, default=10)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/raw/sales.csv")
    args = parser.parse_args()

    config = GenerationConfig(
        n_days=args.n_days,
        n_stores=args.n_stores,
        n_items=args.n_items,
        start_date=args.start_date,
        random_seed=args.random_seed,
    )
    df = generate_sales(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved dataset to %s (%.2f MB)",
        output_path,
        output_path.stat().st_size / (1024 * 1024),
    )


if __name__ == "__main__":
    main()
