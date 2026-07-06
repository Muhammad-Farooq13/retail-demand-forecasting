"""
Streamlit dashboard for interactive retail demand forecasting.

Run:
    streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.models.forecast import forecast_series

ARTIFACT_DIR = Path("models_store")
DATA_PATH = Path("data/raw/sales.csv")


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(ARTIFACT_DIR / "demand_model.joblib")
    encoder = joblib.load(ARTIFACT_DIR / "category_encoder.joblib")
    with open(ARTIFACT_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, encoder, metadata


@st.cache_data
def load_sales_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def main() -> None:
    st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
    st.title("📦 Retail Demand Forecasting")
    st.caption(
        "Gradient-boosted forecast vs. seasonal-naive baseline, per store/item. "
        "Data is synthetic — see the README for details."
    )

    if not (ARTIFACT_DIR / "demand_model.joblib").exists():
        st.error(
            "No trained model found. Run `python -m src.models.train` first "
            "(after `generate_data` and `build_features`)."
        )
        st.stop()

    model, encoder, metadata = load_model_artifacts()
    df = load_sales_data()

    with st.sidebar:
        st.header("Forecast settings")
        store_id = st.selectbox("Store", sorted(df["store_id"].unique()))
        item_options = df[df["store_id"] == store_id][
            ["item_id", "item_category"]
        ].drop_duplicates()
        item_id = st.selectbox(
            "Item",
            sorted(item_options["item_id"].unique()),
            format_func=lambda i: f"Item {i} ({item_options[item_options.item_id == i]['item_category'].iloc[0]})",
        )
        horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30, step=7)
        include_promo = st.checkbox(
            "Assume a promotion runs for the whole forecast window", value=False
        )

    series = df[(df.store_id == store_id) & (df.item_id == item_id)].sort_values("date")
    item_category = series["item_category"].iloc[0]
    last_price = float(series["price"].iloc[-1])

    promo_dates = None
    if include_promo:
        future_dates = pd.date_range(
            series["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        promo_dates = set(future_dates)

    forecast_df = forecast_series(
        history=series[["date", "units_sold"]],
        model=model,
        encoder=encoder,
        horizon_days=horizon,
        store_id=store_id,
        item_id=item_id,
        item_category=item_category,
        assumed_price=last_price,
        promo_dates=promo_dates,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model WAPE (test set)", f"{metadata['model_metrics']['wape_pct']}%")
    col2.metric("Baseline WAPE (seasonal-naive)", f"{metadata['baseline_metrics']['wape_pct']}%")
    col3.metric("Improvement vs. baseline", f"{metadata['wape_improvement_pct_vs_baseline']}%")
    col4.metric(
        f"Forecast total ({horizon}d)",
        f"{forecast_df['predicted_units_sold'].sum():,.0f} units",
    )

    st.subheader(f"Store {store_id} / Item {item_id} ({item_category})")

    history_plot = series[["date", "units_sold"]].tail(120).rename(columns={"units_sold": "value"})
    history_plot["type"] = "Historical"
    forecast_plot = forecast_df.rename(columns={"predicted_units_sold": "value"})
    forecast_plot["type"] = "Forecast"
    combined = pd.concat([history_plot, forecast_plot], ignore_index=True)

    chart_data = combined.pivot(index="date", columns="type", values="value")
    st.line_chart(chart_data)

    st.subheader("Forecast detail")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    with st.expander("Model details"):
        st.json(metadata)


if __name__ == "__main__":
    main()
