"""Rideshare Demand Forecasting dashboard.

Sections: KPIs, demand patterns (hour x day-of-week heatmap), model
performance (actual vs. predicted, feature importance), and a 7-day forecast.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from app.theme import THEME_CSS, apply_plotly_theme, plotly_chart
from ingest.config import FORECAST_PATH, MODEL_PATH, RIDES_PATH, SCORED_PATH, ZONES

st.set_page_config(page_title="Rideshare Demand Forecasting", page_icon="\U0001F697", layout="wide")
apply_plotly_theme(pio)
st.markdown(THEME_CSS, unsafe_allow_html=True)


def build_if_missing() -> bool:
    if RIDES_PATH.exists() and MODEL_PATH.exists() and FORECAST_PATH.exists():
        return False
    with st.spinner("First run: generating ride history, training the demand model, and forecasting..."):
        import build as build_module

        build_module.main()
    return True


@st.cache_data
def load_data():
    rides = pd.read_parquet(RIDES_PATH)
    scored = pd.read_parquet(SCORED_PATH)
    forecast = pd.read_parquet(FORECAST_PATH)
    bundle = joblib.load(MODEL_PATH)
    return rides, scored, forecast, bundle


def main() -> None:
    st.title("Rideshare Demand Forecasting")
    st.caption(
        "Synthetic hourly ride-request history across 6 city zones: demand "
        "patterns, a RandomForest demand model, and a 7-day forecast."
    )

    if build_if_missing():
        st.cache_data.clear()
    rides, scored, forecast, bundle = load_data()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Zones", f"{rides['zone'].nunique()}")
    k2.metric("Avg rides / hour", f"{rides['rides'].mean():.1f}")
    k3.metric("Model MAE", f"{bundle['mae']:.2f} rides/hr")
    k4.metric("Model R²", f"{bundle['r2']:.3f}")
    st.divider()

    tabs = st.tabs(["Demand Patterns", "Model Performance", "7-Day Forecast"])

    with tabs[0]:
        zone = st.selectbox("Zone", ["All zones"] + ZONES)
        scope = rides if zone == "All zones" else rides[rides["zone"] == zone]

        pivot = scope.pivot_table(index="day_of_week", columns="hour", values="rides", aggfunc="mean")
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = pivot.reindex(range(7))
        fig = px.imshow(
            pivot, aspect="auto", color_continuous_scale="Oranges",
            labels=dict(x="Hour of day", y="Day of week", color="Avg rides"),
            y=dow_labels, title=f"Average hourly demand — {zone}",
        )
        plotly_chart(fig, use_container_width=True)

        by_zone = rides.groupby("zone")["rides"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(by_zone, x="zone", y="rides", title="Average hourly rides by zone")
        fig.update_yaxes(title="Avg rides / hour")
        plotly_chart(fig, use_container_width=True)

        rain_effect = rides.groupby("is_raining")["rides"].mean().reset_index()
        rain_effect["is_raining"] = rain_effect["is_raining"].map({True: "Rain", False: "No rain"})
        fig = px.bar(rain_effect, x="is_raining", y="rides", title="Weather effect on average demand")
        fig.update_yaxes(title="Avg rides / hour")
        plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.caption("Evaluated on a trailing 30-day time-based holdout (not a random split, to avoid leaking future data into training).")
        daily = scored.groupby("date")[["rides", "predicted_rides"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["rides"], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["predicted_rides"], name="Predicted",
                                  mode="lines", line=dict(dash="dash")))
        fig.update_layout(title="Daily total rides: actual vs. predicted (holdout period)", height=380,
                           legend=dict(orientation="h", y=-0.2))
        plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            imp = bundle["feature_importances"].reset_index()
            imp.columns = ["feature", "importance"]
            fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                         title="Feature importance")
            plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(scored.sample(min(2000, len(scored)), random_state=1),
                              x="rides", y="predicted_rides", opacity=0.4,
                              title="Predicted vs. actual (hourly, holdout sample)")
            max_v = max(scored["rides"].max(), scored["predicted_rides"].max())
            fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dot", color="#9C9791"))
            plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        zone = st.selectbox("Zone to forecast", ZONES, key="forecast_zone")
        fzone = forecast[forecast["zone"] == zone]
        daily_forecast = fzone.groupby("date")["predicted_rides"].sum().reset_index()
        fig = px.bar(daily_forecast, x="date", y="predicted_rides", title=f"7-day forecast — {zone}")
        fig.update_yaxes(title="Forecasted rides")
        plotly_chart(fig, use_container_width=True)

        fig = px.line(fzone, x="timestamp", y="predicted_rides", title=f"Hourly forecast detail — {zone}")
        plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
