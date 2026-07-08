# Rideshare Demand Forecasting

Hourly rideshare demand forecasting across 6 city zones: synthetic ride-request
history with realistic zone-specific patterns (commuter rush, nightlife,
airport traffic), a RandomForest demand model evaluated on a time-based
holdout, and a 7-day-ahead forecast — served through a Streamlit dashboard.

NYC TLC publishes real for-hire-vehicle trip data, but at millions of rows a
month it's too heavy for a self-contained demo, so this generates synthetic
hourly demand instead (weekday commuter peaks, weekend nightlife spikes,
simulated weather effects, holiday demand surges, and a mild growth trend) —
no API keys, no signup, works immediately.

## Run locally

```bash
pip install -e .
python build.py
streamlit run app/dashboard.py
```

The dashboard also auto-builds on first run if the data files aren't present
yet (e.g. on a fresh Streamlit Cloud deploy) — no manual build step required.

## What it does

- **Demand Patterns** — hour-of-day x day-of-week heatmap per zone, zone
  comparison, and simulated weather effect on demand.
- **Model Performance** — actual vs. predicted daily rides on a trailing
  30-day holdout, feature importance, and a predicted-vs-actual scatter.
- **7-Day Forecast** — RandomForest-projected hourly demand for the week
  ahead, per zone.

## Stack

Python, pandas, NumPy, scikit-learn (RandomForestRegressor), DuckDB, Plotly,
Streamlit.

## Live dashboard

Static read-only preview (pre-computed data, no live filters): https://rideshare-demo-navy.vercel.app

Full interactive app (live filters, auto-rebuild): `<add your Streamlit Community Cloud URL here once deployed>`

`<add your Streamlit Community Cloud URL here once deployed>`
