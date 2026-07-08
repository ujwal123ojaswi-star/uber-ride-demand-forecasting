"""Build the full pipeline end to end:  python build.py

Generates synthetic hourly ride-request history across 6 zones, trains a
demand model (RandomForest, time-based holdout), and produces a 7-day-ahead
forecast. After this, launch the dashboard with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import joblib

from ingest.config import DATA_DIR, FORECAST_PATH, MODEL_PATH, RIDES_PATH, SCORED_PATH
from ingest.generate_data import generate_hourly_rides
from ingest.model import forecast_next_week, train_and_evaluate


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("==> [1/3] Generating synthetic hourly ride-request history")
    rides = generate_hourly_rides()
    rides.to_parquet(RIDES_PATH, index=False)
    print(f"[data] {len(rides):,} hourly rows across {rides['zone'].nunique()} zones")

    print("\n==> [2/3] Training demand model (time-based holdout)")
    result = train_and_evaluate(rides)
    print(f"[model] MAE {result['mae']:.2f} rides/hour, R2 {result['r2']:.3f}")
    result["test_scored"].to_parquet(SCORED_PATH, index=False)
    joblib.dump(
        {
            "model": result["model"],
            "mae": result["mae"],
            "r2": result["r2"],
            "feature_importances": result["feature_importances"],
            "zone_code_map": result["zone_code_map"],
        },
        MODEL_PATH,
    )

    print("\n==> [3/3] Forecasting next 7 days")
    forecast = forecast_next_week(rides, result["model"])
    forecast.to_parquet(FORECAST_PATH, index=False)
    print(f"[forecast] {len(forecast):,} hourly predictions")

    print("\nDone. Launch the dashboard with:")
    print("    streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
