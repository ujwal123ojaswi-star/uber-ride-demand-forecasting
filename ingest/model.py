"""Feature engineering + demand model for hourly ride requests.

A RandomForestRegressor predicts hourly ride counts per zone from calendar +
weather features. Evaluated on a trailing time-based holdout (not a random
split, since this is time-series data and random splits would leak future
information into training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ingest.config import FORECAST_HORIZON_DAYS, RANDOM_SEED, TEST_HOLDOUT_DAYS, ZONES

FEATURES = ["hour", "day_of_week", "is_weekend", "is_holiday", "temp_f", "is_raining", "zone_code"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zone_code"] = df["zone"].astype("category").cat.codes
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_raining"] = df["is_raining"].astype(int)
    return df


def train_and_evaluate(rides: pd.DataFrame) -> dict:
    df = add_features(rides)
    cutoff = df["date"].max() - pd.Timedelta(days=TEST_HOLDOUT_DAYS)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(train[FEATURES], train["rides"])

    pred_test = np.clip(model.predict(test[FEATURES]), 0, None)
    mae = mean_absolute_error(test["rides"], pred_test)
    r2 = r2_score(test["rides"], pred_test)

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    test_scored = test.copy()
    test_scored["predicted_rides"] = pred_test

    zone_map = dict(enumerate(rides["zone"].astype("category").cat.categories))

    return {
        "model": model,
        "mae": float(mae),
        "r2": float(r2),
        "feature_importances": importances,
        "test_scored": test_scored,
        "zone_code_map": zone_map,
    }


def forecast_next_week(rides: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """Project the next FORECAST_HORIZON_DAYS using each zone's recent
    (trailing 28-day) average weather/holiday rate as a simple assumption —
    the point here is demonstrating the demand-forecasting pipeline end to
    end, not weather forecasting."""
    last_date = rides["date"].max()
    recent = rides[rides["date"] > last_date - pd.Timedelta(days=28)]
    avg_temp = recent["temp_f"].mean()
    rain_rate = recent["is_raining"].mean()

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON_DAYS, freq="D")
    zone_codes = dict(zip(rides["zone"].astype("category").cat.categories,
                           range(rides["zone"].astype("category").cat.categories.size)))

    rows = []
    for zone in ZONES:
        for day in future_dates:
            for hour in range(24):
                rows.append(
                    {
                        "zone": zone,
                        "timestamp": day + pd.Timedelta(hours=hour),
                        "date": day,
                        "hour": hour,
                        "day_of_week": day.dayofweek,
                        "is_weekend": int(day.dayofweek >= 5),
                        "is_holiday": 0,
                        "temp_f": avg_temp,
                        "is_raining": int(rain_rate > 0.5),
                        "zone_code": zone_codes[zone],
                    }
                )
    future = pd.DataFrame(rows)
    future["predicted_rides"] = np.clip(model.predict(future[FEATURES]), 0, None)
    return future
