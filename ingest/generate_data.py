"""Synthetic hourly rideshare demand generator.

NYC TLC publishes real for-hire-vehicle trip data, but at millions of rows a
month it's too heavy for a self-contained demo project — so this generates
realistic synthetic hourly demand per zone instead, with zone-specific
hour-of-day profiles (commuter rush vs. nightlife vs. airport), weekday/
weekend effects, simulated weather, holiday spikes, and a mild growth trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ingest.config import N_DAYS_HISTORY, RANDOM_SEED, ZONES

# Base hourly shape (24 values, hour 0-23) per zone archetype. These are
# relative weights, renormalized per zone below.
HOUR_PROFILES = {
    "Downtown": [3, 2, 1, 1, 1, 2, 5, 9, 10, 7, 6, 7, 8, 7, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4],
    "Airport": [4, 3, 2, 2, 3, 5, 8, 9, 7, 6, 6, 6, 6, 6, 6, 7, 8, 9, 8, 7, 6, 5, 5, 4],
    "University": [1, 1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 7, 7, 6, 6, 6, 7, 7, 6, 5, 4, 3, 2],
    "Suburbs-North": [1, 1, 1, 1, 1, 2, 6, 9, 6, 3, 2, 2, 3, 3, 3, 4, 6, 9, 8, 5, 3, 2, 2, 1],
    "Suburbs-South": [1, 1, 1, 1, 1, 2, 6, 9, 6, 3, 2, 2, 3, 3, 3, 4, 6, 9, 8, 5, 3, 2, 2, 1],
    "Nightlife-District": [8, 9, 6, 3, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9, 9, 9, 9],
}

ZONE_BASE_LEVEL = {
    "Downtown": 42,
    "Airport": 30,
    "University": 22,
    "Suburbs-North": 16,
    "Suburbs-South": 15,
    "Nightlife-District": 26,
}

WEEKEND_MULTIPLIER = {
    "Downtown": 1.15,
    "Airport": 1.05,
    "University": 0.6,
    "Suburbs-North": 0.7,
    "Suburbs-South": 0.7,
    "Nightlife-District": 2.1,
}


def _weather_series(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n_days = len(dates)
    # Seasonal temperature curve (deg F) + noise.
    day_of_year = dates.dayofyear.to_numpy()
    temp = 55 + 25 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + rng.normal(0, 6, n_days)
    is_raining = rng.random(n_days) < 0.18
    return pd.DataFrame({"date": dates, "temp_f": temp.round(1), "is_raining": is_raining})


def _holiday_dates(dates: pd.DatetimeIndex, rng: np.random.Generator) -> set:
    # A handful of recurring + random "event" days per year with elevated demand.
    years = sorted(set(dates.year))
    holidays = set()
    for y in years:
        for md in [(1, 1), (7, 4), (11, 27), (12, 25), (12, 31)]:
            holidays.add(pd.Timestamp(year=y, month=md[0], day=md[1]))
        # a few random concert/event days
        n_events = rng.integers(4, 8)
        picks = rng.choice(dates[dates.year == y], size=min(n_events, (dates.year == y).sum()), replace=False)
        holidays.update(pd.Timestamp(d) for d in picks)
    return holidays


def generate_hourly_rides() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS_HISTORY, freq="D")
    weather = _weather_series(dates, rng).set_index("date")
    holidays = _holiday_dates(dates, rng)
    trend = np.linspace(1.0, 1.25, len(dates))

    frames = []
    for zone in ZONES:
        profile = np.array(HOUR_PROFILES[zone], dtype=float)
        profile = profile / profile.mean()
        base = ZONE_BASE_LEVEL[zone]
        weekend_mult = WEEKEND_MULTIPLIER[zone]

        rows = []
        for i, day in enumerate(dates):
            is_weekend = day.dayofweek >= 5
            is_holiday = day.normalize() in holidays
            day_mult = trend[i]
            day_mult *= weekend_mult if is_weekend else 1.0
            day_mult *= 1.6 if is_holiday else 1.0
            w = weather.loc[day]
            weather_mult = 1.0 + (0.35 if w["is_raining"] else 0.0)
            weather_mult *= 1.0 + max(0.0, (40 - w["temp_f"]) / 200)  # cold days push more rides

            mean_hourly = base * profile * day_mult * weather_mult
            counts = rng.poisson(np.clip(mean_hourly, 0.2, None))
            for hour in range(24):
                rows.append(
                    {
                        "zone": zone,
                        "timestamp": day + pd.Timedelta(hours=hour),
                        "date": day,
                        "hour": hour,
                        "day_of_week": day.dayofweek,
                        "is_weekend": is_weekend,
                        "is_holiday": is_holiday,
                        "temp_f": w["temp_f"],
                        "is_raining": bool(w["is_raining"]),
                        "rides": int(counts[hour]),
                    }
                )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)
