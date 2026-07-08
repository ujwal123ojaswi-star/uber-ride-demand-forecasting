from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RIDES_PATH = DATA_DIR / "hourly_rides.parquet"
MODEL_PATH = DATA_DIR / "model.joblib"
SCORED_PATH = DATA_DIR / "scored_hourly.parquet"
FORECAST_PATH = DATA_DIR / "forecast_hourly.parquet"

N_DAYS_HISTORY = 545  # ~18 months
RANDOM_SEED = 7

ZONES = [
    "Downtown",
    "Airport",
    "University",
    "Suburbs-North",
    "Suburbs-South",
    "Nightlife-District",
]

TEST_HOLDOUT_DAYS = 30
FORECAST_HORIZON_DAYS = 7
