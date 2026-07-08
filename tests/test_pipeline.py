from __future__ import annotations

from ingest.config import ZONES
from ingest.generate_data import generate_hourly_rides
from ingest.model import forecast_next_week, train_and_evaluate


def test_generate_hourly_rides_shape():
    rides = generate_hourly_rides()
    assert set(rides["zone"].unique()) == set(ZONES)
    assert (rides["rides"] >= 0).all()
    assert rides["hour"].between(0, 23).all()


def test_train_and_evaluate():
    rides = generate_hourly_rides()
    result = train_and_evaluate(rides)
    assert result["mae"] >= 0
    assert -1 <= result["r2"] <= 1
    assert len(result["test_scored"]) > 0


def test_forecast_next_week():
    rides = generate_hourly_rides()
    result = train_and_evaluate(rides)
    forecast = forecast_next_week(rides, result["model"])
    assert (forecast["predicted_rides"] >= 0).all()
    assert set(forecast["zone"].unique()) == set(ZONES)
