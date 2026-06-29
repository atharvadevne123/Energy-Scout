from __future__ import annotations

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def client():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def patch_models(monkeypatch):
    import api.app as app_module

    mock_forecaster = MagicMock()
    _kwh = [480.0 + i * 10 for i in range(24)]
    mock_forecaster.forecast.return_value = {
        "forecast_kwh": _kwh,
        "lower_kwh":    [v * 0.85 for v in _kwh],
        "upper_kwh":    [v * 1.15 for v in _kwh],
        "peak_demand_kwh": max(_kwh),
    }
    mock_forecaster.explain.return_value = [{"shap_features": {"hour": 0.18, "temperature_c": 0.14}}]

    mock_anomaly = MagicMock()
    mock_anomaly.detect.return_value = [
        {"is_anomaly": False, "anomaly_score": 0.08, "anomaly_type": None, "z_score": 0.2}
    ]

    mock_fe = MagicMock()
    mock_fe.transform.side_effect = lambda df: df

    monkeypatch.setattr(app_module, "_forecaster", mock_forecaster)
    monkeypatch.setattr(app_module, "_anomaly_detector", mock_anomaly)
    monkeypatch.setattr(app_module, "_feature_engineer", mock_fe)
    return mock_forecaster


@pytest.fixture
def valid_forecast_request():
    return {
        "meter_id": "MTR-B001-001",
        "building_type": "commercial",
        "timestamp": "2024-06-15T09:00:00",
        "temperature_c": 22.5,
        "humidity_pct": 55.0,
        "occupancy_rate": 0.85,
        "day_of_week": 5,
        "hour": 9,
        "month": 6,
        "is_holiday": False,
        "solar_generation_kw": 15.0,
        "forecast_horizon_hours": 24,
    }


@pytest.fixture
def valid_anomaly_request():
    return {
        "meter_id": "MTR-B001-001",
        "timestamp": "2024-06-15T09:00:00",
        "consumption_kwh": 450.0,
        "expected_kwh": 480.0,
        "building_type": "commercial",
        "temperature_c": 22.5,
    }
