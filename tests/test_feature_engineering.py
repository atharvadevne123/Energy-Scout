from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.feature_engineering import EnergyFeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "meter_id": "MTR-001", "building_type": "commercial",
            "timestamp": "2024-06-17T09:00:00", "temperature_c": 22.5,
            "humidity_pct": 55.0, "occupancy_rate": 0.85, "day_of_week": 0,
            "hour": 9, "month": 6, "is_holiday": False,
            "solar_generation_kw": 15.0, "forecast_horizon_hours": 24,
            "consumption_kwh": 480.0,
        },
        {
            "meter_id": "MTR-002", "building_type": "residential",
            "timestamp": "2024-01-10T22:00:00", "temperature_c": -5.0,
            "humidity_pct": 70.0, "occupancy_rate": 0.9, "day_of_week": 2,
            "hour": 22, "month": 1, "is_holiday": False,
            "solar_generation_kw": 0.0, "forecast_horizon_hours": 24,
            "consumption_kwh": 320.0,
        },
    ])


class TestEnergyFeatureEngineer:
    def test_fit_transform_returns_dataframe(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_is_peak_hour_feature(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "is_peak_hour" in result.columns
        assert result["is_peak_hour"].iloc[0] == 1   # hour=9 is peak
        assert result["is_peak_hour"].iloc[1] == 0   # hour=22 is off-peak

    def test_is_weekend_feature(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "is_weekend" in result.columns

    def test_net_consumption_feature(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "net_consumption_kwh" in result.columns
        # Net = consumption - solar
        assert result["net_consumption_kwh"].iloc[0] == pytest.approx(480.0 - 15.0)

    def test_degree_days_feature(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "hdd" in result.columns or "cdd" in result.columns

    def test_hour_cyclical_features(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_building_type_encoding(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        building_cols = [c for c in result.columns if c.startswith("building_")]
        assert len(building_cols) > 0

    def test_no_nans_in_output(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        numeric_cols = result.select_dtypes(include="number").columns
        assert not result[numeric_cols].isnull().any().any()

    def test_transform_without_fit_raises(self, sample_df):
        fe = EnergyFeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(sample_df)

    def test_occupancy_interaction(self, sample_df):
        fe = EnergyFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "consumption_x_occupancy" in result.columns
