"""
Energy feature engineering pipeline.

EnergyFeatureEngineer transforms raw meter readings into ML-ready
feature vectors with temporal, weather, lag, rolling, building-type,
solar integration, and occupancy interaction features.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

# US holidays (simplified static set — replace with `holidays` library in production)
_US_FEDERAL_HOLIDAYS = {
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
}

# Peak hours: 9 AM – 6 PM weekdays
_PEAK_START = 9
_PEAK_END = 18

# Building-type encoding
_BUILDING_TYPES = ["residential", "commercial", "industrial", "data_center"]

# Degree-day base temperature (°C)
_BASE_TEMP_C = 18.0

# Temperature bin edges (°C)
_TEMP_BINS = [-np.inf, 0, 10, 18, 25, 32, np.inf]
_TEMP_BIN_LABELS = ["freezing", "cold", "cool", "comfortable", "warm", "hot"]

# Humidity bucket edges (%)
_HUM_BINS = [0, 30, 50, 70, 85, 100]
_HUM_BIN_LABELS = ["very_dry", "dry", "moderate", "humid", "very_humid"]


class EnergyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible transformer that produces a rich feature matrix
    from raw energy meter DataFrames.

    Required input columns:
        - consumption_kwh (float)
        - timestamp (str | datetime)

    Optional input columns (used when present):
        - temperature_c, humidity_pct, occupancy_rate, solar_generation_kw,
          building_type, is_holiday, hour, day_of_week, month
    """

    def __init__(
        self,
        lag_hours: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        add_solar_net: bool = True,
        add_occupancy_interaction: bool = True,
    ) -> None:
        # Lag offsets in hours
        self.lag_hours = lag_hours or [1, 2, 24, 168]
        # Rolling window sizes in hours (1h resolution assumed)
        self.rolling_windows = rolling_windows or [24, 168, 720]  # 24h, 7d, 30d
        self.add_solar_net = add_solar_net
        self.add_occupancy_interaction = add_occupancy_interaction
        self._fitted = False
        self._output_cols: list[str] = []

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y=None) -> EnergyFeatureEngineer:
        """Compute any statistics needed for stable transformation."""
        transformed = self._transform_impl(df, fit=True)
        self._output_cols = list(transformed.columns)
        self._fitted = True
        logger.info("EnergyFeatureEngineer fitted. Output columns: {:,}", len(self._output_cols))
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform input DataFrame into feature matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self._transform_impl(df, fit=False)

    def fit_transform(self, df: pd.DataFrame, y=None, **params) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _transform_impl(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        out = df.copy()

        # 1. Parse timestamps
        out = self._parse_timestamps(out)

        # 2. Temporal features
        out = self._add_temporal_features(out)

        # 3. Weather features
        out = self._add_weather_features(out)

        # 4. Lag features (only meaningful for time-ordered data)
        out = self._add_lag_features(out)

        # 5. Rolling statistics
        out = self._add_rolling_features(out)

        # 6. Building type encoding
        out = self._add_building_type_encoding(out)

        # 7. Solar net consumption
        if self.add_solar_net:
            out = self._add_solar_features(out)

        # 8. Occupancy interaction
        if self.add_occupancy_interaction:
            out = self._add_occupancy_features(out)

        # Drop raw text columns that can't be used by models directly
        drop_cols = ["timestamp", "building_type"]
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])

        return out.fillna(0)

    # ------------------------------------------------------------------
    # Individual feature groups
    # ------------------------------------------------------------------

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday
            df["month"] = df["timestamp"].dt.month
            df["quarter"] = df["timestamp"].dt.quarter
            df["day_of_year"] = df["timestamp"].dt.dayofyear
            df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        hour = df.get("hour", pd.Series(0, index=df.index))
        dow = df.get("day_of_week", pd.Series(0, index=df.index))
        month = df.get("month", pd.Series(1, index=df.index))

        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_weekday"] = (dow < 5).astype(int)
        df["is_peak_hour"] = ((hour >= _PEAK_START) & (hour < _PEAK_END) & (dow < 5)).astype(int)

        # Holiday flag: use input column if available, else derive from static set
        if "is_holiday" not in df.columns:
            df["is_holiday"] = 0
            if "timestamp" in df.columns:

                def _is_holiday(ts):
                    if pd.isna(ts):
                        return 0
                    return int((ts.month, ts.day) in _US_FEDERAL_HOLIDAYS)

                df["is_holiday"] = df["timestamp"].apply(_is_holiday)
        else:
            df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

        # Cyclical encoding for hour, day of week, month
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

        return df

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "temperature_c" in df.columns:
            temp = df["temperature_c"].fillna(_BASE_TEMP_C)

            # Heating / cooling degree-days
            df["hdd"] = np.maximum(0, _BASE_TEMP_C - temp)
            df["cdd"] = np.maximum(0, temp - _BASE_TEMP_C)

            # Temperature bins (one-hot)
            bins = pd.cut(temp, bins=_TEMP_BINS, labels=_TEMP_BIN_LABELS, right=False)
            for label in _TEMP_BIN_LABELS:
                df[f"temp_bin_{label}"] = (bins == label).astype(int)

        if "humidity_pct" in df.columns:
            hum = df["humidity_pct"].clip(0, 100).fillna(50)
            df["humidity_pct"] = hum

            # Humidity buckets
            hbins = pd.cut(
                hum, bins=_HUM_BINS, labels=_HUM_BIN_LABELS, right=True, include_lowest=True
            )
            for label in _HUM_BIN_LABELS:
                df[f"hum_bucket_{label}"] = (hbins == label).astype(int)

            # Feels-like approximation (Heat Index simplified)
            if "temperature_c" in df.columns:
                temp = df["temperature_c"].fillna(_BASE_TEMP_C)
                df["feels_like_c"] = (
                    temp + 0.33 * (hum / 100 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))) - 4.0
                )

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "consumption_kwh" not in df.columns:
            return df
        for lag in self.lag_hours:
            col = f"consumption_lag_{lag}h"
            df[col] = df["consumption_kwh"].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "consumption_kwh" not in df.columns:
            return df
        for window in self.rolling_windows:
            label = f"{window}h" if window < 720 else "30d"
            df[f"rolling_mean_{label}"] = (
                df["consumption_kwh"].rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{label}"] = (
                df["consumption_kwh"].rolling(window=window, min_periods=1).std().fillna(0)
            )
        # 24h max/min
        df["rolling_max_24h"] = df["consumption_kwh"].rolling(window=24, min_periods=1).max()
        df["rolling_min_24h"] = df["consumption_kwh"].rolling(window=24, min_periods=1).min()
        return df

    def _add_building_type_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if "building_type" not in df.columns:
            for bt in _BUILDING_TYPES:
                df[f"building_{bt}"] = 0
            return df

        for bt in _BUILDING_TYPES:
            df[f"building_{bt}"] = (df["building_type"] == bt).astype(int)
        return df

    def _add_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "solar_generation_kw" not in df.columns:
            df["solar_generation_kw"] = 0.0
        if "consumption_kwh" in df.columns:
            df["net_consumption_kwh"] = df["consumption_kwh"] - df["solar_generation_kw"].fillna(0)
        return df

    def _add_occupancy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "occupancy_rate" not in df.columns:
            df["occupancy_rate"] = 1.0
        if "consumption_kwh" in df.columns:
            df["consumption_x_occupancy"] = df["consumption_kwh"] * df["occupancy_rate"].clip(
                0, 1
            ).fillna(1.0)
        return df

    @property
    def output_columns(self) -> list[str]:
        return self._output_cols


def build_forecast_features(request_data: dict) -> pd.DataFrame:
    """Build a single-row feature DataFrame from an API forecast request.

    This is a lightweight version of the full EnergyFeatureEngineer used
    at inference time when historical lag features are not available.
    """
    ts_str = request_data.get("timestamp", datetime.utcnow().isoformat())
    try:
        ts = pd.to_datetime(ts_str)
    except Exception:
        ts = pd.Timestamp.utcnow()

    hour = request_data.get("hour", ts.hour)
    dow = request_data.get("day_of_week", ts.dayofweek)
    month = request_data.get("month", ts.month)

    building_type = request_data.get("building_type", "commercial")
    temp = request_data.get("temperature_c", 20.0)
    hum = request_data.get("humidity_pct", 50.0)
    occupancy = request_data.get("occupancy_rate", 1.0)
    solar = request_data.get("solar_generation_kw", 0.0)
    is_holiday = int(request_data.get("is_holiday", False))

    row = {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "quarter": (month - 1) // 3 + 1,
        "is_weekend": int(dow >= 5),
        "is_weekday": int(dow < 5),
        "is_peak_hour": int(_PEAK_START <= hour < _PEAK_END and dow < 5),
        "is_holiday": is_holiday,
        # Cyclical encodings
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12)),
        # Weather
        "temperature_c": temp,
        "humidity_pct": hum,
        "hdd": max(0.0, _BASE_TEMP_C - temp),
        "cdd": max(0.0, temp - _BASE_TEMP_C),
        "solar_generation_kw": solar,
        "occupancy_rate": occupancy,
        # Building type one-hot
        **{f"building_{bt}": int(building_type == bt) for bt in _BUILDING_TYPES},
        # Placeholder lag features (zero-filled at inference without history)
        **{f"consumption_lag_{lag}h": 0.0 for lag in [1, 2, 24, 168]},
        # Placeholder rolling features
        "rolling_mean_24h": 0.0,
        "rolling_std_24h": 0.0,
        "rolling_mean_168h": 0.0,
        "rolling_std_168h": 0.0,
        "rolling_mean_30d": 0.0,
        "rolling_std_30d": 0.0,
        "rolling_max_24h": 0.0,
        "rolling_min_24h": 0.0,
        # Interaction features
        "net_consumption_kwh": 0.0,
        "consumption_x_occupancy": 0.0,
    }

    return pd.DataFrame([row])
