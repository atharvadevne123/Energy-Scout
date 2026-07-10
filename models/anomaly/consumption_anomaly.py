"""
Energy consumption anomaly detector.

Ensemble of Isolation Forest + Z-score statistical test.
Classifies anomalies into: spike, drop, pattern_shift, equipment_fault.
Supports adaptive thresholds per building type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Per-building-type contamination rates
_CONTAMINATION = {
    "residential": 0.03,
    "commercial": 0.04,
    "industrial": 0.05,
    "data_center": 0.02,
    "default": 0.04,
}

# Z-score threshold map per building type
_ZSCORE_THRESHOLDS = {
    "residential": 2.8,
    "commercial": 3.0,
    "industrial": 3.5,
    "data_center": 4.0,
    "default": 3.0,
}


class ConsumptionAnomalyDetector:
    """
    Isolation Forest + Z-score ensemble for energy consumption anomaly detection.

    Usage:
        detector = ConsumptionAnomalyDetector()
        detector.fit(historical_df)
        result = detector.detect(reading_df)
    """

    def __init__(
        self,
        building_type: str = "default",
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.building_type = building_type
        self.n_estimators = n_estimators
        self.random_state = random_state

        contamination = _CONTAMINATION.get(building_type, _CONTAMINATION["default"])
        self.zscore_threshold = _ZSCORE_THRESHOLDS.get(building_type, _ZSCORE_THRESHOLDS["default"])

        self.iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._feature_cols: list[str] = []
        self._fitted = False

        # Running statistics for z-score
        self._running_mean: float | None = None
        self._running_std: float | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> ConsumptionAnomalyDetector:
        """Fit the detector on historical consumption data.

        Args:
            df: DataFrame with at minimum a ``consumption_kwh`` column.
                May also include weather and temporal features.
        """
        feature_cols = self._select_feature_cols(df)
        self._feature_cols = feature_cols
        X = df[feature_cols].fillna(0).values

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.iso_forest.fit(X_scaled)

        if "consumption_kwh" in df.columns:
            self._running_mean = float(df["consumption_kwh"].mean())
            self._running_std = max(float(df["consumption_kwh"].std()), 1e-6)

        self._fitted = True
        logger.success(
            "ConsumptionAnomalyDetector fitted on {:,} samples (building={}).",
            len(df),
            self.building_type,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, reading_df: pd.DataFrame) -> list[dict]:
        """Detect anomalies in a batch of meter readings.

        Args:
            reading_df: DataFrame with consumption and optional features.
                        Must include ``consumption_kwh``.

        Returns:
            List of dicts per row:
              - anomaly_score  float  0–1  (higher = more anomalous)
              - is_anomaly     bool
              - anomaly_type   str    one of spike/drop/pattern_shift/equipment_fault
              - z_score        float
        """
        self._check_fitted()

        feature_cols = [c for c in self._feature_cols if c in reading_df.columns]
        if not feature_cols:
            feature_cols = self._select_feature_cols(reading_df)

        X = reading_df[feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X) if X.shape[1] == len(self._feature_cols) else X

        # Isolation Forest: scores in (-1, 0); more negative = more anomalous
        raw_scores = self.iso_forest.decision_function(X_scaled)
        # Normalise to [0, 1] anomaly score (1 = most anomalous)
        iso_anomaly_scores = 1.0 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-9
        )

        results = []
        for i, row in reading_df.iterrows():
            idx = list(reading_df.index).index(i)
            consumption = float(row.get("consumption_kwh", 0))
            expected = float(row.get("expected_kwh", self._running_mean or consumption))

            # Z-score test
            z_score = 0.0
            if self._running_mean is not None and self._running_std:
                z_score = (consumption - self._running_mean) / self._running_std

            zscore_anomaly = abs(z_score) > self.zscore_threshold
            iso_anomaly = iso_anomaly_scores[idx] > 0.65

            # Combined score: weighted average
            combined_score = 0.6 * float(iso_anomaly_scores[idx]) + 0.4 * min(
                abs(z_score) / (self.zscore_threshold * 2), 1.0
            )
            is_anomaly = zscore_anomaly or iso_anomaly

            anomaly_type = self._classify_anomaly(consumption, expected, z_score, row)

            results.append(
                {
                    "anomaly_score": round(float(combined_score), 6),
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_type": anomaly_type if is_anomaly else "none",
                    "z_score": round(float(z_score), 4),
                }
            )

        return results

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly scores (0–1) — used by the API."""
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            feature_cols = [c for c in self._feature_cols if c in X.columns]
            arr = X[feature_cols].fillna(0).values if feature_cols else X.values
        else:
            arr = X

        if arr.shape[1] == len(self.scaler.mean_):
            arr = self.scaler.transform(arr)

        raw = self.iso_forest.decision_function(arr)
        span = raw.max() - raw.min() + 1e-9
        return 1.0 - (raw - raw.min()) / span

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "consumption_anomaly.joblib"
        joblib.dump(self, path)
        logger.info("ConsumptionAnomalyDetector saved → {}", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> ConsumptionAnomalyDetector:
        path = Path(path) if path else ARTIFACT_DIR / "consumption_anomaly.joblib"
        obj = joblib.load(path)
        logger.info("ConsumptionAnomalyDetector loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Pick numeric columns suitable for anomaly detection."""
        priority = [
            "consumption_kwh",
            "expected_kwh",
            "temperature_c",
            "humidity_pct",
            "occupancy_rate",
            "solar_generation_kw",
            "hour",
            "day_of_week",
            "month",
        ]
        available = [c for c in priority if c in df.columns]
        # Add remaining numeric columns not already included
        extra = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in available and c not in {"is_anomaly"}
        ]
        return available + extra

    def _classify_anomaly(
        self,
        consumption: float,
        expected: float,
        z_score: float,
        row: pd.Series,
    ) -> str:
        """Classify the type of anomaly based on deviation pattern."""
        if expected <= 0:
            return "pattern_shift"

        ratio = consumption / expected if expected != 0 else 1.0

        if ratio > 1.5 and z_score > 2:
            return "spike"
        if ratio < 0.5 and z_score < -2:
            return "drop"

        hour = int(row.get("hour", -1))
        day_of_week = int(row.get("day_of_week", -1))
        # Off-hours high consumption → equipment fault
        if hour != -1 and (hour < 6 or hour > 22) and ratio > 1.3:
            return "equipment_fault"
        # Weekend commercial high usage → pattern shift
        if day_of_week in (5, 6) and ratio > 1.4:
            return "pattern_shift"

        return "pattern_shift"

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before detect().")
