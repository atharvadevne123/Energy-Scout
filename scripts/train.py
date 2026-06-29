"""
Train Energy-Scout demand forecaster from scratch.
Usage: python scripts/train.py [--data-path PATH] [--model-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.ensemble.demand_forecaster import DemandForecaster
from models.anomaly.consumption_anomaly import ConsumptionAnomalyDetector
from pipeline.feature_engineering import EnergyFeatureEngineer
from loguru import logger


def generate_synthetic_data(n: int = 50000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    building_types = ["residential", "commercial", "industrial", "data_center"]

    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h")
    hours   = timestamps.hour
    months  = timestamps.month
    dow     = timestamps.dayofweek

    base_kwh = np.where(
        np.isin(dow, [5, 6]),  300.0,  # weekend
        np.where(np.isin(hours, range(9, 18)), 600.0, 350.0)  # business hours
    )
    seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * months / 12)
    noise    = rng.normal(0, 50, n)
    consumption = (base_kwh * seasonal + noise).clip(min=50)

    df = pd.DataFrame({
        "meter_id":           [f"MTR-{i % 100:04d}" for i in range(n)],
        "building_type":      rng.choice(building_types, n),
        "timestamp":          timestamps,
        "temperature_c":      rng.normal(15, 10, n).clip(-20, 45),
        "humidity_pct":       rng.uniform(20, 90, n),
        "occupancy_rate":     np.where(np.isin(dow, [5, 6]), rng.uniform(0, 0.3, n),
                                       rng.uniform(0.4, 1.0, n)),
        "day_of_week":        dow,
        "hour":               hours,
        "month":              months,
        "is_holiday":         rng.choice([True, False], n, p=[0.03, 0.97]),
        "solar_generation_kw": np.where(
                                   np.isin(hours, range(8, 19)),
                                   rng.uniform(0, 50, n), 0.0
                               ),
        "forecast_horizon_hours": 24,
        "consumption_kwh":    consumption,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Train Energy-Scout demand forecaster")
    parser.add_argument("--data-path",   type=str, default=None)
    parser.add_argument("--model-dir",   type=str, default=str(ROOT / "models"))
    parser.add_argument("--n-synthetic", type=int, default=50000)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        path = Path(args.data_path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info("Loaded {:,} rows from {}.", len(df), path)
    else:
        logger.info("Generating {:,} synthetic meter readings.", args.n_synthetic)
        df = generate_synthetic_data(args.n_synthetic)

    label_col = "consumption_kwh"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in data.")

    fe = EnergyFeatureEngineer()
    df_feat = fe.fit_transform(df)

    exclude = {label_col, "meter_id", "timestamp", "building_type"}
    feat_cols = [c for c in df_feat.select_dtypes(include="number").columns if c not in exclude]
    X = df_feat[feat_cols].fillna(0)
    y = df_feat[label_col].fillna(0)

    logger.info("Training demand forecaster on {:,} readings with {} features.", len(X), len(feat_cols))

    model   = DemandForecaster()
    metrics = model.train(X, y)
    model.save(model_dir / "demand_forecaster.joblib")
    joblib.dump(fe, model_dir / "feature_engineer.joblib")
    (model_dir / "feature_cols.json").write_text(json.dumps(feat_cols))

    # Train anomaly detector
    logger.info("Training consumption anomaly detector.")
    anomaly_detector = ConsumptionAnomalyDetector()
    anomaly_detector.fit(X)
    anomaly_detector.save(model_dir / "consumption_anomaly.joblib")

    logger.success("Training complete. Artifacts saved to {}. Metrics: {}", model_dir, metrics)


if __name__ == "__main__":
    main()
