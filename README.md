# Energy-Scout

> ML-powered energy demand forecasting and anomaly detection API — XGBoost-LightGBM ensemble with temporal feature engineering, solar net consumption, and Palantir Foundry integration.

## Overview

Energy-Scout forecasts hourly electricity consumption for residential, commercial, industrial, and data-center buildings up to 168 hours ahead, and detects anomalies in real-time meter readings (spikes, drops, pattern shifts, equipment faults). It is built around a soft-voting ensemble of XGBoost, LightGBM, and Random Forest, backed by rich temporal and weather feature engineering, and integrates with Palantir Foundry for dataset sync and model registry.

## Features

- **Demand Forecasting** — multi-horizon (1–168 h) with calibrated confidence intervals and tier labels: `LOW / MEDIUM / HIGH / CRITICAL`
- **Anomaly Detection** — classifies meter anomalies as `spike / drop / pattern_shift / equipment_fault` with a Z-score and anomaly probability
- **Solar Net Consumption** — net energy = consumption − solar generation
- **Temporal Feature Engineering** — cyclical hour encoding (sin/cos), peak-hour flag, HDD/CDD degree-days, lag features (1h, 2h, 24h, 168h), rolling statistics (24h, 7d, 30d)
- **Building Type Encoding** — one-hot for residential, commercial, industrial, data_center
- **Batch Endpoint** — forecast up to 200 meters per request
- **KS-Drift Monitoring** — Evidently-based covariate drift on consumption features
- **Palantir Foundry Integration** — meter data sync, forecast publishing, model registry
- **Automated Retraining** — hourly Airflow DAG with Foundry data I/O
- **Prometheus Metrics** — forecast count, latency, peak demand distribution
- **Efficiency Analysis** — GET `/efficiency` returns building-level consumption benchmarks

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | Flask 3, Flask-RESTX, Gunicorn |
| ML Models | XGBoost 2, LightGBM 4, scikit-learn (Random Forest, isotonic calibration) |
| Anomaly Detection | scikit-learn (Isolation Forest) |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Validation | marshmallow |
| Rate Limiting | Flask-Limiter |
| Drift Monitoring | Evidently |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow 2 |
| Data Platform | Palantir Foundry REST API (Parquet, transaction writes) |
| Observability | Prometheus, prometheus-client |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Containerisation | Docker, docker-compose |
| Testing | pytest, pytest-mock |
| Runtime | Python 3.11 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/forecast` | Forecast energy demand for a meter |
| `POST` | `/forecast/batch` | Forecast up to 200 meters |
| `POST` | `/anomaly` | Detect anomaly in a meter reading |
| `GET` | `/efficiency` | Building-level consumption benchmarks |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/model/info` | Model version and building type metadata |
| `GET` | `/docs` | Swagger UI |

### POST `/forecast` — Request

```json
{
  "meter_id": "MTR-B001-001",
  "building_type": "commercial",
  "timestamp": "2024-06-17T09:00:00",
  "temperature_c": 22.5,
  "humidity_pct": 55.0,
  "occupancy_rate": 0.85,
  "day_of_week": 0,
  "hour": 9,
  "month": 6,
  "is_holiday": false,
  "solar_generation_kw": 15.0,
  "forecast_horizon_hours": 24
}
```

### POST `/forecast` — Response

```json
{
  "meter_id": "MTR-B001-001",
  "forecast_kwh": [512.4, 498.1, "..."],
  "confidence_interval": {
    "lower_kwh": [435.5, 423.4, "..."],
    "upper_kwh": [589.3, 572.8, "..."]
  },
  "peak_demand_kwh": 612.8,
  "demand_tier": "HIGH",
  "forecast_horizon_hours": 24,
  "request_id": "req-abc123",
  "latency_ms": 21.3
}
```

### POST `/anomaly` — Request

```json
{
  "meter_id": "MTR-B001-001",
  "timestamp": "2024-06-17T09:00:00",
  "consumption_kwh": 1850.0,
  "expected_kwh": 520.0,
  "building_type": "commercial",
  "temperature_c": 22.5
}
```

### POST `/anomaly` — Response

```json
{
  "meter_id": "MTR-B001-001",
  "is_anomaly": true,
  "anomaly_score": 0.94,
  "anomaly_type": "spike",
  "z_score": 7.2,
  "request_id": "req-def456",
  "latency_ms": 8.7
}
```

## Project Structure

```
Energy-Scout/
├── api/
│   ├── app.py               # Flask-RESTX application
│   └── wsgi.py
├── foundry/
│   └── foundry_client.py    # Palantir Foundry REST client
├── models/
│   ├── ensemble/
│   │   └── demand_forecaster.py     # XGBoost + LightGBM + RF ensemble
│   └── anomaly/
│       └── consumption_anomaly.py   # Isolation Forest anomaly detector
├── pipeline/
│   ├── feature_engineering.py       # Temporal, weather, lag, rolling features
│   └── airflow/
│       └── retrain_dag.py           # Hourly retraining DAG
├── monitoring/
│   └── drift_monitor.py
├── scripts/
│   └── train.py                     # 50k synthetic meter readings + training
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_feature_engineering.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Palantir Foundry Integration

Energy-Scout syncs with Foundry for:

- **Meter Data Dataset** — historical hourly readings with labels for retraining
- **Forecast Dataset** — real-time forecasts published for downstream analytics
- **Model Registry** — forecaster and anomaly detector artifacts

Configure via `.env`:

```env
FOUNDRY_HOST=https://your-instance.palantirfoundry.com
FOUNDRY_TOKEN=your-bearer-token
METER_DATA_DATASET_RID=ri.foundry.main.dataset.xxxxxxxx
FORECAST_DATASET_RID=ri.foundry.main.dataset.yyyyyyyy
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py        # generates 50k readings + trains both models
gunicorn -b 0.0.0.0:8003 api.wsgi:app
# Or: docker-compose -f docker/docker-compose.yml up
```

## Running Tests

```bash
pytest tests/ -v
```

30 tests — forecast, anomaly, batch, efficiency, and energy feature engineering.

## Airflow DAG

The `energy_scout_retrain` DAG runs hourly:

```
fetch_meter_data
    → check_drift
    → retrain_model
    → evaluate_model
    → push_model_to_foundry
```
