"""
Airflow DAG: energy_scout_retrain
Runs hourly. Pulls smart meter data from Palantir Foundry, checks for drift,
retrains the demand forecaster, evaluates, and pushes updated model to Foundry.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "energy-scout",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="energy_scout_retrain",
    description="Hourly Energy-Scout demand forecaster retraining with Palantir Foundry sync",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["energy-scout", "ml", "foundry", "smart-grid"],
)


def fetch_meter_data_from_foundry(**ctx):
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from foundry.foundry_client import FoundryClient

    client = FoundryClient()
    dataset_rid = os.getenv("METER_DATA_DATASET_RID", "")
    if not dataset_rid:
        raise ValueError("METER_DATA_DATASET_RID env var not set.")

    df = client.read_dataset(dataset_rid)
    if df.empty:
        raise ValueError("No meter data returned from Foundry.")

    tmp_path = "/tmp/energy_training_data.parquet"
    df.to_parquet(tmp_path, index=False)
    ctx["ti"].xcom_push(key="training_data_path", value=tmp_path)
    print(f"Fetched {len(df):,} meter readings from Foundry.")


def check_drift(**ctx):
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from monitoring.drift_monitor import DriftMonitor

    training_path = ctx["ti"].xcom_pull(key="training_data_path")
    df = pd.read_parquet(training_path)
    mid = len(df) // 2

    monitor = DriftMonitor(threshold=0.05)
    report = monitor.detect_drift(df.iloc[:mid], df.iloc[mid:])
    drifted = [f for f, r in report.items() if r.get("drift_detected")]
    print(f"Drift check: {len(drifted)} features drifted out of {len(report)}.")


def retrain_model(**ctx):
    import json
    import sys
    from pathlib import Path

    import joblib
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.ensemble.demand_forecaster import DemandForecaster
    from pipeline.feature_engineering import EnergyFeatureEngineer

    training_path = ctx["ti"].xcom_pull(key="training_data_path")
    df = pd.read_parquet(training_path)

    label_col = "consumption_kwh"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in training data.")

    fe = EnergyFeatureEngineer()
    df_feat = fe.fit_transform(df)

    exclude = {label_col, "meter_id", "timestamp", "building_type"}
    feat_cols = [c for c in df_feat.select_dtypes(include="number").columns if c not in exclude]
    X = df_feat[feat_cols].fillna(0)
    y = df_feat[label_col].fillna(0)

    model = DemandForecaster()
    model.train(X, y)

    model_dir = Path(__file__).parent.parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "demand_forecaster.joblib")
    joblib.dump(fe, model_dir / "feature_engineer.joblib")
    (model_dir / "feature_cols.json").write_text(json.dumps(feat_cols))

    print(f"Demand forecaster retrained on {len(df):,} meter readings.")
    ctx["ti"].xcom_push(key="model_dir", value=str(model_dir))


def evaluate_model(**ctx):
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.ensemble.demand_forecaster import DemandForecaster

    model_dir = Path(ctx["ti"].xcom_pull(key="model_dir"))
    training_path = ctx["ti"].xcom_pull(key="training_data_path")

    model = DemandForecaster.load(model_dir / "demand_forecaster.joblib")
    df = pd.read_parquet(training_path)
    feat_cols = json.loads((model_dir / "feature_cols.json").read_text())
    X = df.reindex(columns=feat_cols, fill_value=0).fillna(0)

    forecasts = model.forecast(X.head(100), horizon=1)
    avg_kwh = float(sum(forecasts) / len(forecasts)) if forecasts else 0.0
    print(f"Evaluation — average 1-step forecast: {avg_kwh:.2f} kWh")
    ctx["ti"].xcom_push(key="avg_forecast_kwh", value=avg_kwh)


def push_model_to_foundry(**ctx):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from foundry.foundry_client import FoundryClient

    client = FoundryClient()
    avg_kwh = ctx["ti"].xcom_pull(key="avg_forecast_kwh")

    client.register_model(
        {
            "name": "energy-scout-demand-forecaster",
            "version": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "framework": "xgboost+lightgbm",
            "metrics": {"avg_forecast_kwh": avg_kwh},
        }
    )
    print("Energy-Scout model registered in Foundry catalog.")


fetch_task = PythonOperator(
    task_id="fetch_meter_data", python_callable=fetch_meter_data_from_foundry, dag=dag
)
drift_task = PythonOperator(task_id="check_drift", python_callable=check_drift, dag=dag)
retrain_task = PythonOperator(task_id="retrain_model", python_callable=retrain_model, dag=dag)
eval_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model, dag=dag)
push_task = PythonOperator(
    task_id="push_model_to_foundry", python_callable=push_model_to_foundry, dag=dag
)

fetch_task >> drift_task >> retrain_task >> eval_task >> push_task
