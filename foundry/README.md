# Foundry Integration — Energy-Scout

A self-contained Python client for the Palantir Foundry REST API, covering
dataset I/O, model registration, pipeline orchestration, streaming ingest,
lineage, webhooks, schema enforcement, and observability.

## Setup

```bash
export FOUNDRY_HOST=https://your-instance.palantirfoundry.com
export FOUNDRY_TOKEN=<service-account-or-user-token>
export FORECAST_DATASET_RID=ri.foundry.main.dataset.xxxx   # optional
```

```python
from foundry.foundry_client import FoundryClient

with FoundryClient() as client:
    df = client.read_dataset("ri.foundry.main.dataset.abc123")
```

## Methods

| Method | Description |
| --- | --- |
| `upload_dataset(df, dataset_rid, branch='master')` | Upload a DataFrame as Parquet in a single transaction. |
| `read_dataset(dataset_rid, branch='master')` | Read all Parquet files in a branch into one DataFrame. |
| `register_model(model_metadata)` | Register a model version in the Foundry model catalog. |
| `log_predictions(predictions_df)` | Append timestamped forecasts to the forecast dataset. |
| `get_schema(dataset_rid, branch='master')` | Fetch a dataset branch schema. |
| `validate_data_quality(df, required_columns, null_threshold=0.05)` | Pre-upload checks for missing columns / null ratios. |
| `stream_records(records_iterator, dataset_rid, chunk_size=1000)` | Batch an iterator of dicts into DataFrames and APPEND each chunk. |
| `get_lineage(dataset_rid)` | Fetch upstream/downstream dataset lineage. |
| `subscribe_to_dataset(dataset_rid, webhook_url, event_types=...)` | Register a webhook for dataset events (default `TRANSACTION_COMMITTED`). |
| `enforce_schema(df, dataset_rid, branch='master')` | Validate DataFrame columns/dtypes against the Foundry schema; raises `FoundrySchemaError`. |
| `export_snapshot(dataset_rid, output_path, branch='master', format='parquet')` | Download a branch and save as Parquet or CSV. |
| `push_metrics(metrics_dict, dashboard_rid, run_id=None)` | Push a metrics payload to a Foundry metrics sink. |
| `health_check()` | Ping the catalog health endpoint → `{'status', 'latency_ms'}`. |
| `async_upload_dataset(df, dataset_rid, branch='master')` | Non-blocking upload via a thread pool; returns a `Future`. |
| `__enter__ / __exit__ / close()` | Context-manager support; closes the session and worker threads. |

## Usage examples

### Streaming ingest

```python
def event_stream():
    for event in kafka_consumer:
        yield {"id": event.id, "score": event.score}

with FoundryClient() as client:
    summary = client.stream_records(event_stream(), DATASET_RID, chunk_size=500)
    print(summary)  # {'rows': 12500, 'chunks': 25, ...}
```

### Schema-enforced upload

```python
from foundry.foundry_client import FoundrySchemaError

try:
    client.enforce_schema(df, DATASET_RID)
    client.upload_dataset(df, DATASET_RID)
except FoundrySchemaError as exc:
    print(f"Schema drift detected: {exc}")
```

### Async upload + metrics

```python
future = client.async_upload_dataset(df, DATASET_RID)
# ... do other work ...
txn = future.result()

client.push_metrics({"auc": 0.94, "rows": len(df)}, DASHBOARD_RID, run_id="run-42")
```

### Health check & lineage

```python
print(client.health_check())        # {'status': 'healthy', 'latency_ms': 42.1}
print(client.get_lineage(DATASET_RID))
```

## Tests

```bash
pytest foundry/tests/ -v
```

All tests mock `requests.Session` — no Foundry instance or network access
is required.
