"""
Palantir Foundry integration client for Energy-Scout.

Wraps the Foundry REST API with:
  - Dataset upload / download (Parquet via Foundry Catalog API)
  - Model registration in Foundry's model registry
  - Forecast and anomaly event logging
  - Efficiency report publishing
  - Automatic retries with exponential back-off
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


def _build_session(max_retries: int = MAX_RETRIES) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["GET", "POST", "PUT"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FoundryError(Exception):
    """Raised when a Foundry API call fails."""


class FoundrySchemaError(FoundryError):
    """Raised when a DataFrame does not match the Foundry dataset schema."""


class FoundryClient:
    """
    Palantir Foundry REST API client for Energy-Scout.

    Reads credentials from environment if not supplied:
        FOUNDRY_HOST   — e.g. https://your-instance.palantirfoundry.com
        FOUNDRY_TOKEN  — bearer token
    """

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.host    = (host or os.getenv("FOUNDRY_HOST", "")).rstrip("/")
        self._token  = token or os.getenv("FOUNDRY_TOKEN", "")
        self.timeout = timeout
        self._session = _build_session()

        if not self.host:
            logger.warning("FOUNDRY_HOST not set — Foundry calls will fail unless mocked.")
        if not self._token:
            logger.warning("FOUNDRY_TOKEN not set — Foundry calls will fail unless mocked.")

    def upload_dataset(self, df: pd.DataFrame, dataset_rid: str, branch: str = "master") -> dict[str, Any]:
        logger.info("Uploading {:,} rows to Foundry dataset {} (branch={}).", len(df), dataset_rid, branch)
        txn     = self._open_transaction(dataset_rid, branch)
        txn_rid = txn["rid"]
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            self._put_file(dataset_rid, txn_rid, "data.parquet", buf.read())
            result = self._commit_transaction(dataset_rid, txn_rid)
            logger.success("Uploaded {:,} rows to {} (txn={}).", len(df), dataset_rid, txn_rid)
            return result
        except Exception as exc:
            self._abort_transaction(dataset_rid, txn_rid)
            raise FoundryError(f"Upload failed: {exc}") from exc

    def read_dataset(self, dataset_rid: str, branch: str = "master") -> pd.DataFrame:
        logger.info("Reading Foundry dataset {} (branch={}).", dataset_rid, branch)
        files = self._list_files(dataset_rid, branch)
        parquet_files = [f for f in files if f.get("logicalPath", "").endswith(".parquet")]
        if not parquet_files:
            logger.warning("No Parquet files found in dataset {}.", dataset_rid)
            return pd.DataFrame()
        frames = []
        for file_info in parquet_files:
            content = self._get_file(dataset_rid, branch, file_info["logicalPath"])
            frames.append(pd.read_parquet(io.BytesIO(content)))
        df = pd.concat(frames, ignore_index=True)
        logger.success("Read {:,} rows from {} ({} file(s)).", len(df), dataset_rid, len(frames))
        return df

    def register_model(self, model_metadata: dict[str, Any]) -> dict[str, Any]:
        required = {"name", "version", "framework"}
        missing  = required - set(model_metadata)
        if missing:
            raise ValueError(f"model_metadata missing required keys: {missing}")
        payload = {
            **model_metadata,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "service": "energy-scout",
        }
        logger.info("Registering model '{}' v{} in Foundry catalog.", payload["name"], payload["version"])
        return self._post(f"{self.host}/api/v1/models", json=payload)

    def log_predictions(self, predictions_df: pd.DataFrame) -> dict[str, Any]:
        dataset_rid = os.getenv("FORECAST_DATASET_RID", "")
        if not dataset_rid:
            logger.warning("FORECAST_DATASET_RID not set. Skipping forecast logging.")
            return {"status": "skipped", "reason": "FORECAST_DATASET_RID not set"}
        predictions_df = predictions_df.copy()
        predictions_df["logged_at"] = datetime.now(timezone.utc).isoformat()
        return self.upload_dataset(predictions_df, dataset_rid)

    def get_schema(self, dataset_rid: str, branch: str = "master") -> dict[str, Any]:
        """Fetch the schema of a dataset branch."""
        return self._get(f"{self.host}/api/v1/datasets/{dataset_rid}/schema?branchId={branch}")

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        null_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Validate a DataFrame before uploading to Foundry.

        Checks that all required columns are present and that no column
        exceeds the allowed null ratio. Returns a dict with a ``passed``
        bool and an ``issues`` list describing every violation found.
        """
        issues: list[str] = []
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        if len(df) == 0:
            issues.append("DataFrame is empty.")
        else:
            for col, ratio in df.isna().mean().items():
                if ratio > null_threshold:
                    issues.append(
                        f"Column '{col}' has {ratio:.1%} nulls "
                        f"(threshold {null_threshold:.1%})."
                    )

        result = {"passed": not issues, "issues": issues, "row_count": len(df)}
        if issues:
            logger.warning("Data quality check failed: {}", issues)
        else:
            logger.info("Data quality check passed ({:,} rows).", len(df))
        return result

    # ------------------------------------------------------------------
    # Advanced API
    # ------------------------------------------------------------------

    def stream_records(
        self,
        records_iterator: Iterable[dict[str, Any]],
        dataset_rid: str,
        chunk_size: int = 1000,
    ) -> dict[str, Any]:
        """
        Stream an iterator of record dicts into a Foundry dataset.

        Records are batched into DataFrames of ``chunk_size`` rows and each
        chunk is uploaded in its own APPEND transaction, so arbitrarily
        large (or unbounded) iterators can be ingested with constant memory.

        Returns a summary dict with total rows and chunk count.
        """
        total_rows = 0
        chunks = 0
        batch: list[dict[str, Any]] = []

        def _flush(rows: list[dict[str, Any]]) -> None:
            nonlocal total_rows, chunks
            if not rows:
                return
            df = pd.DataFrame(rows)
            txn = self._post(
                f"{self.host}/api/v1/datasets/{dataset_rid}/transactions",
                json={"branchId": "master", "transactionType": "APPEND"},
            )
            txn_rid = txn["rid"]
            try:
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                buf.seek(0)
                self._put_file(
                    dataset_rid, txn_rid, f"stream/chunk-{chunks:06d}.parquet", buf.read()
                )
                self._commit_transaction(dataset_rid, txn_rid)
            except Exception as exc:
                self._abort_transaction(dataset_rid, txn_rid)
                raise FoundryError(f"Streaming chunk {chunks} failed: {exc}") from exc
            total_rows += len(df)
            chunks += 1
            logger.info("Streamed chunk {} ({:,} rows) to {}.", chunks, len(df), dataset_rid)

        for record in records_iterator:
            batch.append(record)
            if len(batch) >= chunk_size:
                _flush(batch)
                batch = []
        _flush(batch)

        logger.success(
            "Streaming complete: {:,} rows in {} chunk(s) to {}.",
            total_rows, chunks, dataset_rid,
        )
        return {"rows": total_rows, "chunks": chunks, "dataset_rid": dataset_rid}

    def get_lineage(self, dataset_rid: str) -> dict[str, Any]:
        """Fetch upstream/downstream lineage for a dataset."""
        lineage = self._get(
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}/lineage"
        )
        logger.info(
            "Fetched lineage for {}: {} upstream, {} downstream.",
            dataset_rid,
            len(lineage.get("upstream", [])),
            len(lineage.get("downstream", [])),
        )
        return lineage

    def subscribe_to_dataset(
        self,
        dataset_rid: str,
        webhook_url: str,
        event_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Register a webhook subscription for dataset events.

        By default subscribes to ``TRANSACTION_COMMITTED`` so the webhook
        fires whenever new data lands on the dataset.
        """
        event_types = event_types or ["TRANSACTION_COMMITTED"]
        resp = self._post(
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}/subscriptions",
            json={"webhookUrl": webhook_url, "eventTypes": event_types},
        )
        logger.success(
            "Subscribed {} to {} for events {}.", webhook_url, dataset_rid, event_types
        )
        return resp

    def enforce_schema(
        self,
        df: pd.DataFrame,
        dataset_rid: str,
        branch: str = "master",
    ) -> dict[str, Any]:
        """
        Validate a DataFrame against the dataset's Foundry schema.

        Checks that every schema column exists in the DataFrame and that
        dtypes are compatible. Raises :class:`FoundrySchemaError` on any
        mismatch; returns the fetched schema when validation passes.
        """
        schema = self.get_schema(dataset_rid, branch)
        fields = schema.get("fieldSchemaList", schema.get("fields", []))

        type_map = {
            "STRING": ("object", "string"),
            "INTEGER": ("int8", "int16", "int32", "int64", "Int64"),
            "LONG": ("int8", "int16", "int32", "int64", "Int64"),
            "DOUBLE": ("float16", "float32", "float64"),
            "FLOAT": ("float16", "float32", "float64"),
            "BOOLEAN": ("bool", "boolean"),
            "TIMESTAMP": ("datetime64[ns]", "datetime64[ns, UTC]"),
            "DATE": ("datetime64[ns]", "object"),
        }

        problems: list[str] = []
        for field in fields:
            name = field.get("name")
            ftype = str(field.get("type", "")).upper()
            if name not in df.columns:
                problems.append(f"Missing column '{name}' (expected {ftype}).")
                continue
            expected = type_map.get(ftype)
            if expected and str(df[name].dtype) not in expected:
                problems.append(
                    f"Column '{name}' has dtype {df[name].dtype}, "
                    f"expected one of {expected} for Foundry type {ftype}."
                )

        if problems:
            raise FoundrySchemaError(
                f"DataFrame does not match schema of {dataset_rid}: {problems}"
            )
        logger.info("Schema enforcement passed for {} ({} fields).", dataset_rid, len(fields))
        return schema

    def export_snapshot(
        self,
        dataset_rid: str,
        output_path: str,
        branch: str = "master",
        format: str = "parquet",
    ) -> str:
        """
        Download a dataset branch and save it locally.

        ``format`` may be ``parquet`` or ``csv``. Returns the output path.
        """
        if format not in ("parquet", "csv"):
            raise ValueError(f"Unsupported format '{format}' (use 'parquet' or 'csv').")

        df = self.read_dataset(dataset_rid, branch)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.success(
            "Exported {:,} rows from {} to {} ({}).",
            len(df), dataset_rid, output_path, format,
        )
        return output_path

    def push_metrics(
        self,
        metrics_dict: dict[str, Any],
        dashboard_rid: str,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Push a metrics payload to a Foundry metrics sink / dashboard."""
        payload = {
            "dashboardRid": dashboard_rid,
            "metrics": metrics_dict,
            "runId": run_id,
            "recordedAt": datetime.now(timezone.utc).isoformat(),
            "service": "energy-scout",
        }
        resp = self._post(f"{self.host}/foundry-metrics/api/metrics", json=payload)
        logger.success(
            "Pushed {} metric(s) to dashboard {}.", len(metrics_dict), dashboard_rid
        )
        return resp

    def health_check(self) -> dict[str, Any]:
        """
        Ping the Foundry catalog health endpoint.

        Returns ``{'status': <str>, 'latency_ms': <float>}``. Never raises:
        connection failures are reported as ``status='unreachable'``.
        """
        url = f"{self.host}/foundry-catalog/api/catalog/health"
        start = time.monotonic()
        try:
            resp = self._session.get(url, headers=self._headers(), timeout=self.timeout)
            latency_ms = (time.monotonic() - start) * 1000.0
            status = "healthy" if resp.ok else f"unhealthy ({resp.status_code})"
        except requests.RequestException as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            logger.error("Foundry health check failed: {}", exc)
            status = "unreachable"
        result = {"status": status, "latency_ms": round(latency_ms, 2)}
        logger.info("Foundry health: {} ({} ms).", result["status"], result["latency_ms"])
        return result

    def async_upload_dataset(
        self,
        df: pd.DataFrame,
        dataset_rid: str,
        branch: str = "master",
    ) -> concurrent.futures.Future:
        """
        Upload a DataFrame without blocking the caller.

        Runs :meth:`upload_dataset` in a background thread and returns the
        Future; call ``.result()`` to wait for and retrieve the transaction
        metadata (or re-raise any upload error).
        """
        executor = getattr(self, "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="foundry-upload"
            )
            self._executor = executor
        logger.info(
            "Scheduling async upload of {:,} rows to {} (branch={}).",
            len(df), dataset_rid, branch,
        )
        return executor.submit(self.upload_dataset, df, dataset_rid, branch)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FoundryClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Release the HTTP session and any background upload threads."""
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None
        self._session.close()
        logger.debug("FoundryClient closed.")

    # ------------------------------------------------------------------
    # Low-level REST helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def _open_transaction(self, dataset_rid: str, branch: str) -> dict:
        return self._post(f"{self.host}/api/v1/datasets/{dataset_rid}/transactions", json={"branchId": branch})

    def _commit_transaction(self, dataset_rid: str, txn_rid: str) -> dict:
        return self._post(f"{self.host}/api/v1/datasets/{dataset_rid}/transactions/{txn_rid}/commit", json={})

    def _abort_transaction(self, dataset_rid: str, txn_rid: str) -> None:
        try:
            self._post(f"{self.host}/api/v1/datasets/{dataset_rid}/transactions/{txn_rid}/abort", json={})
        except Exception as exc:
            logger.error("Failed to abort transaction {}: {}", txn_rid, exc)

    def _put_file(self, dataset_rid: str, txn_rid: str, logical_path: str, data: bytes) -> None:
        url = f"{self.host}/api/v1/datasets/{dataset_rid}/transactions/{txn_rid}/files:upload"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/octet-stream",
            "X-Foundry-Logical-Path": logical_path,
        }
        resp = self._session.put(url, data=data, headers=headers, timeout=self.timeout)
        self._raise_for_status(resp, "PUT file")

    def _list_files(self, dataset_rid: str, branch: str) -> list[dict]:
        result = self._get(f"{self.host}/api/v1/datasets/{dataset_rid}/files?branchId={branch}")
        return result.get("data", [])

    def _get_file(self, dataset_rid: str, branch: str, logical_path: str) -> bytes:
        url  = f"{self.host}/api/v1/datasets/{dataset_rid}/files/{logical_path}/content?branchId={branch}"
        resp = self._session.get(url, headers={"Authorization": f"Bearer {self._token}"}, timeout=self.timeout)
        self._raise_for_status(resp, f"GET file {logical_path}")
        return resp.content

    def _get(self, url: str) -> dict:
        resp = self._session.get(url, headers=self._headers(), timeout=self.timeout)
        self._raise_for_status(resp, f"GET {url}")
        return resp.json()

    def _post(self, url: str, **kwargs) -> dict:
        resp = self._session.post(url, headers=self._headers(), timeout=self.timeout, **kwargs)
        self._raise_for_status(resp, f"POST {url}")
        try:
            return resp.json()
        except Exception:
            return {"status": resp.status_code}

    @staticmethod
    def _raise_for_status(resp: requests.Response, context: str = "") -> None:
        if not resp.ok:
            raise FoundryError(f"Foundry API error [{context}]: HTTP {resp.status_code} — {resp.text[:500]}")
