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

import io
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

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
