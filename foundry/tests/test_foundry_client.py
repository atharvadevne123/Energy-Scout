"""Unit tests for foundry.foundry_client.FoundryClient.

All HTTP traffic is mocked at the requests.Session level — no network
access is required to run these tests.
"""

import io
from unittest.mock import MagicMock

import pandas as pd
import pytest

from foundry.foundry_client import (
    FoundryClient,
    FoundryError,
    FoundrySchemaError,
)

HOST = "https://foundry.example.com"


def _response(status_code=200, json_data=None, content=b""):
    resp = MagicMock()
    resp.ok = 200 <= status_code < 300
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.content = content
    resp.text = str(json_data)
    return resp


@pytest.fixture
def client():
    c = FoundryClient(host=HOST, token="test-token")
    c._session = MagicMock()
    return c


@pytest.fixture
def sample_df():
    return pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.5, 0.9]})


# ---------------------------------------------------------------------------
# upload_dataset
# ---------------------------------------------------------------------------


def test_upload_dataset_commits_transaction(client, sample_df):
    client._session.post.side_effect = [
        _response(json_data={"rid": "txn-1"}),  # open transaction
        _response(json_data={"committed": True}),  # commit
    ]
    client._session.put.return_value = _response()

    result = client.upload_dataset(sample_df, "ri.foundry.main.dataset.abc")

    assert result == {"committed": True}
    assert client._session.put.call_count == 1
    put_url = client._session.put.call_args[0][0]
    assert "txn-1" in put_url


def test_upload_dataset_aborts_on_failure(client, sample_df):
    client._session.post.side_effect = [
        _response(json_data={"rid": "txn-1"}),  # open transaction
        _response(json_data={}),  # abort
    ]
    client._session.put.return_value = _response(status_code=500, json_data={})

    with pytest.raises(FoundryError):
        client.upload_dataset(sample_df, "ri.foundry.main.dataset.abc")

    abort_url = client._session.post.call_args[0][0]
    assert abort_url.endswith("/abort")


# ---------------------------------------------------------------------------
# read_dataset
# ---------------------------------------------------------------------------


def test_read_dataset_concatenates_parquet_files(client, sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    client._session.get.side_effect = [
        _response(
            json_data={
                "data": [
                    {"logicalPath": "part-0.parquet"},
                    {"logicalPath": "notes.txt"},
                ]
            }
        ),
        _response(content=parquet_bytes),
    ]

    df = client.read_dataset("ri.foundry.main.dataset.abc")

    pd.testing.assert_frame_equal(df, sample_df)


def test_read_dataset_empty_when_no_parquet(client):
    client._session.get.return_value = _response(json_data={"data": []})
    df = client.read_dataset("ri.foundry.main.dataset.abc")
    assert df.empty


# ---------------------------------------------------------------------------
# validate_data_quality
# ---------------------------------------------------------------------------


def test_validate_data_quality_passes(client, sample_df):
    result = client.validate_data_quality(sample_df, required_columns=["id", "score"])
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["row_count"] == 3


def test_validate_data_quality_flags_missing_columns_and_nulls(client):
    df = pd.DataFrame({"id": [1, None, None, None]})
    result = client.validate_data_quality(df, required_columns=["id", "label"], null_threshold=0.05)
    assert result["passed"] is False
    assert any("label" in issue for issue in result["issues"])
    assert any("nulls" in issue for issue in result["issues"])


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------


def test_register_model_posts_metadata(client):
    client._session.post.return_value = _response(json_data={"modelRid": "m-1"})
    result = client.register_model({"name": "clf", "version": "1.0", "framework": "sklearn"})
    assert result == {"modelRid": "m-1"}
    sent = client._session.post.call_args[1]["json"]
    assert sent["name"] == "clf"
    assert "registered_at" in sent


def test_register_model_requires_keys(client):
    with pytest.raises(ValueError):
        client.register_model({"name": "clf"})


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


def test_health_check_healthy(client):
    client._session.get.return_value = _response(json_data={"status": "ok"})
    result = client.health_check()
    assert result["status"] == "healthy"
    assert isinstance(result["latency_ms"], float)


def test_health_check_unreachable(client):
    import requests as _requests

    client._session.get.side_effect = _requests.ConnectionError("boom")
    result = client.health_check()
    assert result["status"] == "unreachable"


# ---------------------------------------------------------------------------
# stream_records
# ---------------------------------------------------------------------------


def test_stream_records_chunks_and_appends(client):
    # Every POST opens or commits a transaction.
    client._session.post.return_value = _response(json_data={"rid": "txn-s"})
    client._session.put.return_value = _response()

    records = ({"id": i} for i in range(2500))
    summary = client.stream_records(records, "ri.foundry.main.dataset.abc", chunk_size=1000)

    assert summary["rows"] == 2500
    assert summary["chunks"] == 3
    assert client._session.put.call_count == 3
    open_txn_payload = client._session.post.call_args_list[0][1]["json"]
    assert open_txn_payload["transactionType"] == "APPEND"


# ---------------------------------------------------------------------------
# enforce_schema
# ---------------------------------------------------------------------------


def test_enforce_schema_raises_on_mismatch(client, sample_df):
    client._session.get.return_value = _response(
        json_data={
            "fieldSchemaList": [
                {"name": "id", "type": "INTEGER"},
                {"name": "missing_col", "type": "STRING"},
            ]
        }
    )
    with pytest.raises(FoundrySchemaError):
        client.enforce_schema(sample_df, "ri.foundry.main.dataset.abc")


def test_enforce_schema_passes(client, sample_df):
    client._session.get.return_value = _response(
        json_data={
            "fieldSchemaList": [
                {"name": "id", "type": "INTEGER"},
                {"name": "score", "type": "DOUBLE"},
            ]
        }
    )
    schema = client.enforce_schema(sample_df, "ri.foundry.main.dataset.abc")
    assert "fieldSchemaList" in schema


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_session(client):
    with client as c:
        assert c is client
    client._session.close.assert_called_once()
