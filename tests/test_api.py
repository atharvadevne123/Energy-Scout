from __future__ import annotations

import json


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_health_models_not_loaded(self, client):
        r = client.get("/health")
        models = r.get_json()["models_loaded"]
        assert models["forecaster"] is False
        assert models["anomaly_detector"] is False

    def test_health_anomaly_key(self, client):
        assert "anomaly_detector" in client.get("/health").get_json()["models_loaded"]


class TestModelInfo:
    def test_model_info_structure(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.get_json()
        for key in ("forecaster_loaded", "anomaly_loaded", "version", "building_types"):
            assert key in data

    def test_model_info_version(self, client):
        assert client.get("/model/info").get_json()["version"] == "1.0.0"

    def test_model_info_building_types(self, client):
        data = client.get("/model/info").get_json()
        assert set(data["building_types"]) == {"residential", "commercial", "industrial", "data_center"}


class TestForecast:
    def test_forecast_valid(self, client, patch_models, valid_forecast_request):
        r = client.post("/forecast", data=json.dumps(valid_forecast_request),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "forecast_kwh" in data
        assert "demand_tier" in data
        assert "peak_demand_kwh" in data
        assert isinstance(data["forecast_kwh"], list)
        assert len(data["forecast_kwh"]) == 24
        assert data["demand_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_forecast_missing_meter_id(self, client, patch_models, valid_forecast_request):
        payload = {k: v for k, v in valid_forecast_request.items() if k != "meter_id"}
        r = client.post("/forecast", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_forecast_missing_building_type(self, client, patch_models, valid_forecast_request):
        payload = {k: v for k, v in valid_forecast_request.items() if k != "building_type"}
        r = client.post("/forecast", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_forecast_invalid_building_type(self, client, patch_models, valid_forecast_request):
        payload = {**valid_forecast_request, "building_type": "spaceship"}
        r = client.post("/forecast", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_forecast_humidity_out_of_range(self, client, patch_models, valid_forecast_request):
        payload = {**valid_forecast_request, "humidity_pct": 110.0}
        r = client.post("/forecast", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_forecast_occupancy_out_of_range(self, client, patch_models, valid_forecast_request):
        payload = {**valid_forecast_request, "occupancy_rate": 1.5}
        r = client.post("/forecast", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_forecast_response_has_request_id(self, client, patch_models, valid_forecast_request):
        r = client.post("/forecast", data=json.dumps(valid_forecast_request),
                        content_type="application/json")
        assert "X-Request-ID" in r.headers

    def test_forecast_horizon_out_of_range(self, client, patch_models, valid_forecast_request):
        payload = {**valid_forecast_request, "forecast_horizon_hours": 200}
        r = client.post("/forecast", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400


class TestAnomaly:
    def test_anomaly_valid(self, client, patch_models, valid_anomaly_request):
        r = client.post("/anomaly", data=json.dumps(valid_anomaly_request),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert isinstance(data["is_anomaly"], bool)

    def test_anomaly_missing_meter_id(self, client, patch_models, valid_anomaly_request):
        payload = {k: v for k, v in valid_anomaly_request.items() if k != "meter_id"}
        r = client.post("/anomaly", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_anomaly_missing_consumption(self, client, patch_models, valid_anomaly_request):
        payload = {k: v for k, v in valid_anomaly_request.items() if k != "consumption_kwh"}
        r = client.post("/anomaly", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_anomaly_negative_consumption(self, client, patch_models, valid_anomaly_request):
        payload = {**valid_anomaly_request, "consumption_kwh": -10.0}
        r = client.post("/anomaly", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400


class TestBatch:
    def test_batch_forecast_valid(self, client, patch_models, valid_forecast_request):
        payload = {"meters": [valid_forecast_request,
                              {**valid_forecast_request, "meter_id": "MTR-002"}]}
        r = client.post("/forecast/batch", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] == 2

    def test_batch_too_large(self, client, patch_models, valid_forecast_request):
        payload = {"meters": [valid_forecast_request] * 201}
        r = client.post("/forecast/batch", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400
