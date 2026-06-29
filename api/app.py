"""
Flask microservice: Energy-Scout API

Endpoints:
  POST /forecast          — forecast energy consumption for next N hours
  POST /forecast/batch    — batch forecasts for multiple buildings/meters
  POST /anomaly           — detect anomalies in consumption readings
  GET  /efficiency        — efficiency recommendations (?building_id=B001&period=7d)
  GET  /health            — liveness probe
  GET  /metrics           — Prometheus metrics
  GET  /model/info        — model metadata
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api, Resource, fields
from loguru import logger
from marshmallow import Schema, ValidationError, fields as ma_fields, validate as ma_validate
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ─── App bootstrap ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["300 per minute"],
    storage_uri="memory://",
)

api = Api(
    app,
    version="1.0",
    title="Energy-Scout API",
    description="Real-time energy consumption forecasting and anomaly detection",
    doc="/docs",
)

ns = api.namespace("", description="Energy-Scout endpoints")

# ─── Prometheus metrics ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "energy_scout_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
PREDICTION_LATENCY = Histogram(
    "energy_scout_latency_seconds",
    "Request latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
FORECAST_KWH_HISTOGRAM = Histogram(
    "energy_scout_forecast_kwh",
    "Distribution of forecasted peak demand",
    buckets=[0, 100, 250, 500, 1000, 2000, 5000, 10000],
)

# ─── Demand tier thresholds ───────────────────────────────────────────────────

TIER_CRITICAL = float(os.getenv("TIER_CRITICAL", "5000"))   # kWh peak
TIER_HIGH     = float(os.getenv("TIER_HIGH",     "2000"))
TIER_MEDIUM   = float(os.getenv("TIER_MEDIUM",    "500"))

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent.parent / "models")))

_forecaster = None
_anomaly_detector = None
_feature_engineer = None


def _load_models() -> None:
    global _forecaster, _anomaly_detector, _feature_engineer
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import joblib
    from models.ensemble.demand_forecaster import DemandForecaster
    from models.anomaly.consumption_anomaly import ConsumptionAnomalyDetector

    forecaster_path = MODEL_DIR / "ensemble" / "artifacts" / "demand_forecaster.joblib"
    anomaly_path    = MODEL_DIR / "anomaly" / "artifacts" / "consumption_anomaly.joblib"
    fe_path         = MODEL_DIR / "feature_engineer.joblib"

    if forecaster_path.exists():
        _forecaster = DemandForecaster.load(forecaster_path)
        logger.success("DemandForecaster loaded.")
    else:
        logger.warning("Demand forecaster not found at {}. Using mock predictions.", forecaster_path)

    if anomaly_path.exists():
        _anomaly_detector = ConsumptionAnomalyDetector.load(anomaly_path)
        logger.success("ConsumptionAnomalyDetector loaded.")
    else:
        logger.warning("Anomaly detector not found at {}.", anomaly_path)

    if fe_path.exists():
        _feature_engineer = joblib.load(fe_path)
        logger.success("Feature engineer loaded.")


# ─── Input validation schemas ─────────────────────────────────────────────────

_BUILDING_TYPES = {"residential", "commercial", "industrial", "data_center"}


class ForecastSchema(Schema):
    meter_id              = ma_fields.Str(required=True)
    building_type         = ma_fields.Str(
        required=True,
        validate=ma_validate.OneOf(list(_BUILDING_TYPES)),
    )
    timestamp             = ma_fields.Str(required=True)
    temperature_c         = ma_fields.Float(load_default=None)
    humidity_pct          = ma_fields.Float(
        load_default=None,
        validate=ma_validate.Range(min=0, max=100, error="humidity_pct must be 0–100"),
    )
    occupancy_rate        = ma_fields.Float(
        load_default=None,
        validate=ma_validate.Range(min=0, max=1, error="occupancy_rate must be 0–1"),
    )
    day_of_week           = ma_fields.Int(load_default=None,
                                validate=ma_validate.Range(min=0, max=6))
    hour                  = ma_fields.Int(load_default=None,
                                validate=ma_validate.Range(min=0, max=23))
    month                 = ma_fields.Int(load_default=None,
                                validate=ma_validate.Range(min=1, max=12))
    is_holiday            = ma_fields.Bool(load_default=False)
    solar_generation_kw   = ma_fields.Float(
        load_default=0.0,
        validate=ma_validate.Range(min=0, error="solar_generation_kw must be >= 0"),
    )
    forecast_horizon_hours = ma_fields.Int(
        load_default=24,
        validate=ma_validate.Range(min=1, max=168, error="forecast_horizon_hours must be 1–168"),
    )


class AnomalySchema(Schema):
    meter_id         = ma_fields.Str(required=True)
    timestamp        = ma_fields.Str(required=True)
    consumption_kwh  = ma_fields.Float(
        required=True,
        validate=ma_validate.Range(min=0, error="consumption_kwh must be >= 0"),
    )
    expected_kwh     = ma_fields.Float(load_default=None)
    building_type    = ma_fields.Str(
        load_default="commercial",
        validate=ma_validate.OneOf(list(_BUILDING_TYPES)),
    )
    temperature_c    = ma_fields.Float(load_default=None)


_forecast_schema       = ForecastSchema()
_forecast_batch_schema = ForecastSchema(many=True)
_anomaly_schema        = AnomalySchema()

# ─── Swagger models ───────────────────────────────────────────────────────────

forecast_input_model = api.model("ForecastInput", {
    "meter_id":               fields.String(required=True, example="MTR-001"),
    "building_type":          fields.String(required=True, example="commercial"),
    "timestamp":              fields.String(required=True, example="2024-06-15T14:00:00"),
    "temperature_c":          fields.Float(example=22.5),
    "humidity_pct":           fields.Float(example=55.0),
    "occupancy_rate":         fields.Float(example=0.8),
    "day_of_week":            fields.Integer(example=1),
    "hour":                   fields.Integer(example=14),
    "month":                  fields.Integer(example=6),
    "is_holiday":             fields.Boolean(example=False),
    "solar_generation_kw":    fields.Float(example=12.5),
    "forecast_horizon_hours": fields.Integer(example=24),
})

forecast_response_model = api.model("ForecastResponse", {
    "request_id":          fields.String(),
    "meter_id":            fields.String(),
    "forecast_kwh":        fields.List(fields.Float()),
    "confidence_interval": fields.Raw(),
    "peak_demand_kwh":     fields.Float(),
    "demand_tier":         fields.String(),
    "forecast_horizon_hours": fields.Integer(),
    "latency_ms":          fields.Float(),
})

anomaly_input_model = api.model("AnomalyInput", {
    "meter_id":        fields.String(required=True, example="MTR-001"),
    "timestamp":       fields.String(required=True, example="2024-06-15T14:00:00"),
    "consumption_kwh": fields.Float(required=True, example=750.5),
    "expected_kwh":    fields.Float(example=500.0),
    "building_type":   fields.String(example="commercial"),
    "temperature_c":   fields.Float(example=22.5),
})

anomaly_response_model = api.model("AnomalyResponse", {
    "request_id":    fields.String(),
    "meter_id":      fields.String(),
    "is_anomaly":    fields.Boolean(),
    "anomaly_score": fields.Float(),
    "anomaly_type":  fields.String(),
    "z_score":       fields.Float(),
    "latency_ms":    fields.Float(),
})

# ─── Request ID middleware ────────────────────────────────────────────────────

@app.before_request
def _attach_request_id():
    g.request_id = str(uuid.uuid4())


@app.after_request
def _add_request_id_header(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response


# ─── Rate limit exemptions ────────────────────────────────────────────────────

@limiter.request_filter
def _exempt_observability():
    return request.path in ("/health", "/metrics") or request.path.startswith("/docs")


# ─── Error handlers ───────────────────────────────────────────────────────────

@api.errorhandler(Exception)
def handle_generic(e):
    code = getattr(e, "code", 500)
    return {"error": type(e).__name__, "detail": str(e)}, code


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "detail": str(e)}), 404


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded", "detail": str(e)}), 429


# ─── Core forecast logic ──────────────────────────────────────────────────────

def _run_forecast(data: dict) -> dict:
    """Produce energy forecast from a validated request dict."""
    start = time.perf_counter()
    from pipeline.feature_engineering import build_forecast_features

    horizon = data.get("forecast_horizon_hours", 24)
    meter_id = data.get("meter_id")

    # Build feature matrix
    if _feature_engineer is not None:
        X = _feature_engineer.transform(pd.DataFrame([data]))
    else:
        X = build_forecast_features(data)

    # Forecast
    if _forecaster is not None:
        forecast_result = _forecaster.forecast(X, horizon=horizon)
        forecast_kwh = forecast_result["forecast_kwh"]
        lower_kwh    = forecast_result["lower_kwh"]
        upper_kwh    = forecast_result["upper_kwh"]
        peak_demand  = forecast_result["peak_demand_kwh"]
    else:
        # Mock fallback: realistic building-type defaults
        base = {"residential": 250, "commercial": 500, "industrial": 2000, "data_center": 3000}
        base_kw = base.get(data.get("building_type", "commercial"), 500)
        import random
        forecast_kwh = [round(base_kw + random.uniform(-50, 50), 2) for _ in range(horizon)]
        lower_kwh    = [round(v * 0.85, 2) for v in forecast_kwh]
        upper_kwh    = [round(v * 1.15, 2) for v in forecast_kwh]
        peak_demand  = max(forecast_kwh)

    demand_tier = _compute_demand_tier(peak_demand)
    FORECAST_KWH_HISTOGRAM.observe(peak_demand)

    latency_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNT.labels(endpoint="forecast", status="200").inc()

    return {
        "request_id":   getattr(g, "request_id", str(uuid.uuid4())),
        "meter_id":     meter_id,
        "forecast_kwh": forecast_kwh,
        "confidence_interval": {
            "lower_kwh": lower_kwh,
            "upper_kwh": upper_kwh,
        },
        "peak_demand_kwh":     round(peak_demand, 3),
        "demand_tier":         demand_tier,
        "forecast_horizon_hours": horizon,
        "latency_ms":          round(latency_ms, 2),
    }


def _run_anomaly(data: dict) -> dict:
    """Run anomaly detection on a meter reading."""
    start = time.perf_counter()

    reading_df = pd.DataFrame([{
        "consumption_kwh": data["consumption_kwh"],
        "expected_kwh":    data.get("expected_kwh", data["consumption_kwh"]),
        "temperature_c":   data.get("temperature_c", 20.0),
        "building_type":   data.get("building_type", "commercial"),
    }])

    if _anomaly_detector is not None:
        results = _anomaly_detector.detect(reading_df)
        r = results[0]
        anomaly_score = r["anomaly_score"]
        is_anomaly    = r["is_anomaly"]
        anomaly_type  = r["anomaly_type"]
        z_score       = r["z_score"]
    else:
        # Mock fallback
        consumption = data["consumption_kwh"]
        expected    = data.get("expected_kwh", consumption)
        ratio       = consumption / expected if expected > 0 else 1.0
        anomaly_score = min(abs(ratio - 1.0), 1.0)
        is_anomaly    = anomaly_score > 0.3
        z_score       = (consumption - expected) / max(expected * 0.1, 1.0)
        if is_anomaly:
            if ratio > 1.5:
                anomaly_type = "spike"
            elif ratio < 0.5:
                anomaly_type = "drop"
            else:
                anomaly_type = "pattern_shift"
        else:
            anomaly_type = "none"

    latency_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNT.labels(endpoint="anomaly", status="200").inc()

    return {
        "request_id":   getattr(g, "request_id", str(uuid.uuid4())),
        "meter_id":     data.get("meter_id"),
        "is_anomaly":   bool(is_anomaly),
        "anomaly_score": round(float(anomaly_score), 6),
        "anomaly_type": anomaly_type,
        "z_score":      round(float(z_score), 4),
        "latency_ms":   round(latency_ms, 2),
    }


def _compute_demand_tier(peak_kwh: float) -> str:
    if peak_kwh >= TIER_CRITICAL:
        return "CRITICAL"
    if peak_kwh >= TIER_HIGH:
        return "HIGH"
    if peak_kwh >= TIER_MEDIUM:
        return "MEDIUM"
    return "LOW"


# ─── Routes ───────────────────────────────────────────────────────────────────

@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        return {
            "status": "ok",
            "models_loaded": {
                "forecaster":       _forecaster is not None,
                "anomaly_detector": _anomaly_detector is not None,
            },
        }, 200


@ns.route("/metrics")
class Metrics(Resource):
    def get(self):
        from flask import Response
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@ns.route("/model/info")
class ModelInfo(Resource):
    def get(self):
        return {
            "version":              "1.0.0",
            "forecaster_loaded":    _forecaster is not None,
            "anomaly_loaded":       _anomaly_detector is not None,
            "feature_engineer":     _feature_engineer is not None,
            "supported_horizons":   "1–168 hours",
            "building_types":       list(_BUILDING_TYPES),
        }, 200


@ns.route("/forecast")
class Forecast(Resource):
    @ns.expect(forecast_input_model)
    @ns.marshal_with(forecast_response_model, code=200)
    def post(self):
        try:
            data = _forecast_schema.load(request.get_json(force=True) or {})
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="forecast", status="400").inc()
            return {"error": str(exc.messages)}, 400

        with PREDICTION_LATENCY.time():
            result = _run_forecast(data)

        return result, 200


@ns.route("/forecast/batch")
class ForecastBatch(Resource):
    def post(self):
        payload = request.get_json(force=True) or {}
        meters = payload.get("meters")
        if meters is None:
            return {"error": "Missing 'meters' key."}, 400
        if not meters:
            return {"error": "No meter readings provided."}, 400
        if len(meters) > 200:
            return {"error": "Batch size limited to 200."}, 400

        try:
            data_list = _forecast_batch_schema.load(meters)
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="forecast_batch", status="400").inc()
            return {"error": str(exc.messages)}, 400

        results = []
        with PREDICTION_LATENCY.time():
            for item in data_list:
                results.append(_run_forecast(item))

        REQUEST_COUNT.labels(endpoint="forecast_batch", status="200").inc()
        return {"results": results, "count": len(results)}, 200


@ns.route("/anomaly")
class AnomalyDetect(Resource):
    @ns.expect(anomaly_input_model)
    @ns.marshal_with(anomaly_response_model, code=200)
    def post(self):
        try:
            data = _anomaly_schema.load(request.get_json(force=True) or {})
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="anomaly", status="400").inc()
            return {"error": str(exc.messages)}, 400

        with PREDICTION_LATENCY.time():
            result = _run_anomaly(data)

        return result, 200


@ns.route("/efficiency")
class EfficiencyRecommendations(Resource):
    def get(self):
        building_id = request.args.get("building_id", "UNKNOWN")
        period = request.args.get("period", "7d")

        # In production these come from aggregating historical forecasts +
        # anomalies from the data store (Foundry / Redis).
        recommendations = _generate_efficiency_recommendations(building_id, period)

        REQUEST_COUNT.labels(endpoint="efficiency", status="200").inc()
        return {
            "building_id":     building_id,
            "period":          period,
            "recommendations": recommendations,
            "generated_at":    pd.Timestamp.utcnow().isoformat(),
        }, 200


def _generate_efficiency_recommendations(building_id: str, period: str) -> list[dict]:
    """Generate efficiency recommendations (rule-based + model-informed)."""
    return [
        {
            "priority": "HIGH",
            "category": "HVAC",
            "recommendation": "Shift HVAC pre-cooling to 6–8 AM off-peak window to reduce peak demand by ~15%.",
            "estimated_savings_pct": 15.0,
            "implementation_complexity": "LOW",
        },
        {
            "priority": "MEDIUM",
            "category": "Lighting",
            "recommendation": "Install occupancy-based lighting controls in zones with <30% detected occupancy during business hours.",
            "estimated_savings_pct": 8.0,
            "implementation_complexity": "LOW",
        },
        {
            "priority": "MEDIUM",
            "category": "Equipment Scheduling",
            "recommendation": "Schedule non-critical server maintenance tasks during 11 PM–4 AM off-peak window.",
            "estimated_savings_pct": 5.0,
            "implementation_complexity": "MEDIUM",
        },
        {
            "priority": "LOW",
            "category": "Solar Optimization",
            "recommendation": "Increase solar storage utilization during 12–2 PM peak generation to offset evening grid demand.",
            "estimated_savings_pct": 3.5,
            "implementation_complexity": "MEDIUM",
        },
    ]


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
