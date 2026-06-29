"""
WSGI entry point for gunicorn.

Usage:
    gunicorn api.wsgi:application --bind 0.0.0.0:8000 --workers 4
"""

from __future__ import annotations

from api.app import app, _load_models

_load_models()

application = app
