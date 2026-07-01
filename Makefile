.DEFAULT_GOAL := help
PORT ?= 8003

.PHONY: help install test lint format train run docker-up docker-down clean

help:
	@echo "Energy-Scout — available targets:"
	@echo "  install      Install all dependencies"
	@echo "  test         Run pytest suite"
	@echo "  lint         Check code with ruff"
	@echo "  format       Auto-format with ruff"
	@echo "  train        Train forecaster + anomaly detector on synthetic data"
	@echo "  run          Start API with gunicorn (PORT=$(PORT))"
	@echo "  docker-up    Start full stack via docker-compose"
	@echo "  docker-down  Stop docker-compose stack"
	@echo "  clean        Remove build artefacts and cache"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

lint:
	ruff check . --ignore E501,F401

format:
	ruff format .
	ruff check . --fix --ignore E501,F401

train:
	python scripts/train.py

run:
	gunicorn -b 0.0.0.0:$(PORT) api.wsgi:app --workers 2 --timeout 120

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
