.PHONY: setup eda features train serve test
PYTHON ?= python3

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

eda:
	$(PYTHON) -m src.data.load

features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.models.train_xgb
	$(PYTHON) -m src.models.train_gnn

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q --cov=src --cov-report=term-missing --cov-report=json:reports/tests/coverage.json
