.PHONY: install train simulate serve serve-docker test test-cov lint clean

# ── Environment ────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Pipeline & Simulation ──────────────────────────────────────────────────
train:
	python scripts/train_forecast.py --n-days 730

simulate:
	python scripts/run_simulation.py

# ── Serving ────────────────────────────────────────────────────────────────
serve:
	uvicorn src.api.main:app --reload --port 8000

serve-docker:
	docker compose up --build

# ── Testing ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --asyncio-mode=auto

test-cov:
	pytest tests/ -v --asyncio-mode=auto --cov=src --cov-report=term-missing

# ── Code quality ───────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
