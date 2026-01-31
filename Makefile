.PHONY: help format lint type-check test clean all

help:
	@echo "Model Evaluation Suite - Development Commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  format      - Format code with ruff"
	@echo "  lint        - Run ruff linter (auto-fix safe issues)"
	@echo "  lint-check  - Run ruff linter (check only, no fixes)"
	@echo "  type-check  - Run mypy type checker"
	@echo "  test        - Run pytest with coverage"
	@echo "  test-fast   - Run pytest without coverage"
	@echo "  clean       - Remove build artifacts and caches"
	@echo "  all         - Run format, lint, type-check, and test"

format:
	@echo "🎨 Formatting code with ruff..."
	ruff format src/ tests/

lint:
	@echo "🔍 Linting code with ruff (auto-fixing)..."
	ruff check src/ tests/ --fix

lint-check:
	@echo "🔍 Checking code with ruff (no fixes)..."
	ruff check src/ tests/

type-check:
	@echo "🔬 Type checking with mypy..."
	mypy src/model_eval_suite/

test:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ src/model_eval_suite/tests/ -v --cov=src/model_eval_suite --cov-report=term-missing

test-fast:
	@echo "⚡ Running tests (no coverage)..."
	pytest tests/ src/model_eval_suite/tests/ -v --no-cov

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint type-check test
	@echo "✅ All checks passed!"
