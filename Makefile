.PHONY: install install-dev lint format test train app clean help

# Default target
help:
	@echo "Queens Solver - Available commands:"
	@echo ""
	@echo "  make install      Install package in editable mode"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make lint         Run linters (ruff)"
	@echo "  make format       Format code with black"
	@echo "  make test         Run tests"
	@echo "  make train        Run HRM training"
	@echo "  make app          Launch Gradio web interface"
	@echo "  make clean        Remove build artifacts"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

# Code quality
lint:
	ruff check src/ scripts/ app/

format:
	black src/ scripts/ app/

# Testing
test:
	pytest tests/ -v

# Training
train:
	python scripts/train.py

ablation:
	python scripts/ablation.py

sweep:
	python scripts/sweep.py

# Web app
app:
	python app/gradio_app.py

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
