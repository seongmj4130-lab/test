# Makefile for Quant Trading System
.PHONY: help install format lint test clean docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  format      Format code with black and isort"
	@echo "  lint        Lint code with ruff"
	@echo "  test        Run tests with pytest"
	@echo "  typecheck   Run type checking with mypy"
	@echo "  precommit   Run pre-commit on all files"
	@echo "  clean       Clean cache files"
	@echo "  docs        Generate documentation"

# Installation
install:
	pip install -e ".[dev,docs]"

# Code formatting
format:
	black src tests scripts experiments
	isort --profile black src tests scripts experiments

# Linting
lint:
	ruff check src tests scripts experiments
	ruff format --check src tests scripts experiments

# Testing
test:
	pytest tests/ -v

# Type checking
typecheck:
	mypy src

# Pre-commit
precommit:
	pre-commit run --all-files

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build/
	rm -rf dist/

# Documentation
docs:
	sphinx-build docs docs/_build/html

# Development setup
setup: install
	pre-commit install
	@echo "Development environment setup complete!"

# CI/CD
ci:
	black --check src tests
	ruff format --check src tests
	pytest tests/test_pipeline/ -m ci
	python -m compileall src/components src/core src/interfaces src/pipeline src/utils src/tracks tests/test_pipeline
	@echo "CI checks passed!"

# Quick checks
check: lint test
	@echo "Basic checks passed!"
