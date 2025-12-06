.PHONY: help install test clean lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make examples   - Run example scripts"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m unittest discover -s tests -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist
	rm -f *.wav *.mp3

lint:
	@echo "Running linters..."
	python -m py_compile zzchattts/*.py
	python -m py_compile examples/*.py
	python -m py_compile tests/*.py

format:
	@echo "Formatting code with black (if available)..."
	@which black > /dev/null && black zzchattts/ tests/ examples/ || echo "black not installed, skipping"

examples:
	@echo "Running basic example..."
	python examples/basic_example.py
