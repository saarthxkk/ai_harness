.PHONY: setup run test clean

setup:
	pip install -r requirements.txt --break-system-packages

run:
	AI_API_KEY=$(AI_API_KEY) python src/main.py

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
