.PHONY: setup run test clean demo

setup:
	python3 -m pip install -r requirements.txt --break-system-packages

run:
	AI_API_KEY=$(AI_API_KEY) python3 src/main.py

demo:
	python3 src/demo.py

test:
	python3 -m pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
