# AI Coding-Agent Harness

A Python harness for orchestrating an AI coding agent with tool-use capabilities.

## Quick Start

```bash
# 1. Copy the env template and fill in your API key
cp .env.example .env

# 2. Install dependencies
make setup

# 3. Run the harness
make run
```

## Project Structure

```
├── Makefile              # Build & run targets
├── config.yaml           # Model and runtime configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── main.py           # Entry point
│   ├── model_client.py   # LLM API client
│   ├── orchestrator.py   # Agent loop orchestrator
│   ├── context_manager.py# Conversation context management
│   ├── verifier.py       # Output verification
│   └── tools/
│       ├── file_ops.py   # File read/write operations
│       ├── shell.py      # Shell command execution
│       └── search.py     # Code/file search
└── tests/
    └── test_harness.py   # Test suite
```

## Available Make Targets

| Target        | Description                              |
|---------------|------------------------------------------|
| `make setup`  | Install Python dependencies              |
| `make run`    | Launch the harness                       |
| `make test`   | Run the test suite with pytest           |
| `make clean`  | Remove `__pycache__` and pytest caches   |

## Configuration

Edit `config.yaml` to set the model, iteration limits, and token budget.

## Environment Variables

| Variable      | Required | Description          |
|---------------|----------|----------------------|
| `AI_API_KEY`  | Yes      | API key for the LLM  |
