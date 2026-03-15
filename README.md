# ai_swing_trader

`ai_swing_trader` is a production-oriented Python project scaffold for building an algorithmic swing trading system.

## Structure

- `ai_swing_trader/core/` - trading logic, strategy, and domain models
- `ai_swing_trader/config/` - configuration loaders and schema definitions
- `ai_swing_trader/infra/` - infrastructure adapters (brokers, data providers, persistence)
- `tests/` - automated test suite
- `notebooks/` - research and experimentation notebooks

## Requirements

- Python 3.11+

## Setup

Install the project and development dependencies:

```bash
pip install -e .[dev]
```

Install git hooks:

```bash
pre-commit install
```

## Development Commands

Run tests:

```bash
pytest
```

Format code:

```bash
black .
```

Lint code:

```bash
ruff check .
```

Run all pre-commit hooks:

```bash
pre-commit run --all-files
```
