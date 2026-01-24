# Contributing to torch-audit

Thanks for helping improve **torch-audit**!

## Development setup

This project uses **Poetry**.

```bash
poetry install --with dev
```

Run the test suite:

```bash
poetry run pytest
```

## Linting & formatting

```bash
poetry run ruff check .
poetry run black --check .
```

Auto-format:

```bash
poetry run ruff check . --fix
poetry run black .
```

## Pre-commit (recommended)

```bash
poetry run pip install pre-commit
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Adding a new validator / rule

### 1) Pick a rule ID

Rule IDs are grouped by pack:

- `TA1xx` — Stability (NaNs, Infs, exploding grads, dead units)
- `TA2xx` — Hardware (device/layout issues, tensor core hints, split-brain)
- `TA3xx` — Data integrity (bad ranges, invalid tokens, device mismatch)
- `TA4xx` — Architecture & optimization (weight decay pitfalls, redundant bias)
- `TA5xx` — Runtime graph / hooks (unused layers, stateful reuse)

### 2) Define the Rule

Create a `Rule` with:

- **clear title** (short)
- **description** (what is happening)
- **remediation** (what to do next)
- **category** (pack)
- **default severity**

Also register it:

```python
RuleRegistry.register(MY_RULE)
```

### 3) Implement the validator

Inherit from `BaseValidator` and implement `check(context)`.

Design principles:

- **Never mutate** the model, parameters, grads, or batch.
- Keep checks **cheap** by default (avoid GPU sync unless necessary).
- Put heavy work behind `torch.no_grad()`.
- Metadata should be **JSON-serializable** (avoid tensors, modules, callables).

### 4) Tests

Add tests under `tests/validators/`.

- One test for the **clean** case
- One test for the **finding** case

If a test requires CUDA, use `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.

## Updating docs

If you add rules or change behavior:

- Update `README.md` examples if needed
- Update `CHANGELOG.md`
- Regenerate `RULES.md` (see `scripts/generate_rules.py`)

## Release checklist (maintainers)

- [ ] `poetry run ruff check .`
- [ ] `poetry run black --check .`
- [ ] `poetry run pytest`
- [ ] `poetry build` + `poetry run twine check dist/*`
- [ ] Update version in `pyproject.toml` + `CHANGELOG.md`
- [ ] Tag `vX.Y.Z`
