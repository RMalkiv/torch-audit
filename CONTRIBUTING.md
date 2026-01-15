# Contributing to torch-audit

## Adding a New Validator
1. **Choose a Pack**: 
   - Stability (`TA1xx`), Hardware (`TA2xx`), Data (`TA3xx`), or Architecture (`TA4xx`).
2. **Define the Rule**:
   - Create a `Rule` object with a unique ID and remediation advice.
3. **Implement**:
   - Inherit from `BaseValidator`.
   - Implement `check(context)`.
4. **Test**:
   - Add a test case in `tests/validators/`.
   - Ensure you test both the "clean" case and the "failure" case.

## Run the Tests
```bash
poetry run pytest