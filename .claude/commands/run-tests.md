---
description: Run the full test suite with coverage inside Docker
allowed-tools: Bash, Read
---

# Run Tests

Execute the full pytest suite with coverage reporting.

## Steps

1. Ensure docker-compose is running: `docker-compose ps`
2. Run tests inside the app container:
   ```bash
   docker-compose exec app pytest -v --cov=backend/app --cov-report=term-missing 2>&1
   ```
3. If docker-compose isn't running, run locally:
   ```bash
   cd backend && pytest -v --cov=app --cov-report=term-missing 2>&1
   ```
4. Parse the output and report:
   - Total tests: passing / failing / skipped
   - Coverage: overall % + any module below 75%
   - For any failures: show the test name + error message

## Coverage thresholds
- Overall: ≥ 75% (hard fail in CI)
- backend/app/ml/: ≥ 80%
- backend/app/routers/: ≥ 80%

If any threshold is missed, spawn the test-engineer agent to add the missing tests.
