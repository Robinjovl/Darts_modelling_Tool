---
name: test-open-darts
description: Execute open-DARTS regression and component tests, including model suite, interpolator tests, discretizer tests, and ctest. Use when validating behavioral changes or reproducing test failures.
---

# Test open-DARTS

Use this skill for regression-safe validation.

## Steps

1. Run the model regression suite.
2. Run interpolator and discretizer tests.
3. Collect logs and identify failing model categories.

## Primary commands

- Model suite: `cd models && darts run_test_suite2.py LOG`
- Interpolator tests: `cd tests/engines/src/interpolation && darts test_interpolation.py`
- Discretizer tests: `cd discretizer/tests/compare_discretizers && darts main.py`

## Reference

Read `references/testing.md` for test matrix, artifacts, and troubleshooting.
