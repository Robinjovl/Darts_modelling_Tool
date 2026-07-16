---
name: verify-open-darts
description: Run the mandatory verification stack when changes affect runtime code, numerical behavior, tests, packaging, or build/test behavior in open-DARTS. Trigger for edits under darts/, engines/, discretizer/, linear_solvers/, models/, tests/, helper_scripts/, pyproject.toml, setup.py, CMakeLists.txt, .gitlab-ci.yml, or .cicd/jobs/.
---

# Verify open-DARTS

Use this skill to choose and run the minimum required validation before handoff.
Use the session conda environment. If the prompt defines one, use it. Otherwise, create and activate one session-level conda environment once, then reuse it across the rest of the session.

## Steps

1. Confirm the session conda environment is active; create it once if needed.
2. Classify touched files and map them to the required checks.
3. Run the relevant lint, build, and test commands from the repo root.
4. Do not mark the task complete until required checks pass or an explicit blocker is documented.

## Primary commands

- Lint changed Python files: `pre-commit run -v --files /absolute/path/to/file.py --show-diff-on-failure`
- Editable install after build-impacting changes: `./helper_scripts/install_darts.sh -e`
- Model regression suite: `cd models && darts run_test_suite2.py LOG`
- Interpolator tests: `cd tests/engines/src/interpolation && darts test_interpolation.py`
- Discretizer tests: `cd discretizer/tests/compare_discretizers && darts main.py`

## Reference

Read `references/verification.md` for the validation matrix, path-based triggers, and reporting rules.
