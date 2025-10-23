#!/bin/bash
./helper_scripts/install_darts.sh "$@"

# Install dev tools once if missing (from [dev] extras): ruff, pre-commit
if ! python3 -m pip show ruff >/dev/null 2>&1 || ! python3 -m pip show pre-commit >/dev/null 2>&1; then
python3 -m pip install --upgrade ruff pre-commit
fi
if command -v pre-commit >/dev/null 2>&1; then
pre-commit install
fi
