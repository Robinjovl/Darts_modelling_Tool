#!/bin/bash
python3 setup.py clean
python3 setup.py build bdist_wheel
WHEEL_PATH=$(ls -t dist/*.whl | head -n1)
python3 -m pip install --no-deps --force-reinstall "file://$PWD/${WHEEL_PATH}"
# Install dev tools once if missing (from [dev] extras): ruff, pre-commit
if ! python3 -m pip show ruff >/dev/null 2>&1 || ! python3 -m pip show pre-commit >/dev/null 2>&1; then
python3 -m pip install --upgrade ruff pre-commit
fi
if command -v pre-commit >/dev/null 2>&1; then
pre-commit install
fi
