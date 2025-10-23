#!/bin/bash

# Parse args: -e (editable), --with-deps (install dependencies)
EDITABLE=0
WITH_DEPS=0
for arg in "$@"; do
  case "$arg" in
    -e|--editable)
      EDITABLE=1
      ;;
    --with-deps)
      WITH_DEPS=1
      ;;
  esac
done

cp CHANGELOG.md darts

if [ $EDITABLE -eq 1 ]; then
  if [ $WITH_DEPS -eq 1 ]; then
    python3 -m pip install -e .
  else
    python3 -m pip install --no-deps -e .
  fi
else
  python3 setup.py clean
  python3 setup.py build bdist_wheel
  WHEEL_PATH=$(ls -t dist/*.whl | head -n1)
  if [ $WITH_DEPS -eq 1 ]; then
    python3 -m pip install "file://$PWD/${WHEEL_PATH}"
  else
    python3 -m pip install --no-deps --force-reinstall "file://$PWD/${WHEEL_PATH}"
  fi
fi
