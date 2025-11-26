#!/bin/bash

# add CHANGELOG to a wheel
cp CHANGELOG.md darts

# build a wheel
python3 -m build --wheel

# reinstall the wheel (without dependencies to make it faster)
WHEEL_PATH=$(ls -t dist/*.whl | head -n1)
python3 -m pip install --no-deps --force-reinstall "file://$PWD/${WHEEL_PATH}"
