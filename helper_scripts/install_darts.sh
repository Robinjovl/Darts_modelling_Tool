#!/bin/bash
cp CHANGELOG.md darts
python3 setup.py clean
python3 setup.py build bdist_wheel
WHEEL_PATH=$(ls -t dist/*.whl | head -n1)
python3 -m pip install --no-deps --force-reinstall "file://$PWD/${WHEEL_PATH}"
