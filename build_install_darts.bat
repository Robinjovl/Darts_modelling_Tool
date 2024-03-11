python setup.py clean
python setup.py build bdist_wheel
python -m pip install --upgrade --no-deps --force-reinstall dist/open_darts-1.0.5-cp311-cp311-win_amd64.whl
