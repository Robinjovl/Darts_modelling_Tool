python setup.py clean
python setup.py build bdist_wheel
python -m pip install --upgrade --no-deps --force-reinstall dist/open_darts-1.2.1-cp39-cp39-win_amd64.whl
