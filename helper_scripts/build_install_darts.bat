copy CHANGELOG.md darts
python setup.py clean
python setup.py build bdist_wheel
python -m pip install --upgrade --no-deps --force-reinstall dist/open_darts-1.3.1-cp310-cp310-win_amd64.whl
