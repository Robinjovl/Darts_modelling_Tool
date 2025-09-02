python -m pip install --upgrade pip || exit /b 1
python -m pip install -e ".[dev]" || exit /b 1
python -m pre_commit install || exit /b 1
