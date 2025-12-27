call helper_scripts\install_darts.bat %*

rem One-time dev tools install (ruff, pre-commit) if missing
set "NEED_DEV=1"
python -m pip show ruff >nul 2>&1 || set "NEED_DEV=1"
python -m pip show pre-commit >nul 2>&1 || set "NEED_DEV=1"
if defined NEED_DEV (
    python -m pip install --upgrade ruff pre-commit || exit /b 1
)
python -m pip show pre-commit >nul 2>&1 && python -m pre_commit install
