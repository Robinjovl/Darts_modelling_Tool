copy CHANGELOG.md darts || exit /b 1
python setup.py clean || exit /b 1
python setup.py build bdist_wheel || exit /b 1

setlocal enabledelayedexpansion
set "WHEEL="
for /f "delims=" %%F in ('dir /b /a:-d /o:-d dist\*.whl') do (
    set "WHEEL=%%F"
    goto :foundwheel
)
echo No wheel found in dist\*.whl
exit /b 1

:foundwheel
python -m pip install --no-deps --force-reinstall "dist\%WHEEL%" || exit /b 1

rem One-time dev tools install (ruff, pre-commit) if missing
set "NEED_DEV=1"
python -m pip show ruff >nul 2>&1 || set "NEED_DEV=1"
python -m pip show pre-commit >nul 2>&1 || set "NEED_DEV=1"
if defined NEED_DEV (
    python -m pip install --upgrade ruff pre-commit || exit /b 1
)
python -m pip show pre-commit >nul 2>&1 && python -m pre_commit install
