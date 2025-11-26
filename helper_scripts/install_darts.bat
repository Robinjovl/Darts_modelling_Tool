rem add CHANGELOG to a wheel
copy CHANGELOG.md darts || exit /b 1

rem Build a wheel
python -m build --wheel || exit /b 1

setlocal enabledelayedexpansion
set "WHEEL="
for /f "delims=" %%F in ('dir /b /a:-d /o:-d dist\*.whl') do (
    set "WHEEL=%%F"
    goto :foundwheel
)
echo No wheel found in dist\*.whl
exit /b 1

rem reinstall the wheel (without dependencies to make it faster)
:foundwheel
python -m pip install --no-deps --force-reinstall "dist\%WHEEL%" || exit /b 1
