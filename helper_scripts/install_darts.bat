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
