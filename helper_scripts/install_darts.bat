@echo off
setlocal enabledelayedexpansion

REM Parse args: -e (editable), --with-deps (install dependencies)
set "EDITABLE=0"
set "WITH_DEPS=0"
for %%A in (%*) do (
  if /I "%%~A"=="-e" set "EDITABLE=1"
  if /I "%%~A"=="--editable" set "EDITABLE=1"
  if /I "%%~A"=="--with-deps" set "WITH_DEPS=1"
)

copy CHANGELOG.md darts || exit /b 1

if "%EDITABLE%"=="1" (
  if "%WITH_DEPS%"=="1" (
    python -m pip install -e . || exit /b 1
  ) else (
    python -m pip install --no-deps -e . || exit /b 1
  )
  exit /b 0
)

python setup.py clean || exit /b 1
python setup.py build bdist_wheel || exit /b 1

set "WHEEL="
for /f "delims=" %%F in ('dir /b /a:-d /o:-d dist\*.whl') do (
    set "WHEEL=%%F"
    goto :foundwheel
)
echo No wheel found in dist\*.whl
exit /b 1

:foundwheel
if "%WITH_DEPS%"=="1" (
  python -m pip install "dist\%WHEEL%" || exit /b 1
) else (
  python -m pip install --no-deps --force-reinstall "dist\%WHEEL%" || exit /b 1
)
