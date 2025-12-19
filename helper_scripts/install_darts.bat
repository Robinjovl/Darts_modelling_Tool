@echo off
setlocal enabledelayedexpansion

REM Parse args: -e (editable), -j/--jobs N, --with-deps (install dependencies)
set "EDITABLE=0"
set "WITH_DEPS=0"
set "JOBS=8"

:parse_args
if "%~1"=="" goto :args_done
set "ARG=%~1"
if /I "%ARG%"=="-e" (
  set "EDITABLE=1"
  shift
  goto :parse_args
)
if /I "%ARG%"=="--editable" (
  set "EDITABLE=1"
  shift
  goto :parse_args
)
if /I "%ARG%"=="--with-deps" (
  set "WITH_DEPS=1"
  shift
  goto :parse_args
)
if /I "%ARG%"=="--jobs" (
  if "%~2"=="" (
    echo Error: -j requires a numeric argument
    exit /b 1
  )
  call :__validate_num "%~2" || ( echo Error: -j requires a numeric argument & exit /b 1 )
  set "JOBS=%~2"
  shift
  shift
  goto :parse_args
)
if /I "%ARG:~0,2%"=="-j" (
  set "NUM=%ARG:~2%"
  if not "%NUM%"=="" (
    call :__validate_num "%NUM%" || ( echo Error: -j requires a numeric argument & exit /b 1 )
    set "JOBS=%NUM%"
    shift
    goto :parse_args
  ) else (
    if "%~2"=="" (
      echo Error: -j requires a numeric argument
      exit /b 1
    )
    call :__validate_num "%~2" || ( echo Error: -j requires a numeric argument & exit /b 1 )
    set "JOBS=%~2"
    shift
    shift
    goto :parse_args
  )
)
shift
goto :parse_args

:args_done

copy CHANGELOG.md darts || exit /b 1

echo Building C++ extensions...
if exist "build\" (
  pushd build
  make -j%JOBS% || exit /b 1
  make install || exit /b 1
  popd
)

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

rem reinstall the wheel (without dependencies to make it faster)
:foundwheel
if "%WITH_DEPS%"=="1" (
  python -m pip install "dist\%WHEEL%" || exit /b 1
) else (
  python -m pip install --no-deps --force-reinstall "dist\%WHEEL%" || exit /b 1
)

:__validate_num
REM Returns ERRORLEVEL 0 if argument is all digits, else 1
setlocal
set "VAL=%~1"
if "%VAL%"=="" ( endlocal & exit /b 1 )
for /f "delims=0123456789" %%i in ("%VAL%") do ( endlocal & exit /b 1 )
endlocal & exit /b 0
