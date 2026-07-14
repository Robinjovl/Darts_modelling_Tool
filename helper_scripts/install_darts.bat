@echo off
setlocal enabledelayedexpansion

REM Parse args: -e (editable), -j/--jobs N, --with-deps (install dependencies)
set "EDITABLE=0"
set "WITH_DEPS=0"
set "JOBS=8"

python -m pip install build

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

rem add CHANGELOG to a wheel
copy CHANGELOG.md darts || exit /b 1

echo Building C++ extensions...
if exist "build\" (
  pushd build

  rem Detect build configuration (default to Release)
  set "BUILD_CONFIG=Release"
  if exist "CMakeCache.txt" (
    for /f "tokens=2 delims==" %%I in ('findstr /b /c:"CMAKE_BUILD_TYPE:STRING=" CMakeCache.txt') do (
      if not "%%~I"=="" (
        set "BUILD_CONFIG=%%~I"
      )
    )
  )

  rem Try GNU Make if available, otherwise fall back to CMake-driven build (msbuild/Ninja)
  set "BUILT_WITH_TOOL=0"
  if exist "Makefile" (
    where make >nul 2>&1
    if %errorlevel%==0 (
      echo -- Building via make with %JOBS% jobs
      make -j%JOBS% || exit /b 1
      make install || exit /b 1
      set "BUILT_WITH_TOOL=1"
    ) else (
      echo -- 'make' not found; trying CMake-driven build instead.
    )
  )

  if "%BUILT_WITH_TOOL%"=="0" (
    where cmake >nul 2>&1
    if errorlevel 1 (
      echo Error: 'cmake' not found in PATH. Please install CMake or add it to PATH.
      exit /b 1
    )

    set "CMAKE_BUILD_CMD=cmake --build . --config !BUILD_CONFIG! --target install"
    if exist "build.ninja" (
      set "CMAKE_BUILD_CMD=!CMAKE_BUILD_CMD! -- -j !JOBS!"
    ) else (
      if exist "*.sln" (
        set "CMAKE_BUILD_CMD=!CMAKE_BUILD_CMD! -- /m:!JOBS!"
      )
    )

    echo -- !CMAKE_BUILD_CMD!
    !CMAKE_BUILD_CMD! || exit /b 1
  )

  popd
)

if "%EDITABLE%"=="1" (
  if "%WITH_DEPS%"=="1" (
    python -m pip install -e . || exit /b 1
    call :ensure_reaktoro_conda || exit /b 1
  ) else (
    python -m pip install --no-deps -e . || exit /b 1
  )
  exit /b 0
)

python -m build --wheel || exit /b 1

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
  call :ensure_reaktoro_conda || exit /b 1
) else (
  python -m pip install --no-deps --force-reinstall "dist\%WHEEL%" || exit /b 1
)

goto :eof

:ensure_reaktoro_conda
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('reaktoro') else 1)" >nul 2>&1
if %errorlevel%==0 (
  exit /b 0
)

where conda >nul 2>&1
if errorlevel 1 (
  echo Warning: 'conda' command not found; please install Reaktoro manually via 'conda install -c conda-forge reaktoro'.
  exit /b 0
)

set "conda_prefix=%CONDA_PREFIX%"
if not defined conda_prefix (
  echo Warning: CONDA_PREFIX is empty; activate the target Conda environment before running install_darts.bat to auto-install Reaktoro.
  exit /b 0
)

REM Check Python version compatibility (Reaktoro on conda-forge requires Python >=3.10, <3.13)
for /f %%v in ('python -c "import sys; print(sys.version_info.minor)"') do set "py_minor=%%v"
if !py_minor! LSS 10 goto :reaktoro_version_error
if !py_minor! GEQ 13 goto :reaktoro_version_error
goto :reaktoro_install

:reaktoro_version_error
for /f %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "py_version=%%v"
echo Warning: Reaktoro on conda-forge requires Python ^>=3.10 and ^<3.13, but the current environment has Python !py_version!.
echo.
echo To install Reaktoro, create a compatible conda environment (e.g., Python 3.11):
echo   conda create -n darts-rkt python=3.11 -y
echo   conda activate darts-rkt
echo.
echo Then re-run this script with --with-deps flag.
exit /b 0

:reaktoro_install
set "REAKTORO_LOG=%cd%\make_reaktoro.log"
>> "%REAKTORO_LOG%" (
  echo + conda install -y -c conda-forge -p "!conda_prefix!" reaktoro
)
call conda install -y -c conda-forge -p "!conda_prefix!" reaktoro >> "%REAKTORO_LOG%" 2>&1 || exit /b 1
echo -- Install Reaktoro: DONE!
exit /b 0

:__validate_num
REM Returns ERRORLEVEL 0 if argument is all digits, else 1
setlocal
set "VAL=%~1"
if "%VAL%"=="" ( endlocal & exit /b 1 )
for /f "delims=0123456789" %%i in ("%VAL%") do ( endlocal & exit /b 1 )
endlocal & exit /b 0
