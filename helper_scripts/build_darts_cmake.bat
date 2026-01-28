@echo off
setlocal enabledelayedexpansion

REM Read input arguments ---------------------------------------------
set clean_mode=false
set testing=false
set wheel=false
set bos_solvers_artifact=false
set bos_solvers_dir=""
set iter_solvers=false
set MT=true
set GPU=%false
set skip_req=false
set config=Release
set NT=8
set skip_req=false
set phreeqc=false

:parse_args
if "%~1"=="" goto :process_input
set option=%1
shift
if "%option%"=="-h" goto :help_info
if "%option%"=="-c" set clean_mode=true & goto parse_args
if "%option%"=="-t" set testing=true & goto parse_args
if "%option%"=="-w" set wheel=true & goto parse_args
if "%option%"=="-m" set MT=true & goto parse_args
if "%option%"=="-G" set GPU=true & goto parse_args
if "%option%"=="-r" set skip_req=true & goto parse_args
if "%option%"=="-d" set config=%1 & shift & goto parse_args
if "%option%"=="-j" set NT=%1 & shift & goto parse_args
if "%option%"=="-a" set bos_solvers_artifact=true & set iter_solvers=true & goto parse_args
if "%option%"=="-b" set bos_solvers_dir=%1 & set iter_solvers=true & shift & goto parse_args
if "%option%"=="-p" set phreeqc=true & goto parse_args
goto parse_args

:process_input
if %bos_solvers_artifact%==true (
  cd engines
  call .\update_private_artifacts.bat %SMBNAME% %SMBLOGIN% %SMBPASS%
  cd ..
  set bos_solvers_dir="%cd%\engines\lib\darts_linear_solvers"
  if %testing%==true (
    set testing=false
  )
)
REM ODLS version does not support OpenMP yet
if %iter_solvers%==false (
  if %GPU%==true (
    echo Error: GPU build requires GPU bos solvers. Specify the path with -b.
    exit 1
  )
  if %MT%==true (
    echo Warning: ODLS version does not support OpenMP yet. Switched to the sequentional build.
    set MT=false
  )
)

echo - Report configuration of this script: START
echo    bos_solvers_dir = %bos_solvers_dir%
echo    fetch bos_solvers_artifact = %bos_solvers_artifact%
echo    config = %config%
echo    gpu = %GPU%
echo    testing = %testing%
echo    generate python wheel = %wheel%
echo    Multi thread = %MT%
echo    Phreeqc support = %phreeqc%
echo - Report configuration of this script: DONE!
REM ----------------------------------------------------------------

del darts\*.pyd 2> NUL
rmdir /s /q dist 2> NUL

if %clean_mode%==true (
  echo - Cleaning up
  rmdir /s /q build 2> NUL
  rmdir /s /q "thirdparty/build" 2> NUL
  REM goto :eof
)

if %skip_req%==false (
  echo - Update submodules: START
  rmdir /s /q thirdparty\eigen thirdparty\pybind11 thirdparty\MshIO thirdparty\hypre
  git submodule sync --recursive
  git submodule update --init --recursive -- ^
             thirdparty\pybind11 ^
             thirdparty\MshIO ^
             thirdparty\hypre || goto :error
  if %phreeqc%==true (
    git submodule update --init --recursive thirdparty\iphreeqc || goto :error
  )
  echo - Update submodules: DONE!

  cd thirdparty

  echo - Install requirements: START
  mkdir build

  rem -- Install Hypre
  cd hypre\src\cmbuild
  rem For debugging: -DHYPRE_ENABLE_PRINT
  cmake -D HYPRE_BUILD_TESTS=ON ^
        -D HYPRE_BUILD_EXAMPLES=ON ^
        -D HYPRE_WITH_MPI=OFF ^
        -D CMAKE_INSTALL_PREFIX=..\..\..\install .. > ..\..\..\..\make_hypre.log || goto :error
  msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:8 >> ..\..\..\..\make_hypre.log || goto :error
  cd ..\..\..\

  echo -- Install SuperLU
  cd SuperLU_5.2.1
  msbuild superlu.sln /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:%NT% > ..\..\make_superlu.log || goto :error
  cd ..\..

  if %phreeqc%==true (
    echo -- Install IPhreeqc: START
    cd thirdparty\build
    if not exist iphreeqc mkdir iphreeqc
    cd iphreeqc
	  cmake ^
      -D CMAKE_INSTALL_PREFIX=..\..\install\iphreeqc ^
      -D BUILD_TESTING=OFF ^
      -D BUILD_SHARED_LIBS=ON ^
      ..\..\iphreeqc > ..\..\..\make_iphreeqc.log 2>&1
    msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:8 >> ..\..\..\make_iphreeqc.log || goto :error
    cd ..\..\..
  )

  echo - Install requirements: DONE!
)

echo ========================================================================
echo   Building openDARTS: START
echo ========================================================================

rmdir /s /q build 2> NUL
mkdir build
cd build

REM Setup build with CMake
set cmake_options=-D CMAKE_INSTALL_PREFIX=..\darts -D CMAKE_BUILD_TYPE=%config%
if %testing%==true (
  set cmake_options=%cmake_options% -D ENABLE_TESTING=ON
)
if %MT%==true (
  set cmake_options=%cmake_options% -D OPENDARTS_CONFIG=MT
)
if %GPU%==true (
  set cmake_options=%cmake_options% -D OPENDARTS_CONFIG=GPU
)
if %phreeqc%==true (
  set cmake_options=%cmake_options% -D WITH_PHREEQC=ON
  echo Phreeqc support: ENABLED
) else (
  echo Phreeqc support: DISABLED
)
if not %bos_solvers_dir%=="" (
  set cmake_options=%cmake_options% -D BOS_SOLVERS_DIR=%bos_solvers_dir%
)

echo CMake options: %cmake_options%
cmake %cmake_options% ..

REM build and install
msbuild openDARTS.sln /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:%NT% > ..\make_darts.log || goto :error
msbuild INSTALL.vcxproj /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:%NT% > ..\make_darts.log || goto :error

if %testing%==true ctest -C %config%  || goto :error

cd ..
echo ========================================================================
echo   Building openDARTS: DONE!
echo ========================================================================

echo ************************************************************************
echo   Building python package open-darts: START
echo ************************************************************************

python darts\print_build_info.py
if %wheel%==true (
  echo -- build darts.whl for windows started
  copy CHANGELOG.md darts
  rem copy VS redist libraries
  rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\msvcp140.dll .\darts
  rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\vcruntime140.dll .\darts
  rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.OpenMP\vcomp140.dll .\darts
  python -m pip install --upgrade build > make_wheel.log || goto :error
  python -m build --wheel >> make_wheel.log || goto :error
  echo -- Python wheel generated!
)
python -m pip install . >> make_wheel.log

if %phreeqc%==true (
  call :ensure_reaktoro_conda || goto :error
)

echo ************************************************************************
echo   Building python package open-darts: DONE!
echo ************************************************************************

rem || goto :error checks exit code of command
rem if one of the commands fails, interrupt batch and return error code
:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%
goto :eof

REM Help info --------------------------------------------------------
:help_info
echo helper_scripts\build_darts_cmake.bat [-h] [-c] [-t] [-w] [-m] [-r] [-a] [-b BOS_SOLVER_DIRECTORY] [-d INSTALL CONFIGURATION] [-j NUM THREADS]
echo    Script to install opendarts on Windows.
echo USAGE:
echo    -h : displays this help menu.
echo    -c : cleans up build to prepare a new fresh build. Default: don't clean
echo    -t : Enable testing: ctest of solvers. Default: don't test
echo    -w : Enable generation of python wheel. Default: false
echo    -m : Enable Multi-thread MT (with OMP) build. Warning: Solvers is not MT. Default: true
echo    -r : Skip building thirdparty libraries (if you have them already compiled). Default: false
echo    -a : Update private artifacts bos_solvers (instead of openDARTS solvers). This is meant to be used by CI/CD. Default: false
echo    -b SPATH  : Path to bos_solvers (instead of openDARTS solvers), example: -b ./darts-linear-solvers containing lib/libdarts_linear_solvers.a (already compiled).
echo    -d MODE   : Configuration for C++ code [Release, Debug]. Example: -d Debug
echo    -j N      : Set number of threads (N) for compilation. Default: 8. Example: -j 4
echo    -p : Enable Phreeqc + Reaktoro (requires Conda). Default: false
goto :eof
REM ----------------------------------------------------------------

:ensure_reaktoro_conda
REM Use a local copy of CONDA_PREFIX and quoted comparisons to avoid parser
REM errors when the prefix contains spaces or parentheses (observed as
REM "<token> was unexpected at this time" failures in CI).
set "conda_prefix=%CONDA_PREFIX%"

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('reaktoro') else 1)" >NUL 2>&1
if %errorlevel%==0 (
  echo -- Reaktoro already available in current Python interpreter.
  exit /b 0
)

where conda >NUL 2>&1
if errorlevel 1 (
  echo Error: 'conda' command not found. Install Conda and activate an environment before using -p.
  exit /b 1
)

if not defined conda_prefix (
  echo Error: CONDA_PREFIX is empty. Activate the target Conda environment before using -p.
  exit /b 1
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
echo To install Reaktoro, create a compatible conda environment (e.g., Python 3.12):
echo   conda create -n darts-rkt python=3.12 -y
echo   conda activate darts-rkt
echo.
echo Then re-run this script with the -p flag.
exit /b 0

:reaktoro_install
set "REAKTORO_LOG=%cd%\make_reaktoro.log"
echo -- Install Reaktoro via conda (prefix "!conda_prefix!"). Full log: %REAKTORO_LOG%
>> "%REAKTORO_LOG%" (
  echo + conda install -y -c conda-forge -p "!conda_prefix!" reaktoro
)
call conda install -y -c conda-forge -p "!conda_prefix!" reaktoro >> "%REAKTORO_LOG%" 2>&1 || exit /b 1
echo -- Install Reaktoro: DONE!
exit /b 0
