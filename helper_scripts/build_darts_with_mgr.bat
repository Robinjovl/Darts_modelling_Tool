@echo off
setlocal enabledelayedexpansion

REM ========================================================================
REM   Build script for open-darts with MGR Linear Solver support
REM   Based on build_darts_cmake.bat
REM ========================================================================

REM Read input arguments ---------------------------------------------
set clean_mode=false
set skip_req=false
set config=Release
set NT=8
set rebuild_hypre=false

:parse_args
if "%~1"=="" goto :process_input
set option=%~1
shift
if "%option%"=="-h" goto :help_info
if "%option%"=="-c" set clean_mode=true & goto parse_args
if "%option%"=="-r" set skip_req=true & goto parse_args
if "%option%"=="-d" set config=%~1 & shift & goto parse_args
if "%option%"=="-j" set NT=%~1 & shift & goto parse_args
if "%option%"=="--rebuild-hypre" set rebuild_hypre=true & goto parse_args
goto parse_args

:process_input
REM Since we're in helper_scripts, parent is open-darts root
REM Use a different approach: get current directory and assume we're called from root
set CURRENT_DIR=%CD%

REM Check if we're already in the right place
if not exist "%CURRENT_DIR%\CMakeLists.txt" (
    echo ERROR: CMakeLists.txt not found in current directory!
    echo Current directory: %CURRENT_DIR%
    echo Please run this script from the open-darts root directory.
    goto :error
)

set OPEN_DARTS_ROOT=%CURRENT_DIR%

echo ========================================================================
echo   Building open-darts with MGR Linear Solver
echo ========================================================================
echo.
echo Configuration:
echo   config        = %config%
echo   threads       = %NT%
echo   clean_mode    = %clean_mode%
echo   skip_req      = %skip_req%
echo.
echo Working directory: %CD%
echo.
echo OPEN_DARTS_ROOT: %OPEN_DARTS_ROOT%
echo.

del darts\*.pyd 2> NUL
rmdir /s /q dist 2> NUL

if %clean_mode%==true (
  echo - Cleaning up
  rmdir /s /q build 2> NUL
)

if %skip_req%==false (
  echo ========================================================================
  echo   Step 0/3: Building HYPRE 2.29.0 with MGR support
  echo ========================================================================
  echo.

  if %rebuild_hypre%==true (
      echo Cleaning HYPRE build...
      rmdir /s /q thirdparty\hypre\src\cmbuild 2> NUL
      rmdir /s /q thirdparty\install 2> NUL
  )

  pushd thirdparty

  echo - Install HYPRE with MGR support: START
  cd hypre\src\cmbuild
  cmake -D HYPRE_ENABLE_TIMING=OFF ^
        -D HYPRE_ENABLE_TESTS=OFF ^
        -D HYPRE_BUILD_EXAMPLES=OFF ^
        -D HYPRE_WITH_MPI=OFF ^
        -D HYPRE_WITH_MGR=ON ^
        -D CMAKE_INSTALL_PREFIX=..\..\..\install .. > ..\..\..\..\make_hypre.log || goto :error
  msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:%NT% >> ..\..\..\..\make_hypre.log || goto :error
  cd ..\..\..\
  echo - Install HYPRE with MGR support: DONE!

  echo -- Install SuperLU
  cd SuperLU_5.2.1
  msbuild superlu.sln /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:%NT% > ..\..\make_superlu.log || goto :error
  cd ..\..

  popd
)

echo ========================================================================
echo   Step 1/3: Building openDARTS with MGR integration
echo ========================================================================
echo.

REM Ensure we're in the correct directory
cd /d "%OPEN_DARTS_ROOT%"

rmdir /s /q build 2> NUL
mkdir build

REM Setup build with CMake (using absolute path)
set cmake_options=-D CMAKE_INSTALL_PREFIX=%OPEN_DARTS_ROOT%\darts -D CMAKE_BUILD_TYPE=%config%
echo CMake options: %cmake_options%
cmake -G "Visual Studio 17 2022" -A x64 %cmake_options% -S %OPEN_DARTS_ROOT% -B %OPEN_DARTS_ROOT%\build

REM build and install
msbuild build\openDARTS.sln /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:%NT% > make_darts.log || goto :error
msbuild build\INSTALL.vcxproj /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:%NT% >> make_darts.log || goto :error

echo ========================================================================
echo   Step 2/3: Building openDARTS with MGR integration: DONE!
echo ========================================================================

echo ========================================================================
echo   Step 3/3: Verifying MGR integration
echo ========================================================================
echo.

echo Checking build artifacts:
if exist "darts\engines.cp311-win_amd64.pyd" (
    echo   [OK] darts\engines.cp311-win_amd64.pyd
) else (
    echo   [ERROR] darts\engines.cp311-win_amd64.pyd not found!
    goto :error
)

if exist "darts\discretizer.cp311-win_amd64.pyd" (
    echo   [OK] darts\discretizer.cp311-win_amd64.pyd
) else (
    echo   [ERROR] darts\discretizer.cp311-win_amd64.pyd not found!
    goto :error
)

if exist "build\solvers\solver_mgr\Release\mgr-linear-solver.lib" (
    echo   [OK] MGR solver library
) else (
    echo   [ERROR] MGR solver library not found!
    goto :error
)

if exist "build\solvers\solver_mgr\Release\linsolv_mgr.lib" (
    echo   [OK] MGR integration library
) else (
    echo   [ERROR] MGR integration library not found!
    goto :error
)

if exist "build\solvers\linear_solvers\src\Release\linear_solvers.lib" (
    echo   [OK] Linear solvers library
) else (
    echo   [ERROR] Linear solvers library not found!
    goto :error
)

echo.
echo ========================================================================
echo   Build Summary
echo ========================================================================
echo.
echo Build completed successfully!
echo.
echo Components built:
echo   [OK] HYPRE library (with MGR support)
echo   [OK] SuperLU library
echo   [OK] MGR Linear Solver library
echo   [OK] MGR Integration (linsolv_mgr)
echo   [OK] open-darts engines and discretizer
echo.
echo Output locations:
echo   - HYPRE library:  thirdparty\install\lib\HYPRE.lib
echo   - SuperLU library: thirdparty\SuperLU_5.2.1\Release\superlu.lib
echo   - MGR library:    build\solvers\solver_mgr\Release\mgr-linear-solver.lib
echo   - Integration:    build\solvers\solver_mgr\Release\linsolv_mgr.lib
echo   - open-darts:      darts\*.pyd
echo.
echo Usage in open-darts:
echo   The MGR solver is now available as linsolv_mgr.
echo   To use it in your models, set:
echo     n.params.linear_type = n.params.linear_solver_t.cpu_gmres_mgr
echo.

python darts\print_build_info.py

goto :eof

:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%
goto :eof

REM Help info --------------------------------------------------------
:help_info
echo helper_scripts\build_darts_with_mgr.bat [OPTIONS]
echo.
echo   Build open-darts with MGR Linear Solver support
echo.
echo USAGE:
echo   build_darts_with_mgr.bat [-h] [-c] [-r] [-d CONFIG] [-j N] [--rebuild-hypre]
echo.
echo OPTIONS:
echo   -h               : Display this help message
echo   -c               : Clean build (delete all build artifacts and rebuild)
echo   -r               : Skip building thirdparty libraries (HYPRE, SuperLU)
echo   -d CONFIG        : Build configuration [Release, Debug]. Default: Release
echo   -j N             : Set number of threads for compilation. Default: 8
echo   --rebuild-hypre  : Force rebuild of HYPRE library with MGR support
echo.
echo EXAMPLES:
echo   # Standard build (incremental)
echo   build_darts_with_mgr.bat
echo.
echo   # Clean rebuild everything including HYPRE
echo   build_darts_with_mgr.bat -c --rebuild-hypre
echo.
echo   # Build with Debug configuration
echo   build_darts_with_mgr.bat -d Debug
echo.
goto :eof
REM ----------------------------------------------------------------
