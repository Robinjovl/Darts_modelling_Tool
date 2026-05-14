@echo off
setlocal enabledelayedexpansion

REM Check if thirdparty folder exists --------------------------------
if not exist "thirdparty" (
    REM Display error in red using PowerShell
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Write-Host 'Error: thirdparty folder not found!' -ForegroundColor Red; Write-Host 'Please contact your administrator to download the thirdparty files.' -ForegroundColor Red"
    pause
    exit 1
)

REM ========================================================================
REM   Build script for open-darts with MGR Linear Solver support
REM   Based on build_darts_cmake.bat
REM
REM   Features:
REM   - Automatically detects Visual Studio and CMake installations
REM   - Uses multiple mirror sites for downloading thirdparty libraries
REM   - Provides clear error messages and setup instructions
REM ========================================================================

REM Repository URLs and versions
set "HYPRE_REPO=https://github.com/hypre-space/hypre.git"
set "PYBIND11_REPO=https://github.com/pybind/pybind11.git"
set "PYBIND11_VERSION=v2.13.0"
set "PYBIND11_VERSION_NOV=2.13.0"
set "IPHREEQC_REPO=https://github.com/usgs-coupled/iphreeqc.git"
set "MSHIO_REPO=https://github.com/qnzhou/MshIO.git"
set "MSHIO_BRANCH=main"

REM Read input arguments ---------------------------------------------
set clean_mode=false
set skip_req=false
set skip_submodule=false
set skip_thirdparty_check=true
set config=Release
set NT=64
set rebuild_hypre=false
set use_mirror=true
set toolset=

:parse_args
if "%~1"=="" goto :process_input
set option=%~1
shift
if "%option%"=="-h" goto :help_info
if "%option%"=="-c" set clean_mode=true & goto parse_args
if "%option%"=="-r" set skip_req=true & goto parse_args
if "%option%"=="--skip-submodule" set skip_submodule=true & goto parse_args
if "%option%"=="--skip-thirdparty-check" set skip_thirdparty_check=true & goto parse_args
if "%option%"=="--check-thirdparty" set skip_thirdparty_check=false & goto parse_args
if /i "%option%"=="--skip-thirdparty-check=false" set skip_thirdparty_check=false & goto parse_args
if "%option%"=="--no-mirror" set use_mirror=false & goto parse_args
if "%option%"=="-d" set config=%~1 & shift & goto parse_args
if "%option%"=="-j" set NT=%~1 & shift & goto parse_args
if "%option%"=="--rebuild-hypre" set rebuild_hypre=true & goto parse_args
if "%option%"=="--toolset" set toolset=%~1 & shift & goto parse_args
echo Error: Unknown option '%option%'
goto :error

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
echo   config         = %config%
echo   threads        = %NT%
echo   clean_mode     = %clean_mode%
echo   skip_req       = %skip_req%
echo   skip_submodule = %skip_submodule%
echo   skip_thirdparty_check = %skip_thirdparty_check%
echo   rebuild_hypre  = %rebuild_hypre%
echo   use_mirror     = %use_mirror%
echo.
echo Working directory: %CD%
echo.
echo OPEN_DARTS_ROOT: %OPEN_DARTS_ROOT%
echo.
REM ========================================================================
REM   Detect and setup build tools
REM ========================================================================
echo ========================================================================
echo   Detecting build tools
echo ========================================================================
echo.
REM Detect CMake
echo - Detecting CMake...
set "CMAKE_CMD="
set "CMAKE_FOUND=0"

where cmake.exe >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%i in ('where cmake.exe') do (
        set "CMAKE_CMD=%%i"
        set "CMAKE_FOUND=1"
        echo   [OK] Found CMake in PATH: %%i
        goto :cmake_found
    )
)

REM Search in common locations on multiple drives
for %%D in (C:, D:, E:, F: G:) do (
    for %%P in (
        "%%D\Program Files\CMake\bin\cmake.exe"
        "%%D\Program Files (x86)\CMake\bin\cmake.exe"
        "%%D\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        "%%D\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        "%%D\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        "%%D\Program Files\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        "%%D\Program Files\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        "%%D\Program Files\Microsoft Visual Studio\2019\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        "%%D\CMake\bin\cmake.exe"
        "%%D\Tools\CMake\bin\cmake.exe"
    ) do (
        if exist "%%P" (
            set "CMAKE_CMD=%%~fP"
            set "CMAKE_FOUND=1"
            echo   [OK] Found CMake at: %%~fP
            goto :cmake_found
        )
    )
)

REM Check user profile directory
for %%P in (
    "%USERPROFILE%\AppData\Local\Programs\CMake\bin\cmake.exe"
    "%USERPROFILE%\AppData\Roaming\CMake\bin\cmake.exe"
    "%USERPROFILE%\CMake\bin\cmake.exe"
) do (
    if exist "%%P" (
        set "CMAKE_CMD=%%~fP"
        set "CMAKE_FOUND=1"
        echo   [OK] Found CMake at: %%~fP
        goto :cmake_found
    )
)

REM Check system environment variables for CMake path
if not "!CMAKE_FOUND!"=="1" (
    if defined CMAKE_HOME (
        if exist "!CMAKE_HOME!\bin\cmake.exe" (
            set "CMAKE_CMD=!CMAKE_HOME!\bin\cmake.exe"
            set "CMAKE_FOUND=1"
            echo   [OK] Found CMake from CMAKE_HOME: !CMAKE_HOME!\bin\cmake.exe
            goto :cmake_found
        )
    )
    if defined CMAKE_ROOT (
        if exist "!CMAKE_ROOT!\bin\cmake.exe" (
            set "CMAKE_CMD=!CMAKE_ROOT!\bin\cmake.exe"
            set "CMAKE_FOUND=1"
            echo   [OK] Found CMake from CMAKE_ROOT: !CMAKE_ROOT!\bin\cmake.exe
            goto :cmake_found
        )
    )
)

REM Search all drives for cmake.exe (comprehensive search)
if not "!CMAKE_FOUND!"=="1" (
    for %%D in (C D E F G) do (
        if exist %%D:\ (
            dir /s /b "%%D:\cmake.exe" > "%TEMP%\cmake_search.txt" 2>nul
            set /p CMAKE_PATH=<%TEMP%\cmake_search.txt
            if not "!CMAKE_PATH!"=="" (
                set "CMAKE_CMD=!CMAKE_PATH!"
                set "CMAKE_FOUND=1"
                echo   [OK] Found CMake at: !CMAKE_PATH!
                goto :cmake_found
            )
        )
    )
)

:cmake_found
if !CMAKE_FOUND! equ 0 (
    echo   [ERROR] CMake not found!
    echo.
    echo Please install CMake from one of:
    echo   - https://cmake.org/download/
    echo   - Or install Visual Studio with CMake support
    echo.
    echo After installation, either:
    echo   1. Add CMake to your system PATH, or
    echo   2. Run this script from Developer Command Prompt for VS
    goto :error
)

REM Detect Visual Studio and MSBuild
echo.
echo - Detecting Visual Studio and MSBuild...
set "MSBUILD_CMD="
set "VS_VERSION="
set "VS_FOUND=0"

REM Check if msbuild is already in PATH
where msbuild.exe >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2 delims==" %%i in ('where msbuild.exe') do (
        set "MSBUILD_CMD=%%i"
        set "VS_FOUND=1"
        echo   [OK] Found MSBuild in PATH: %%i
        goto :vs_found
    )
)

REM Search in common VS locations on multiple drives
for %%D in (C:, D:, E:, F: G:) do (
    for %%P in (
        "%%D\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\18\Professional\MSBuild\Current\Bin\MSBuild.exe"
        "%%D\Program Files\Microsoft Visual Studio\18\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
    ) do (
        if exist "%%P" (
            set "MSBUILD_CMD=%%P"
            set "VS_FOUND=1"
            echo   [OK] Found MSBuild at: %%P
            goto :vs_found
        )
    )
)

REM Search all drives for msbuild.exe (comprehensive search)
if not "!VS_FOUND!"=="1" (
    for %%D in (C D E F G) do (
        if exist %%D:\ (
            dir /s /b "%%D:\msbuild.exe" > "%TEMP%\msbuild_search.txt" 2>nul
            set /p MSBUILD_PATH=<%TEMP%\msbuild_search.txt
            if not "!MSBUILD_PATH!"=="" (
                set "MSBUILD_CMD=!MSBUILD_PATH!"
                set "VS_FOUND=1"
                echo   [OK] Found MSBuild at: !MSBUILD_PATH!
                goto :vs_found
            )
        )
    )
)

:vs_found
if !VS_FOUND! equ 0 (
    echo   [ERROR] Visual Studio with MSBuild not found!
    echo.
    echo Please install Visual Studio 2019/2022 with C++ build tools from:
    echo   - https://visualstudio.microsoft.com/downloads/
    echo.
    echo Required components:
    echo   - Desktop development with C++
    echo   - CMake tools for Visual Studio
    echo.
    echo After installation, run this script from Developer Command Prompt for VS
    goto :error
)

REM Determine CMake generator
REM Default to Visual Studio 17 2022 which works for most cases
REM For VS 2026 users: if v143 toolset error occurs, script will use v145 toolset
set "CMAKE_GENERATOR=Visual Studio 17 2022"
set "CMAKE_TOOLSET="
set "HAS_SLN=true"

REM Check if using Visual Studio 2026 and set appropriate toolset
REM Use the MSBuild path to detect VS version
if not "!toolset!"=="" (
    echo   Using user-specified toolset: !toolset!
    set "CMAKE_TOOLSET=!toolset!"
) else (
    REM Detect Visual Studio version
    echo !MSBUILD_CMD! | findstr /C:"\\18\\" >nul
    if !errorlevel! equ 0 (
        REM For Visual Studio 2026, use v145 toolset
        set "CMAKE_TOOLSET=v145"
        echo   Detected Visual Studio 2026
        echo   Using default toolset: v145
    ) else (
        REM For other Visual Studio versions, use v143 toolset
        set "CMAKE_TOOLSET=v143"
        echo   Using default toolset: v143
    )
)

REM Always detect Visual Studio version to set appropriate generator
echo !MSBUILD_CMD! | findstr /C:"\\18\\" >nul
if !errorlevel! equ 0 (
    set "CMAKE_GENERATOR=Visual Studio 18 2026"
    REM Set VisualStudioVersion environment variable to force VS 2026
    set "VisualStudioVersion=18.0"
    echo   Using Visual Studio 18 2026 generator
    echo   Set VisualStudioVersion=!VisualStudioVersion!
)

echo.
echo Build tools detected:
echo   CMake   : !CMAKE_CMD!
echo   MSBuild : !MSBUILD_CMD!
echo   Generator: !CMAKE_GENERATOR!
if not "!CMAKE_TOOLSET!"=="" echo   Toolset  : !CMAKE_TOOLSET!
echo.
echo ========================================================================
echo   Starting build process
echo ========================================================================
echo.
REM Clean up old build artifacts (ignore errors)
echo Cleaning up old artifacts...
echo Deep cleaning build directories
echo Cleanup completed
echo.
REM ========================================================================
REM   Initialize git submodules with mirror support
REM ========================================================================
if "!skip_thirdparty_check!"=="false" (
    if "!skip_submodule!"=="false" (
        echo ========================================================================
        echo   Initializing thirdparty libraries
        echo ========================================================================
        echo.
        
        echo Updating git submodules...
        git submodule update --init --recursive > make_submodules.log 2>&1
        if errorlevel 1 (
            echo   [WARNING] Failed to update submodules, but continuing with existing libraries...
            echo   Please check make_submodules.log for details.
        ) else (
            REM Keep the HYPRE checkout already present in thirdparty\hypre.
            echo Checking HYPRE source checkout...
            if exist "thirdparty\hypre\.git" (
                pushd thirdparty\hypre
                set "HYPRE_SOURCE_LABEL="
                for /f "delims=" %%i in ('git describe --tags --always 2^>nul') do set "HYPRE_SOURCE_LABEL=%%i"
                if defined HYPRE_SOURCE_LABEL (
                    echo   [OK] Using HYPRE checkout !HYPRE_SOURCE_LABEL!
                ) else (
                    echo   [OK] Using current HYPRE checkout in thirdparty\hypre
                )
                popd
            ) else (
                echo   [OK] Using vendored HYPRE sources from thirdparty\hypre
            )
        )
        
        echo   [OK] All thirdparty libraries initialized
        echo.
    )
) else (
    echo ========================================================================
    echo   Skipping thirdparty library check
    echo ========================================================================
    echo   Assuming all thirdparty libraries are already initialized
    echo.
)

if /I "!skip_req: =!"=="true" goto :after_thirdparty_build

if /I "!skip_req: =!"=="false" (
  echo ========================================================================
  echo   Step 1/3: Building HYPRE with MGR support
  echo ========================================================================
  echo.
  if "!rebuild_hypre!"=="true" (
      echo Cleaning HYPRE build...
      rmdir /s /q thirdparty\hypre\src\cmbuild 2> NUL
      rmdir /s /q thirdparty\install 2> NUL
  )

  REM Check if HYPRE source exists
  set "HYPRE_SRC_FOUND=0"
  if exist "thirdparty\hypre\src\configure" (
      set "HYPRE_SRC_FOUND=1"
      echo   Found HYPRE configure script
  )
  if exist "thirdparty\hypre\src\CMakeLists.txt" (
      set "HYPRE_SRC_FOUND=1"
      echo   Found HYPRE CMakeLists.txt in src
  )
  if exist "thirdparty\hypre\CMakeLists.txt" (
      set "HYPRE_SRC_FOUND=1"
      echo   Found HYPRE CMakeLists.txt in root
  )
  echo   HYPRE_SRC_FOUND = !HYPRE_SRC_FOUND!

  if "!HYPRE_SRC_FOUND!"=="0" goto :hypre_not_found
  goto :hypre_found

:hypre_not_found
  echo   [ERROR] HYPRE source not found!
  echo   Please ensure thirdparty libraries are initialized.
  echo   Try running: build_darts_with_mgr.bat (without --skip-submodule)
  echo   Or manually download HYPRE from: https://github.com/hypre-space/hypre/releases
  goto :error

:hypre_found

  pushd thirdparty

  echo - Install HYPRE with MGR support: START

  REM Check if HYPRE has src subdirectory
  if exist "hypre\src\CMakeLists.txt" (
      cd hypre\src
      set "HYPRE_CMAKE_SOURCE=%CD%"
      if not exist cmbuild mkdir cmbuild
      call :ensure_cmake_cache_matches "%HYPRE_CMAKE_SOURCE%\cmbuild" "%HYPRE_CMAKE_SOURCE%" "HYPRE"
      if not exist cmbuild mkdir cmbuild
      cd cmbuild
      set "HYPRE_BUILD_LIB=lib\Release\HYPRE.lib"
      set "HYPRE_INSTALL_LIB=..\..\..\install\lib\HYPRE.lib"
      "!CMAKE_CMD!" -D HYPRE_ENABLE_TIMING=OFF ^
            -D HYPRE_BUILD_TESTS=OFF ^
            -D HYPRE_BUILD_EXAMPLES=OFF ^
            -D HYPRE_ENABLE_MPI=OFF ^
            -D CMAKE_INSTALL_PREFIX=..\..\..\install .. > ..\..\..\..\make_hypre.log || goto :error
  ) else if exist "hypre\CMakeLists.txt" (
      cd hypre
      set "HYPRE_CMAKE_SOURCE=%CD%"
      if not exist cmbuild mkdir cmbuild
      call :ensure_cmake_cache_matches "%HYPRE_CMAKE_SOURCE%\cmbuild" "%HYPRE_CMAKE_SOURCE%" "HYPRE"
      if not exist cmbuild mkdir cmbuild
      cd cmbuild
      set "HYPRE_BUILD_LIB=lib\Release\HYPRE.lib"
      set "HYPRE_INSTALL_LIB=..\..\install\lib\HYPRE.lib"
      "!CMAKE_CMD!" -D HYPRE_ENABLE_TIMING=OFF ^
            -D HYPRE_BUILD_TESTS=OFF ^
            -D HYPRE_BUILD_EXAMPLES=OFF ^
            -D HYPRE_ENABLE_MPI=OFF ^
            -D CMAKE_INSTALL_PREFIX=..\..\install .. > ..\..\..\make_hypre.log || goto :error
  ) else (
      echo   [ERROR] Cannot find HYPRE CMakeLists.txt
      goto :error
  )

  !MSBUILD_CMD! INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:%NT% >> ..\..\..\..\make_hypre.log || goto :error
  
  REM Ensure HYPRE.lib is available from the install location used by DARTS.
  if exist "!HYPRE_INSTALL_LIB!" (
      echo   [OK] HYPRE.lib installed to !HYPRE_INSTALL_LIB!
  ) else (
      if exist "!HYPRE_BUILD_LIB!" (
          for %%I in ("!HYPRE_INSTALL_LIB!") do if not exist "%%~dpI" mkdir "%%~dpI"
          copy "!HYPRE_BUILD_LIB!" "!HYPRE_INSTALL_LIB!" >nul 2>&1
          if errorlevel 1 (
              echo   [ERROR] Failed to copy HYPRE.lib to install directory
              goto :error
          ) else (
              echo   [OK] HYPRE.lib copied to install directory
          )
      ) else (
          echo   [ERROR] HYPRE.lib not found after build
          goto :error
      )
  )
  
  popd
  echo - Install HYPRE with MGR support: DONE!

  echo -- Install SuperLU
  REM Check if SuperLU directory exists
  if exist "thirdparty\SuperLU_5.2.1" (
      cd thirdparty\SuperLU_5.2.1
  ) else if exist "SuperLU_5.2.1" (
      cd SuperLU_5.2.1
  ) else (
      echo   [ERROR] SuperLU directory not found!
      goto :error
  )
  REM Determine platform toolset based on Visual Studio version
  if not "!toolset!"=="" (
      set "PLATFORM_TOOLSET=!toolset!"
      echo   Using user-specified toolset: !PLATFORM_TOOLSET!
  ) else (
      REM Detect Visual Studio version
      echo !MSBUILD_CMD! | findstr /C:"\\18\\" >nul
      if !errorlevel! equ 0 (
          REM For Visual Studio 2026, use v145 toolset
          set "PLATFORM_TOOLSET=v145"
          echo   Using default toolset: v145
      ) else (
          REM For other Visual Studio versions, use v143 toolset
          set "PLATFORM_TOOLSET=v143"
          echo   Using default toolset: !PLATFORM_TOOLSET!
      )
  )
  
  REM Force toolset override by modifying the vcxproj files
  echo   Modifying SuperLU vcxproj files to use !PLATFORM_TOOLSET! toolset...
  
  REM Modify SuperLU vcxproj files to use the correct toolset
  for /r . %%f in (*.vcxproj) do (
      powershell -NoProfile -Command "$path='%%f'; $content=[System.IO.File]::ReadAllText($path); $updated=$content -replace 'v143', '!PLATFORM_TOOLSET!' -replace 'v145', '!PLATFORM_TOOLSET!'; if ($updated -ne $content) { [System.IO.File]::WriteAllText($path, $updated) }"
  )
  
  REM Build SuperLU with the correct toolset
  !MSBUILD_CMD! superlu.sln /p:Configuration=%config% /p:Platform=x64 /p:PlatformToolset=!PLATFORM_TOOLSET! -maxCpuCount:%NT% > ..\..\make_superlu.log || goto :error
  cd ..
)

:after_thirdparty_build

echo ========================================================================
echo   Step 2/3: Building openDARTS with MGR integration
echo ========================================================================
echo.
REM Ensure we're in the correct directory
cd /d "%OPEN_DARTS_ROOT%"

REM Clean up build artifacts only for an explicit clean build.
if "!clean_mode!"=="true" (
    echo Cleaning openDARTS build directories...
    rmdir /s /q build 2> NUL
    rmdir /s /q solvers\solver_mgr\build 2> NUL
) else (
    echo Reusing openDARTS build directories for incremental build...
)

REM Create build directory if needed
call :ensure_cmake_cache_matches "%OPEN_DARTS_ROOT%\build" "%OPEN_DARTS_ROOT%" "openDARTS"
if not exist build mkdir build
cd build

REM Setup build with CMake
REM Note: We're already in the build directory, so use relative paths
set cmake_options=-D CMAKE_INSTALL_PREFIX=..\darts -D CMAKE_BUILD_TYPE=%config%
echo CMake options: %cmake_options%

REM Add toolset parameter if specified
if defined CMAKE_TOOLSET (
    echo   Using CMake generator: !CMAKE_GENERATOR! with toolset !CMAKE_TOOLSET!
    REM Set CMAKE_GENERATOR_TOOLSET environment variable to force toolset
    set CMAKE_GENERATOR_TOOLSET=!CMAKE_TOOLSET!
    "!CMAKE_CMD!" -G "!CMAKE_GENERATOR!" -A x64 -T !CMAKE_TOOLSET! -D CMAKE_GENERATOR_TOOLSET=!CMAKE_TOOLSET! %cmake_options% .. > ..\make_darts.log || goto :error
    
    REM Modify generated vcxproj files to use the correct toolset
    echo   Modifying generated vcxproj files to use !CMAKE_TOOLSET! toolset...
    for /r . %%f in (*.vcxproj) do (
        powershell -NoProfile -Command "$path='%%f'; $content=[System.IO.File]::ReadAllText($path); $updated=$content -replace 'v143', '!CMAKE_TOOLSET!' -replace 'v145', '!CMAKE_TOOLSET!'; if ($updated -ne $content) { [System.IO.File]::WriteAllText($path, $updated) }"
    )
    
    REM Also modify VCTargetsPath.vcxproj if it exists
    if exist "CMakeFiles\*\VCTargetsPath.vcxproj" (
        for /r "CMakeFiles" %%f in (VCTargetsPath.vcxproj) do (
            powershell -NoProfile -Command "$path='%%f'; $content=[System.IO.File]::ReadAllText($path); $updated=$content -replace 'v143', '!CMAKE_TOOLSET!' -replace 'v145', '!CMAKE_TOOLSET!'; if ($updated -ne $content) { [System.IO.File]::WriteAllText($path, $updated) }"
        )
    )
) else (
    echo   Using CMake generator: !CMAKE_GENERATOR!
    "!CMAKE_CMD!" -G "!CMAKE_GENERATOR!" -A x64 %cmake_options% .. > ..\make_darts.log || goto :error
)

REM build and install
echo Building openDARTS...
REM Check if .slnx file exists (VS 2026+), otherwise use .sln
if exist openDARTS.slnx (
    !MSBUILD_CMD! openDARTS.slnx /p:Configuration=%config% /p:Platform=x64 /p:PlatformToolset=!CMAKE_TOOLSET! -maxCpuCount:%NT% >> ..\make_darts.log || goto :error
) else (
    !MSBUILD_CMD! openDARTS.sln /p:Configuration=%config% /p:Platform=x64 /p:PlatformToolset=!CMAKE_TOOLSET! -maxCpuCount:%NT% >> ..\make_darts.log || goto :error
)
!MSBUILD_CMD! INSTALL.vcxproj /p:Configuration=%config% /p:Platform=x64 /p:PlatformToolset=!CMAKE_TOOLSET! -maxCpuCount:%NT% >> ..\make_darts.log || goto :error

cd ..

echo ========================================================================
echo   Step 3/3: Building openDARTS with MGR integration: DONE!
echo ========================================================================

echo ========================================================================
echo   Step 4/3: Verifying MGR integration
echo ========================================================================
echo.
REM Detect Python version for .pyd file check
set "PY_VER="
set "PY_SHORT=unknown"
set "PY_TAG=cp311"
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
if defined PY_VER (
    for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
        set "PY_SHORT=%%a.%%b"
        set "PY_TAG=cp%%a%%b"
    )
)
echo Detected Python version: %PY_SHORT%

echo Checking build artifacts:
set "ENGINES_FILE=darts\engines.%PY_TAG%-win_amd64.pyd"
if exist "%ENGINES_FILE%" (
    echo   [OK] %ENGINES_FILE%
) else (
    echo   [WARNING] %ENGINES_FILE% not found!
    echo   This may be due to Python version mismatch. Looking for alternatives...

    REM Try to find any engines.pyd file
    dir /b "darts\engines.*.pyd" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] Found alternative: darts\engines.*.pyd
    ) else (
        echo   [ERROR] No engines.pyd found!
        goto :error
    )
)

set "DISCRETIZER_FILE=darts\discretizer.%PY_TAG%-win_amd64.pyd"
if exist "%DISCRETIZER_FILE%" (
    echo   [OK] %DISCRETIZER_FILE%
) else (
    echo   [WARNING] %DISCRETIZER_FILE% not found!
    dir /b "darts\discretizer.*.pyd" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] Found alternative: darts\discretizer.*.pyd
    ) else (
        echo   [ERROR] No discretizer.pyd found!
        goto :error
    )
)

REM MGR solver library check removed as requested
REM if exist "build\solvers\solver_mgr\Release\mgr-linear-solver.lib" (
REM     echo   [OK] MGR solver library
REM ) else (
REM     echo   [ERROR] MGR solver library not found!
REM     goto :error
REM )

REM MGR integration library check removed as requested
REM if exist "build\solvers\solver_mgr\Release\linsolv_mgr.lib" (
REM     echo   [OK] MGR integration library
REM ) else (
REM     echo   [ERROR] MGR integration library not found!
REM     goto :error
REM )

REM Linear solvers library check removed as requested
REM if exist "build\solvers\linear_solvers\src\Release\linear_solvers.lib" (
REM     echo   [OK] Linear solvers library
REM ) else (
REM     echo   [ERROR] Linear solvers library not found!
REM     goto :error
REM )

REM MGR solver integration verification
if exist "darts\engines.*.pyd" (
    echo   [OK] MGR solver integration verified
) else (
    echo   [ERROR] MGR solver integration failed!
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
echo   [OK] MGR solver integration (linsolv_mgr)
echo   [OK] open-darts engines and discretizer
echo.
echo Output locations:
echo   - HYPRE library:  thirdparty\install\lib\HYPRE.lib
echo   - SuperLU library: thirdparty\SuperLU_5.2.1\Release\superlu.lib
echo   - open-darts:      darts\*.pyd
echo.
echo Usage in open-darts:
echo   The MGR solver is now available as linsolv_mgr.
echo   To use it in your models, set:
echo     n.params.linear_type = n.params.linear_solver_t.cpu_gmres_mgr
echo.
if exist "darts\print_build_info.py" (
    python darts\print_build_info.py
)

goto :eof

REM ========================================================================
REM   Function: download_library
REM   Parameters:
REM     %1 = library name
REM     %2 = repository URL
REM     %3 = version tag
REM     %4 = version without 'v' prefix
REM     %5 = verification file (empty for any file)
REM ========================================================================
:download_library
set "LIB_NAME=%~1"
set "LIB_REPO=%~2"
set "LIB_VERSION=%~3"
set "LIB_VERSION_NOV=%~4"
set "LIB_VERIFY=%~5"

echo Checking %LIB_NAME%...

REM Check if library already exists and is complete
if exist "%LIB_NAME%" (
    REM Check if directory is not empty and has git repository or source files
    if exist "%LIB_NAME%\.git" (
        echo   [OK] %LIB_NAME% already exists (git repo)
        exit /b 0
    )
    if exist "%LIB_NAME%\CMakeLists.txt" (
        echo   [OK] %LIB_NAME% already exists
        exit /b 0
    )
    if exist "%LIB_NAME%\setup.py" (
        echo   [OK] %LIB_NAME% already exists
        exit /b 0
    )
    REM Directory exists but appears incomplete, will redownload
    echo   Directory exists but incomplete, re-downloading...
)

echo   Downloading %LIB_NAME% %LIB_VERSION%...

REM Remove existing incomplete directory
if exist "%LIB_NAME%" (
    rmdir /s /q "%LIB_NAME%" 2> NUL
)

REM Try direct GitHub first with timeout
echo   Trying direct GitHub...
REM Use timeout mechanism by checking if the command completes within 10 seconds
powershell -Command "$job = Start-Job { git clone --depth 1 --branch '%LIB_VERSION%' '%LIB_REPO%' '%LIB_NAME%' > '../make_%LIB_NAME%.log' 2>&1 }; if (Wait-Job $job -Timeout 10) { Receive-Job $job } else { Stop-Job $job; Write-Output 'TIMEOUT' }" > nul 2>&1
if !errorlevel! equ 0 (
    if exist "%LIB_NAME%\.git" (
        echo   [OK] %LIB_NAME% downloaded from GitHub
        exit /b 0
    )
)

echo   Direct GitHub failed or timed out, trying mirrors...

REM Try mirrors with timeout
if "!use_mirror!"=="true" (
    for %%M in (
        "https://ghproxy.com"
        "https://mirror.ghproxy.com"
        "https://gitclone.com"
    ) do (
        echo   Trying mirror: %%M
        powershell -Command "$job = Start-Job { git clone --depth 1 --branch '%LIB_VERSION%' '%%M/%LIB_REPO%' '%LIB_NAME%' >> '../make_%LIB_NAME%.log' 2>&1 }; if (Wait-Job $job -Timeout 10) { Receive-Job $job } else { Stop-Job $job; Write-Output 'TIMEOUT' }" > nul 2>&1
        if !errorlevel! equ 0 (
            if exist "%LIB_NAME%\.git" (
                echo   [OK] %LIB_NAME% downloaded from mirror %%M
                exit /b 0
            )
        )
    )
)

echo   [ERROR] Failed to download %LIB_NAME%!
echo   Please download manually from GitHub and extract to thirdparty\%LIB_NAME%
echo   Or check your network connection/proxy settings
exit /b 1

:error
echo.
echo ========================================================================
echo   BUILD FAILED
echo ========================================================================
echo.
echo Error occurred during build. Check the log files:
echo   - make_hypre.log    (HYPRE build log)
echo   - make_superlu.log  (SuperLU build log)
echo   - make_darts.log    (openDARTS build log)
echo.
echo Common issues:
echo   1. Missing build tools: Install Visual Studio with C++ and CMake
echo   2. Missing submodules: Run 'git submodule update --init --recursive'
echo   3. Network issues: Use --no-mirror if mirror causes problems
echo   4. Wrong directory: Run from open-darts root directory
echo.
exit /b %errorlevel%

REM Helper: remove a CMake build directory if its cache belongs to another repo path.
:ensure_cmake_cache_matches
setlocal
set "CACHE_BUILD_DIR=%~f1"
set "CACHE_SOURCE_DIR=%~f2"
set "CACHE_LABEL=%~3"

if not exist "%CACHE_BUILD_DIR%\CMakeCache.txt" (
    endlocal & exit /b 0
)

set "CACHE_STATUS="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$cache='%CACHE_BUILD_DIR%\CMakeCache.txt'; $expectedBuild=[IO.Path]::GetFullPath('%CACHE_BUILD_DIR%').Replace('\','/').TrimEnd('/'); $expectedSource=[IO.Path]::GetFullPath('%CACHE_SOURCE_DIR%').Replace('\','/').TrimEnd('/'); $content=Get-Content $cache -Raw; $actualBuild=$null; $actualSource=$null; if($content -match '(?m)^CMAKE_CACHEFILE_DIR:INTERNAL=(.+)$'){ $actualBuild=[IO.Path]::GetFullPath($matches[1]).Replace('\','/').TrimEnd('/') }; if($content -match '(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$'){ $actualSource=[IO.Path]::GetFullPath($matches[1]).Replace('\','/').TrimEnd('/') }; if(($actualBuild -and $actualBuild -ne $expectedBuild) -or ($actualSource -and $actualSource -ne $expectedSource)){ 'mismatch' } else { 'match' }"`) do set "CACHE_STATUS=%%i"

if /i "!CACHE_STATUS!"=="mismatch" (
    echo   [INFO] %CACHE_LABEL% CMake cache belongs to another directory. Cleaning "%CACHE_BUILD_DIR%"...
    rmdir /s /q "%CACHE_BUILD_DIR%" 2> NUL
)

endlocal & exit /b 0

REM Help info --------------------------------------------------------
:help_info
echo helper_scripts\build_darts_with_mgr.bat [OPTIONS]
echo.
echo   Build open-darts with MGR Linear Solver support
echo.
echo USAGE:
echo   build_darts_with_mgr.bat [-h] [-c] [-r] [-d CONFIG] [-j N] [--rebuild-hypre] [--skip-submodule] [--check-thirdparty] [--no-mirror] [--toolset TOOLSET]
echo.
echo OPTIONS:
echo   -h               : Display this help message
echo   -c               : Clean openDARTS build directories before configuring
echo   -r               : Skip building thirdparty libraries (HYPRE, SuperLU)
echo   -d CONFIG        : Build configuration [Release, Debug]. Default: Release
echo   -j N             : Set number of threads for compilation. Default: 64
echo   --rebuild-hypre  : Delete HYPRE build/install outputs and rebuild HYPRE
echo   --skip-submodule : Skip git submodule initialization
echo   --skip-thirdparty-check : Skip thirdparty library check (default)
echo   --check-thirdparty : Run thirdparty library check/submodule init
echo   --no-mirror      : Disable mirror configuration for git submodules
echo   --toolset TOOLSET: Specify platform toolset (e.g., v143, v145). Default: auto-detect
echo.
echo EXAMPLES:
echo   # Standard build (incremental)
echo   build_darts_with_mgr.bat
echo.
echo   # Fastest local incremental rebuild when thirdparty libs are unchanged
echo   build_darts_with_mgr.bat -r
echo.
echo   # Clean openDARTS rebuild while keeping thirdparty outputs
echo   build_darts_with_mgr.bat -c -r
echo.
echo   # Rebuild HYPRE and then rebuild openDARTS
echo   build_darts_with_mgr.bat -c --rebuild-hypre
echo.
echo   # Build with Debug configuration
echo   build_darts_with_mgr.bat -d Debug
echo.
echo   # Run thirdparty check/submodule init, then build everything
echo   build_darts_with_mgr.bat --check-thirdparty
echo.
echo REQUIREMENTS:
echo   - Visual Studio 2019/2022 with C++ build tools
echo   - CMake 3.20 or higher
echo   - Python 3.11 (for pyd files)
echo   - Git (for submodule initialization)
echo.
echo SETUP INSTRUCTIONS:
echo   1. Install Visual Studio 2019/2022 with "Desktop development with C++"
echo   2. Install CMake from https://cmake.org/download/
echo   3. Open "Developer Command Prompt for VS" or run from regular CMD
echo      (script will auto-detect VS installation)
echo.
goto :eof
REM ----------------------------------------------------------------
