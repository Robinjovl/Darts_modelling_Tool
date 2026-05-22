@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "TARGET_SCRIPT=%SCRIPT_DIR%sync_agent_skills.py"

if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" "%TARGET_SCRIPT%" --check
    exit /b %ERRORLEVEL%
)

if exist "%LOCALAPPDATA%\anaconda3\envs\darts\python.exe" (
    "%LOCALAPPDATA%\anaconda3\envs\darts\python.exe" "%TARGET_SCRIPT%" --check
    exit /b %ERRORLEVEL%
)

if exist "%LOCALAPPDATA%\anaconda3\envs\DARTS\python.exe" (
    "%LOCALAPPDATA%\anaconda3\envs\DARTS\python.exe" "%TARGET_SCRIPT%" --check
    exit /b %ERRORLEVEL%
)

if exist "%LOCALAPPDATA%\anaconda3\python.exe" (
    "%LOCALAPPDATA%\anaconda3\python.exe" "%TARGET_SCRIPT%" --check
    exit /b %ERRORLEVEL%
)

for /f "delims=" %%I in ('where python.exe 2^>nul') do (
    echo %%I | findstr /i /c:"WindowsApps" >nul
    if errorlevel 1 (
        "%%I" "%TARGET_SCRIPT%" --check
        exit /b !ERRORLEVEL!
    )
)

echo Unable to find a usable Python interpreter for sync_agent_skills.py 1>&2
exit /b 1
