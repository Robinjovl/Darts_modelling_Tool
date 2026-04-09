@echo off
setlocal

set MSBUILD="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
set LOGFILE=%~dp0make_darts_hys.log

echo Building openDARTS (incremental Debug)... > %LOGFILE%
echo Build started: %date% %time% >> %LOGFILE%

%MSBUILD% build\openDARTS.sln /p:Configuration=Debug /p:Platform=x64 /maxCpuCount:8 >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo MAIN BUILD FAILED with errorlevel %errorlevel%
    exit /b %errorlevel%
)
echo Main build: OK >> %LOGFILE%

%MSBUILD% build\INSTALL.vcxproj /p:Configuration=Debug /p:Platform=x64 /maxCpuCount:8 >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo INSTALL FAILED with errorlevel %errorlevel%
    exit /b %errorlevel%
)
echo Install: OK >> %LOGFILE%
echo Build finished: %date% %time% >> %LOGFILE%
echo BUILD COMPLETE
exit /b 0
