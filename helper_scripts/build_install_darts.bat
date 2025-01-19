@echo off

:: Check Python architecture and set the platform name accordingly
for /f "delims=" %%a in ('python -c "import platform; print(platform.architecture()[0])"') do set ARCH=%%a

if "%ARCH%"=="64bit" (
    set PLAT_NAME=win_amd64
) else (
    set PLAT_NAME=win32
)

:: Clean and build the package
python setup.py clean
python setup.py build bdist_wheel --plat-name %PLAT_NAME%

:: Install the built wheel
for %%f in (dist\open_darts-*.whl) do (
    echo Installing %%f...
    python -m pip install --upgrade --no-deps --force-reinstall "%%f"
)