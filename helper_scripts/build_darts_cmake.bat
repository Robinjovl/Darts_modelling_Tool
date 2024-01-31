set config=Release

set bos_solvers_option=""
if "%1"=="-a" (
  cd engines
  call .\update_private_artifacts.bat %SMBNAME% %SMBLOGIN% %SMBPASS%
  cd ..
  set bos_solvers_option=-D BOS_SOLVERS_DIR=%cd%\engines\lib\darts_linear_solvers -DOPENDARTS_CONFIG=MT
)

if "%1"=="-b" (
  set bos_solvers_option=-D BOS_SOLVERS_DIR=%2 -DOPENDARTS_CONFIG=MT
)

if "%1"=="-d" (
  set config=Debug
)

echo "bos_solvers_option=" %bos_solvers_option%

rem - Update submodules: START
rmdir /s /q thirdparty\eigen thirdparty\pybind11
git submodule update --recursive --remote --init || goto :error
rem - Update submodules: DONE!

rem - Install requirements: START
rem -- Install Eigen 3
cd thirdparty
mkdir build
cd build
mkdir eigen
cd eigen
cmake -D CMAKE_INSTALL_PREFIX=../../install ../../eigen/ || goto :error
msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:1 || goto :error
cd ..\..

rem -- Install SuperLU
cd SuperLU_5.2.1
msbuild superlu.sln /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:8 || goto :error
cd ..\..
rem - Install requirements: DONE !  

rem ========================================================================
rem   Building openDARTS: START 
rem ========================================================================

rmdir build
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=..\darts -DENABLE_TESTING=OFF %bos_solvers_option% ..
msbuild openDARTS.sln /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:8 || goto :error
msbuild INSTALL.vcxproj /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:8 || goto :error

cd ..
rem ========================================================================
rem   Building openDARTS: DONE! 
rem ========================================================================

rem ************************************************************************
rem   Building python package open-darts: START 
rem ************************************************************************

echo 'build darts.whl for windows started'
rem copy VS redist libraries 
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\msvcp140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\vcruntime140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.OpenMP\vcomp140.dll .\darts
python darts/print_build_info.py
python setup.py build bdist_wheel --plat-name=win-amd64 || goto :error
rem -- Python wheel generated!

python -m pip install .[cpg]

rem ************************************************************************
rem   Building python package open-darts: DONE! 
rem ************************************************************************

rem || goto :error checks exit code of command 
rem if one of the commands fails, interrupt batch and return error code
:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%
