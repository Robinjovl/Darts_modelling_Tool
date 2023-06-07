rem update submodules
git submodule update --recursive --remote --init || goto :error
echo "build_darts arg:" "%1"
if "%1"=="0" (
  cd darts-engines
  .\update_private_artifacts.bat $env:SMBNAME $env:SMBLOGIN $env:SMBPASS
  cd ..
  rem 2 Compile engines with iterative linear solver
  set configengine=ReleaseMT
  set config=Release
) else (
  rem 1 build opendarts linear solvers step
  cd opendarts_linear_solvers
  rem 1a build superLU
  cd thirdparty\SuperLU_5.2.1
  msbuild superlu.sln /p:Configuration=Release /p:Platform=x64 -maxCpuCount:8 || goto :error

  rem 1b build opendarts linear solvers
  cd ..\..
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=..\..\darts-engines\lib\darts_linear_solvers -DENABLE_TESTING=TRUE ..
  msbuild opendarts_linear_solvers.sln /p:Configuration=Release /p:Platform=x64 -maxCpuCount:8 || goto :error
  msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:8 || goto :error
  cd ..\..
  
  set configengine=Release_ODLS
  set config=Release_ODLS
)

rem 2 Compile engines with open-darts linear solver
cd darts-engines
msbuild darts-engines.vcxproj /p:Configuration=%configengine% /p:Platform=x64 -maxCpuCount:8  || goto :error
cd ..

rem 3 Compile discretizer
cd darts-discretizer
msbuild darts-discretizer.vcxproj /p:Configuration=%config% /p:Platform=x64 -maxCpuCount:8  || goto :error
cd ..

rem 4 Build wheel
echo 'build darts.whl for windows started'
rem copy VS redist libraries 
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\msvcp140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\vcruntime140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.OpenMP\vcomp140.dll .\darts
python darts-package/darts/print_build_info.py
python setup.py build bdist_wheel --plat-name=win-amd64 || goto :error

rem || goto :error checks exit code of command 
rem if one of the commands fails, interrupt batch and return error code
:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%
