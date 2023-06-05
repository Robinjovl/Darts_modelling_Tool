rem update submodules
git submodule update --recursive --remote --init || goto :error

rem 1 build opendarts linear solvers step
cd opendarts_linear_solvers
rem 1a build superLU
cd thirdparty\SuperLU_5.2.1
msbuild superlu.sln /p:Configuration=Release /p:Platform=x64 -maxCpuCount:2 || goto :error

rem 1b build opendarts linear solvers
cd ..\..
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DENABLE_TESTING=TRUE ..
msbuild opendarts_linear_solvers.sln /p:Configuration=Release /p:Platform=x64 -maxCpuCount:2 || goto :error
msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:1 || goto :error
move install ..\..\darts-engines\lib\darts_linear_solvers || goto :error
cd ..\..

rem 2 Compile engines
cd darts-engines
msbuild darts-engines.vcxproj /p:Configuration=Release_ODLS /p:Platform=x64 -maxCpuCount:2  || goto :error
cd ..

rem 3 Compile discretizer
cd darts-discretizer
msbuild darts-discretizer.vcxproj /p:Configuration=Release_ODLS /p:Platform=x64 -maxCpuCount:2  || goto :error
cd ..

rem 4 Build wheel
echo 'build darts.whl for windows started'
rem copy VS redist libraries 
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\msvcp140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\vcruntime140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.OpenMP\vcomp140.dll .\darts

python darts/print_build_info.py
python setup.py build bdist_wheel --plat-name=win-amd64 || goto :error

rem || goto :error checks exit code of command 
rem if one of the commands fails, interrupt batch and return error code
:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%

