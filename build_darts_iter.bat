rem update submodules
git submodule update --recursive --remote --init || goto :error


rem 2 Compile engines
cd darts-engines
msbuild darts-engines.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:2  || goto :error
cd ..

rem 3 Compile discretizer
cd darts-discretizer
msbuild darts-discretizer.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:2  || goto :error
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
