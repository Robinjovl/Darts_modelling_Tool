# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2019

SHELL ["cmd", "/S", "/C"]

# Step 1: Install Visual Studio Build Tools with C++ workload
RUN curl -SL --output vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe `
    && start /w vs_buildtools.exe --quiet --wait --norestart --nocache `
       --installPath "C:\BuildTools" `
       --add Microsoft.VisualStudio.Workload.MSBuildTools `
       --add Microsoft.VisualStudio.Workload.VCTools `
       --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core `
       --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
       --add Microsoft.VisualStudio.Component.VC.140 `
       --add Microsoft.VisualStudio.Component.VC.v141.x86.x64 `
       --add Microsoft.VisualStudio.Component.Windows10SDK.19041 `
       --add Microsoft.VisualStudio.Component.Windows10SDK `
       --add Microsoft.Component.MSBuild `
       --includeRecommended `
    && del /q vs_buildtools.exe

# Step 2: Install Windows SDK (ensures rc.exe and mt.exe are available)
RUN curl.exe -SL --output winsdksetup.exe "https://go.microsoft.com/fwlink/?linkid=2272610" `
    && start /w winsdksetup.exe /features OptionId.DesktopCPPx64 /quiet /norestart `
    && del /q winsdksetup.exe

# Step 3: Install CMake 3.29.3
RUN curl -SL --output cmake.zip https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-windows-x86_64.zip `
    && powershell -Command "Expand-Archive -Path cmake.zip -DestinationPath C:\cmake" `
    && del /q cmake.zip

# Step 4: Install Miniforge (conda-forge)
RUN curl -SL --output miniforge.exe https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe `
    && start /w miniforge.exe /S /D=C:\conda `
    && del /q miniforge.exe

# Step 5: Install Git
RUN curl -SL --output git-installer.exe https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/Git-2.45.2-64-bit.exe `
    && start /w git-installer.exe /VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /DIR="C:\Git" `
    && del /q git-installer.exe

# Step 6: Add Windows SDK tools (rc.exe, mt.exe) to PATH
RUN powershell -Command `
    "$rcPath = (Get-ChildItem 'C:\Program Files (x86)\Windows Kits' -Recurse -Filter 'rc.exe' | Where-Object { $_.FullName -match 'x64' } | Select-Object -First 1).DirectoryName; `
    $oldPath = [Environment]::GetEnvironmentVariable('PATH','Machine'); `
    [Environment]::SetEnvironmentVariable('PATH', \"$rcPath;$oldPath\", 'Machine'); `
    Write-Host \"Added SDK path: $rcPath\""

# Step 7: Set up environment variables
RUN setx /M PATH "C:\cmake\cmake-3.29.3-windows-x86_64\bin;%PATH%"
RUN setx /M PATH "C:\Git\cmd;%PATH%"
RUN setx /M PATH "C:\conda;C:\conda\Scripts;%PATH%"

# Set up the MSVC developer environment on entry
ENTRYPOINT ["cmd", "/k", "C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat", "&&"]
CMD ["cmd"]
