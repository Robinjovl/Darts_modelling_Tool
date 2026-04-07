# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2019

SHELL ["cmd", "/S", "/C"]

# Download and install Build Tools
RUN curl -SL --output vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe `
    && start /w vs_buildtools.exe --quiet --wait --norestart --nocache `
       --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" `
       --add Microsoft.VisualStudio.Workload.VCTools `
       --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
       --add Microsoft.VisualStudio.Component.Windows10SDK.19041 `
    || IF "%ERRORLEVEL%"=="3010" EXIT 0 `
    && del /q vs_buildtools.exe

# Set up the developer command prompt environment
ENTRYPOINT ["C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat", "&&"]
CMD ["cmd"]
