# Import the MSVC compiler environment into the current PowerShell session.
#
# WHAT IT DOES
#   Runs the Visual Studio environment batch file (vsdevcmd.bat, or
#   vcvars64.bat as a fallback) and copies every variable it produces into
#   this PowerShell session: PATH, INCLUDE, LIB, LIBPATH, VCToolsRedistDir
#   and friends.
#
# WHY IT IS NEEDED
#   Installing Build Tools does not by itself make the compiler usable.
#   cl.exe is not on PATH until the VS environment batch file has run, and
#   a batch file can only set variables inside the cmd.exe process that
#   executes it -- a PowerShell session cannot inherit them.
#
#   The builder image works around this with an ENTRYPOINT that runs
#   vcvars64.bat first (see .cicd/build-windows/Dockerfile). That covers an
#   interactive "docker run", but NOT CI: the GitLab docker-windows executor
#   replaces the ENTRYPOINT with its own shell, so a CI job starts with no
#   compiler environment at all. Two things then fail in build-windows:
#     1. CMake cannot find a C++ compiler, and
#     2. $env:VCToolsRedistDir is empty, so the commands that copy
#        msvcp140.dll and the other redist DLLs into the wheel are handed a
#        malformed path.
#   Calling this script first is what prevents both.
#
# HOW IT WORKS
#   cmd /c "<batch> && set" runs the batch file and then prints the resulting
#   environment as NAME=VALUE lines. Those lines are parsed and written back
#   into the session. This is the standard way to move a batch-file
#   environment into PowerShell.
#
# USAGE
#   .\helper_scripts\build_vars.ps1
#
#   Looks in C:\BuildTools by default -- where the CI image installs Build
#   Tools. Set VS_BUILDTOOLS_PATH to point somewhere else, e.g. a local
#   install:
#     $env:VS_BUILDTOOLS_PATH = 'C:\Program Files\Microsoft Visual Studio\2022\Community'
#
#   Throws if no environment batch file is found, rather than continuing
#   without a compiler and failing later somewhere less obvious.

$ErrorActionPreference = 'Stop'
$env:PreferredToolArchitecture = 'x64'

# C:\BuildTools is where .cicd/build-windows/Dockerfile installs Build Tools.
$installationPath = if ($env:VS_BUILDTOOLS_PATH) { $env:VS_BUILDTOOLS_PATH } else { 'C:\BuildTools' }

$candidates = @(
    (Join-Path $installationPath 'Common7\Tools\vsdevcmd.bat'),
    (Join-Path $installationPath 'VC\Auxiliary\Build\vcvars64.bat')
)

$batch = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $batch) {
    throw "No MSVC environment script found under '$installationPath'. Checked: $($candidates -join ', '). Set VS_BUILDTOOLS_PATH to override."
}

$batchArgs = if ($batch -like '*vsdevcmd.bat') { '-arch=x64 -host_arch=x64 -no_logo' } else { '' }

Write-Host "Loading MSVC environment from: $batch $batchArgs"

# "&& set" dumps the environment the batch file produced; copy it back here.
& "${env:COMSPEC}" /s /c "`"$batch`" $batchArgs && set" | ForEach-Object {
    $name, $value = $_ -split '=', 2
    if ($name -and $null -ne $value) {
        Set-Content env:\"$name" $value
    }
}

# Sanity check: if this is unset the environment did not load, and the build
# would otherwise fail much later with a far less obvious error.
if (-not $env:VCToolsRedistDir) {
    throw 'MSVC environment was not loaded: VCToolsRedistDir is not set.'
}
