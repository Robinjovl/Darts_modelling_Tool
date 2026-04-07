# Running the container locally
# If you want to leave the container running but detach from it, press Ctrl+P then Ctrl+Q.

# run MS container
#docker run -it mcr.microsoft.com/windows/servercore:ltsc2019 cmd

# build a docker image from a Dockerfile
docker build -t vs-buildtools .

# run the docker image
docker run -it -m 16g vs-buildtools powershell

# create and activate a conda environment
$env:CONDA_ENV = "darts"
$env:PY_VER = "3.10"
conda create -n $env:CONDA_ENV python=$env:PY_VER --yes
# init conda and environment, need to restart the shell
conda init
cmd /k "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" "&" powershell
conda activate $env:CONDA_ENV

# download and compile open-DARTS 
git clone --branch development --recursive https://gitlab.com/open-darts/open-darts.git
cd open-darts
cmd /c helper_scripts\build_darts_cmake.bat -w -p -j 10
