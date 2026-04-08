# How to compile and run DARTS in the container locally
# If you want to leave the container running but detach from it, press Ctrl+P then Ctrl+Q.

# run bare MS container for testing purposes
#docker run -it mcr.microsoft.com/windows/servercore:ltsc2019 cmd

# build a docker image from a Dockerfile
docker build -t vs-buildtools .

# run the docker image, allocate 16G RAM and run command line in the docker
docker run -it -m 16g vs-buildtools cmd

# create and activate a conda environment
conda create -n darts python=3.10 --yes
conda init
# need to restart the shell, with the compiler environment
cmd /k "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" "&" cmd
conda activate darts

# download, compile and install open-DARTS 
git clone --branch development --recursive https://gitlab.com/open-darts/open-darts.git
cd open-darts
helper_scripts\build_darts_cmake.bat -w -p -j 10
helper_scripts\install_darts.bat

# run a test
cd models\cpg_sloping_fault
python main.py 
