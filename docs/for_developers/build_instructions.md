# DARTS build instructions 

## Windows (multithread)

1. Generate git keys pair and upload the public key to GitLab
2. Clone the DARTS project from GitLab
3. Install Python 3.6 or higher. Set the `PYTHONPATH` environment variable. You should restart VS to take them updated in it.
Add path to Python to your environment variable `PATH` (for example `C:\WPy64-39100\python-3.9.10.amd64`)
4. Open the Visual Studio 2022 solution darts-engines.sln in the directory darts-engines
5. Select the configuration you want (Release_MT for multithreaded Release) and build the solution. 
10. To install DARTS modules to your Python distribution: run `WinPython-64bit-3.6.0.1Qt5\WinPython Command Prompt.exe`
then run `darts\darts-package\build_install_darts.bat` in this console.

## Windows (multithread+GPU)

1. Install CUDA Toolkit and get AMGX. 
You can get AMGX sources as a submodule of darts-linear-solvers, then you will found it in `darts-linear-solvers\lib\AMGX`
If you want to build AMGX yourself: run cmake to configure and create VS projects.
Example: 
`cmake -DCUDA_ARCH="70" -DCMAKE_BUILD_TYPE=Release -A x64  -B "./AMGX/build"`  
If needed add `-DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin/nvcc.exe"`
Ready to use amgxsh.dll can be downloaded via the [link](https://surfdrive.surf.nl/files/index.php/s/OMqRRqB9oeeN0cr)
Copy amgxsh.dll and all Cuda DLL files from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin` to 
the python package directory, for example `WPy64-39100\python-3.9.10.amd64\Lib\site-packages\darts`
2. Do the items 1-4 from the Windows (multithread) instruction
3. Use _gpu.sln and `Release_MT_gpu` configuration for linear_solvers and darts-engines projects
4. Do the items 6-10 from Windows (multithread) instruction

## Linux (multithread)

1. Install python3, python3-dev, and pip
2. Install numpy (`pip install numpy`)
3. run `build_darts.sh`

## Linux (multithread+GPU)

1. Download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
Errors may fixed by manually installing liburcu6  
<code>
wget http://ftp.de.debian.org/debian/pool/main/libu/liburcu/liburcu6_0.12.2-1_amd64.deb   
sudo dpkg -i liburcu6_0.12.2-1_amd64.deb  
</code> 
2. Download and build AMGX  
<code>
git clone https://github.com/NVIDIA/AMGX.git  
cmake -DCMAKE_BUILD_TYPE=Release -B "./build"  
cd build && make -j 4
</code>  
The '4' is number of threads used for parallel compilation.
In case you use gcc 11.2 you probably will get a compilation error in std_function.h, 
use this [solution](https://github.com/NVIDIA/nccl/issues/102#issuecomment-1021420403)
3. Do the items from the Linux (multithread) instruction with run `build_darts.sh 0 gpu` in third step.
In case you want to compile only engine, run `make gpu USE_OPENDARTS_LINEAR_SOLVERS=0` in engines folder.

