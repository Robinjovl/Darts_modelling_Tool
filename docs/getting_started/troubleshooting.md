# Troubleshooting

### Compilation issues

#### cmake

`A subdirectory or file build already exists. The current CMakeCache.txt directory ... is different than the directory ... where CMakeCache.txt was created. CMake Error: The source "..." does not match the source "..." used to generate cache. Re-run cmake with a different source directory.`

Solution: run "helper_scripts\\build_darts_cmake" with -r argument.

### Wheels generation 

1. If open-darts wheels you generated in the open-darts/dist folder contains `UNKNOWN.0.0.0` in its name, please upgrade setuptools:

   `pip install --upgrade setuptools`

### Runtime issues

1. `cannot import darts.engines`

If one opens the root open-DARTS folder from the gitlab repository in PyCharm, Python will try to import darts from that local folder (which has no engines library if it DARTS not compiled) instead of importing it from the installation path. As a workaround, the `models` folder can be opened as a project in PyCharm.

2. Cannot import darts.engines:

```
ImportError: DLL load failed while importing engines: The specified module could not be found.
```

Solution: check engines.pyd (engines.so) has been compiled with the same Python version as you use. Check all DLL files are in the PATH (Windows) or in LD_LIBRARY_PATH (Linux).

### Installation issues

For installing open-DARTS using `helper_scripts\build_install_darts.bat`, Python version 3.9 is required. Using a different Python version may result in errors, displaying the following message:

`ERROR: open_darts-1.2.2-cp39-cp39-win_amd64.whl is not a supported wheel on this platform.`

#### GPU related issues

- CUDA installation

Errors may fixed by manually installing liburcu6

```bash
wget http://ftp.de.debian.org/debian/pool/main/libu/liburcu/liburcu6_0.12.2-1_amd64.deb   
sudo dpkg -i liburcu6_0.12.2-1_amd64.deb  
```

- AMGX compilation

In case you use gcc 11.2 you probably will get a compilation error in std_function.h, use this [solution](https://github.com/NVIDIA/nccl/issues/102#issuecomment-1021420403)
