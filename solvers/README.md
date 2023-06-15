# Opendarts Linear Solvers
Simple block CSR matrix storage implementation

## Contents description
The folder `linear_solvers` contains the open source implementation of block csr
matrix and linear solvers. Currently, data storage and direct solver using Super LU
are the only functionalities, i.e., linear algebra
such as matrix-matrix multiplication of matrix-vector multiplication is **NOT** implemented.

The folder `tests`, contains the tests that also show how to use the library `open-linear-solvers`.

## Quick build instructions 
If you wish to use `opendarts-linear-solvers` in `opendarts`, the easiest way to build and install `opendarts-linear-solvers` is to follow the instructions below. As a pre-requisite you need to have the `gcc` compiler installed (on macOS you need version 11 installed with homebrew and available as `gcc-11`). Additionally you need to have `cmake` available.

Download `opendarts` into folder `opendarts`. Move inside the folder `opendarts`. Inside this folder download `opendarts-linear-solvers` as a subfolder. You should now have the following directory structure:

```
- opendarts
    |------ darts-engine 
    |------ darts-models
    \------ opendarts-linear-solvers 
```

Move into folder `helper_scripts` inside `opendarts-linear-solvers`:
 
```bash
cd helper_scripts
```

### Linux 
In Linux you just need to execute the `build_linux.sh` script with 

```bash
./build_linux.sh 
```

This will build all thirdparty dependencies and then `opendarts-linear-solvers`. It will also run the unit tests, to make sure everything was built correctly. Finally it installs `opendarts-linear-solvers` into the correct subfolder inside `darts-engines`: `opendarts/darts-engines/lib/opendarts-linear-solvers`.

### macOS 
In macOS you just need to execute the `build_macos.sh` script with 

```bash
./build_macos.sh 
```

This will build all thirdparty dependencies and then `opendarts-linear-solvers`. It will also run the unit tests, to make sure everything was built correctly. Finally it installs `opendarts-linear-solvers` into the correct subfolder inside `darts-engines`: `opendarts/darts-engines/lib/opendarts-linear-solvers`.

### Windows
To be added soon.

## Detailed build instructions for custom build
The tests and the library `linear_solvers` are built using `cmake`. Before building `linear_solvers` the user must build SuperLU.

### Building SuperLU 
#### Windows 
In the subfolder `thirdparty/SuperLU_5.2.1/` open the `SuperLU.sln` file with Visual studio (you need to have Visual Studio 2022). Build in `Release` mode.

After these two steps you need to have the following files to proceed to the next steps:

1. `thirdparty/SuperLU_5.2.1/x64/Release/SuperLU.lib`
2. `thirdparty/SuperLU_5.2.1/x64/Release/slu_blas.lib`


#### Linux and macOS
In the subfolder `thirdparty/SuperLU_5.2.1` copy the Makefile configuration files for your system.
You need to create a `conf.mk` and a `make.inc` files. Currently the following are available:
- `conf.mk`: following templates are available
  - `conf_gcc_linux.mk`: for linux systems
  - `conf_gcc-11_macOS_m1.mk`: for macOS with M1 processors (still experimental, use with care)
  
- `make.inc`: following templates are available
  - `make_gcc_linux.inc`: for linux systems
  - `make_gcc-11_macOS_m1.mk`: for macOS with M1 processors (still experimental, use with care)

run make 
```bash
make all
make install
```

After this step you need to have the following files to proceed to the next steps: 
1. `thirdparty/SuperLU_5.2.1/libsuperlu_5.1.a`
2. `thirdparty/SuperLU_5.2.1/libblas.a`

### Building `opendarts_linear_solvers`
#### Setup build system with cmake (all OSes)
Switch to the root directory of the repository, the one containing this README file.

Since it is recommended to have an out of source build, create a sub-directory where to build the code and switch to that directory

```bash
mkdir build_cmake
cd build_cmake
```

Then you can run `cmake` with your particular choice of general `cmake` options.
See below for specific options.

For example to build using an Xcode (generate a Xcode project), and install the built code in the folder `../install` use the following line

```bash
cmake  -D CMAKE_INSTALL_PREFIX=../install -G "Xcode"  ../
```

If instead you wanted to build with the specific version `g++-11` of your compiler and use `Makefiles` use the following line

```bash
cmake -D CMAKE_CXX_COMPILER=g++-11 -D CMAKE_INSTALL_PREFIX=../install ../
```

See `cmake` for more general options, [look here](https://cmake.org/cmake/help/latest/manual/cmake.1.html).

**NOTE:** Several different build systems can be used. If you type `cmake --help` you will get a list of the available build systems on your platform (Generators at the end of the help text). For example the following list on macOS:

```bash
The following generators are available on this platform (* marks default):
* Unix Makefiles                  = Generates standard UNIX makefiles.
  Ninja                           = Generates build.ninja files.
  Ninja Multi-Config              = Generates build-<Config>.ninja files.
  Watcom WMake                    = Generates Watcom WMake makefiles.
  Xcode                           = Generate Xcode project files.
  CodeBlocks - Ninja              = Generates CodeBlocks project files.
  CodeBlocks - Unix Makefiles     = Generates CodeBlocks project files.
  CodeLite - Ninja                = Generates CodeLite project files.
  CodeLite - Unix Makefiles       = Generates CodeLite project files.
  Eclipse CDT4 - Ninja            = Generates Eclipse CDT 4.0 project files.
  Eclipse CDT4 - Unix Makefiles   = Generates Eclipse CDT 4.0 project files.
  Kate - Ninja                    = Generates Kate project files.
  Kate - Unix Makefiles           = Generates Kate project files.
  Sublime Text 2 - Ninja          = Generates Sublime Text 2 project files.
  Sublime Text 2 - Unix Makefiles = Generates Sublime Text 2 project files.
```

Once the `cmake` step is finished you can build your code.

### CMake project specific options
- `ENABLE_TESTING`: Specifies if tests are built or not. `TRUE`: builds the tests. `FALSE`: tests are not built. <DEFAULT>: FALSE.
By default this is set to `FALSE`.

- `SET_CXX11_ABI_0`: Specifies if the compiler flag -`D_GLIBCXX_USE_CXX11_ABI=0` is to be added or not. This is required for compatibility with other codes that have the ABI set to a different value than the default. `TRUE`: adds the compiler flag. `FALSE`: does not add the compiler flag. By default this is set to `FALSE`.

### Build code Windows Visual Studio
In Windows with Visual Studio you need to first open a command prompt with your Visual Studio environment. To do that you need press your windows key and type `x64 native tools`. You now get a list for your available options. You need to select the 2022 version (if not available, please install it). Once you click on this option you get a command prompt that is ready to be used.

Now you can run the cmake command inside the `build_cmake` folder:

```bash
cmake -D CMAKE_INSTALL_PREFIX=../../darts-engines/lib/darts_linear_solvers -D ENABLE_TESTING=TRUE ../
```

Once `cmake` finishes, you should have the solution and project files available.
 
You need to open the solution file and then you can build the code. Be sure to select `Release` mode. 

You should first build the `ALL_BUILD` target (right click on that target and select build).

Then you can build the `RUN_TESTS` target to check if all is working correctly (right click on that target and select build).

Finally, you can build the `INSTALL` target to install the binaries and header files in the specified install folder `../../darts-engines/lib/darts_linear_solvers` (right click on that target and select build).

### Build code Xcode
Once you ran cmake specifying the `-G "Xcode"` option for building with `Xcode`, you should have the solution and project files available.
 
You need to open the solution file and then you can build the code. 

You can build all by building the `ALL_BUILD` target.

You can run the tests with the `RUN_TESTS` target.

If you choose the `INSTALL` target you will install the binaries and header files in the specified install folder.

### Build code `Makefile`
If you ran `cmake` on linux or macOS without specifying the `-G` flag or if you specified `-G "Unix Makefiles"` then `cmake` will generate Makefiles to build the code.

You can just run `make` to make all targets.

You can then run `make install` to install to the install folder you specified.

### Running tests 
#### cmake with Makefiles
If you are using `cmake` in the default mode (`Unix Makefiles`) then you can just run `ctest` and all tests will run.

#### cmake with Visual Studio or Xcode
If you are using `cmake` with Xcode or Visual Studio then you just need to first build the `ALL_BUILD` target and then build the `RUN_TESTS` target.

## Quick tips
To compile `opendarts_linear_solvers` on linux to integrate with opendarts, you can use the following. Assuming you have the following directory structure:

```
some_path
   |-opendarts  
       |- darts-engines
       |     |- lib 
       |           |- opendarts_linear_solvers
       |- opendarts-linear-solvers
             |- build_make 
```
The you can do:

```bash 
cd some_path/opendarts/opendarts-linear-solvers/build_make
cmake -D CMAKE_INSTALL_PREFIX=some_path/opendarts/darts-engines/lib/opendarts_linear_solvers -D SET_CXX11_ABI_0=TRUE -D ENABLE_TESTING=TRUE ../
```
Note that in this way we set the ABI flag and generate the tests.

To also make the `Debug` version, we would do

```bash 
cd some_path/opendarts/opendarts-linear-solvers/build_make
cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_INSTALL_PREFIX=some_path/opendarts/darts-engines/lib/opendarts_linear_solvers -D SET_CXX11_ABI_0=TRUE -D ENABLE_TESTING=TRUE ../
```
The `CMAKE_BUILD_TYPE` option is a built in `cmake` option, hence not explicitly added to the list of options available.
