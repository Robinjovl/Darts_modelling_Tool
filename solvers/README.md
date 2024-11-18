# Opendarts Linear Solvers
Simple block CSR matrix storage implementation

## Contents description
The folder `linear_solvers` contains the open source implementation of block csr
matrix and linear solvers. Currently, data storage and direct solver using Super LU
are the only functionalities, i.e., linear algebra
such as matrix-matrix multiplication of matrix-vector multiplication is **NOT** implemented.

The folder `open-darts/tests/cpp`, contains the tests that also show how to use the library `opendarts-solvers`.

## Quick build instructions
If you wish to use `opendarts-solvers` in `openDARTS`, the easiest way to build and install `opendarts-solvers` is to follow the instructions below. As a pre-requisite you need to have the `gcc` compiler installed (on macOS you need version 11 or later installed with homebrew and available as `gcc-11`). Additionally you need to have `cmake` available.

Clone this project and then move into folder `helper_scripts` inside `solvers`:

```bash
cd solvers/helper_scripts
```

### Unix systems (Linux and MacOS)

In Linux execute the script: `./build_linux.sh`
Or in macOS execute the script `./build_macos.sh`

This will build all thirdparty dependencies and `opendarts-solvers`. Adding the option `-t` will also run the unit tests, to make sure everything was built correctly. Finally it installs `opendarts-solvers` into the correct subfolder inside `engines`: `open-darts/engines/lib/solvers`.

Explore other options: `./build_linux.sh -h`

### Windows

Execute `build_windows.bat`. This will build all thirdparty dependencies and `openDARTS.sln` containing only the solvers.

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
You need to create a `conf.mk` and a `make.inc` files from the following templates:
- `conf.mk`
  - `conf_gcc_linux.mk`: for linux systems
  - `conf_gcc-11_macOS_m1.mk`: for macOS with M1 processors (still experimental, use with care)
  
- `make.inc`
  - `make_gcc_linux.inc`: for linux systems
  - `make_gcc-11_macOS_m1.mk`: for macOS with M1 processors (still experimental, use with care)

Run make
```bash
make all
make install
```

After this step you need to have the following files to proceed to the next steps:
1. `thirdparty/SuperLU_5.2.1/libsuperlu_5.1.a`
2. `thirdparty/SuperLU_5.2.1/libblas.a`

### Building `opendarts_linear_solvers`
#### Setup build system with cmake (all OSes)
Switch to the root directory of the repository (open-darts).

Since it is recommended to have an out of source build, create a sub-directory where to build the code and switch to that directory

```bash
mkdir build
cd build
```

Then you can run `cmake` with your particular choice of general `cmake` options.
See below for specific options.

For example to build using an Xcode (generate a Xcode project), and install the built code in the folder `../install` use the following line

```bash
cmake  -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON -G "Xcode"  ../
```

If instead you wanted to build with the specific version `g++-11` of your compiler and use `Makefiles` use the following line

```bash
cmake -D CMAKE_CXX_COMPILER=g++-11 -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON ../
```

See `cmake` for more general options, [look here](https://cmake.org/cmake/help/latest/manual/cmake.1.html).

**NOTE:** Several different build systems can be used. If you type `cmake --help` you will get a list of the available build systems on your platform (Generators at the end of the help text).

Once the `cmake` step is finished you can build your code.

### CMake project specific options

- `ENABLE_TESTING=FALSE`: Specifies if tests are built or not. `TRUE`: builds the tests. `FALSE`: tests are not built. <DEFAULT>: FALSE.
By default this is set to `FALSE`.
- `SET_CXX11_ABI_0=FALSE`: Specifies if the compiler flag -`D_GLIBCXX_USE_CXX11_ABI=0` is to be added or not. This is required for compatibility with other codes that have the ABI set to a different value than the default. `TRUE`: adds the compiler flag. `FALSE`: does not add the compiler flag. By default this is set to `FALSE`.
- `ONLY_SOLVERS=ON`: Specifies that you only want to install the solvers and not the whole openDARTS. `OFF`: compiles the whole darts project. `ON`: compiles only the linear solvers.
- `CMAKE_BUILD_TYPE=RelWithDebInfo`: Specifies the build type: Release or Debug. This is a built in `cmake` option. The default is `RelWithDebInfo`. For more information [look here](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html).

### Build code in Visual Studio (Windows)
In Windows with Visual Studio you need to first open a command prompt with your Visual Studio environment. To do that you need press your windows key and type `x64 native tools`. You now get a list for your available options. You need to select the 2022 version (if not available, please install it). Once you click on this option you get a command prompt that is ready to be used.

Now you can run the cmake command inside the `build` folder:

```bash
cmake -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON ../
```

Once `cmake` finishes, you should have the solution and project files available.

Open the solution file and then you can build the code. Select `Release` mode.

Build the `ALL_BUILD` target (right click on that target and select build).

Then you can build the `RUN_TESTS` target to check if all is working correctly (right click on that target and select build).

Finally, you can build the `INSTALL` target to install the binaries and header files in the specified install folder `../engines/lib/solvers` (right click on that target and select build).

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
#### CMake with Makefiles
If you are using `cmake` in the default mode (`Unix Makefiles`) then you can just run `ctest` and all tests will run.

#### CMake with Visual Studio or Xcode
If you are using `cmake` with Xcode or Visual Studio build the `ALL_BUILD` target and then build the `RUN_TESTS` target.

## Quick tips
To compile `opendarts_solvers` on linux to integrate with `open-darts`, you can better compile the entire project. Assuming you have the following directory structure:

```
some_path
   |-open-darts  
       |- engines
       |     |- lib 
       |         |- solvers
       |- solvers
       |- build 
```

Then you can do:

```bash
cd some_path/opendarts/build
cmake -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D SET_CXX11_ABI_0=TRUE -D ENABLE_TESTING=TRUE ../
```

Note that in this way we set the ABI flag and generate the tests.
