#!/bin/bash

# Setup shell script run -------------------------------------------------------
# Exit when any command fails
set -e
set -o pipefail
# ------------------------------------------------------------------------------

################################################################################
# Help info                                                                    #
################################################################################
Help_Info()
{
  echo "$(basename "$0") [-h] [-c] [-t] [-w] [-m] [-r] [-a] [-b BOS_SOLVER_DIRECTORY] [-d INSTALL CONFIGURATION] [-j NUM THREADS] [-g g++-13] [-p] [-v]"
  echo "   Script to install opendarts on unix (linux and macOS)."
  echo "USAGE: "
  echo "   -h : displays this help menu."
  echo "   -c : cleans up build to prepare a new fresh build. Default: don't clean"
  echo "   -t : Enable testing: ctest of solvers. Default: don't test"
  echo "   -w : Enable generation of python wheel. Default: false"
  echo "   -m : Enable Multi-thread MT (with OMP) build. Warning: Solvers is not MT. Default: true"
  echo "   -G : Enable GPU build. Warning: Requires GPU bos solvers. Default: false"
  echo "   -r : Skip building thirdparty libraries (if you have them already compiled). Default: false"
  echo "   -a : Update private artifacts bos_solvers (instead of openDARTS solvers). This is meant to be used by CI/CD. Default: false"
  echo "   -b SPATH  : Path to bos_solvers (instead of openDARTS solvers), example: -b ./darts-linear-solvers containing lib/libdarts_linear_solvers.a (already compiled)."
  echo "   -d MODE   : Configuration for C++ code [Release, Debug]. Example: -d Debug"
  echo "   -j N      : Set number of threads (N) for compilation. Default: 8. Example: -j 4"
  echo "   -g g++VER : Specify a compiler (g++) version. Example: -g g++-13"
  echo "   -p        : Enable building & installing IPhreeqc (third-party)  (OFF by default)"
  echo "   -v        : Enable build with valgrind support (OFF by default)"
}
################################################################################
# Main program                                                                 #
################################################################################

# Read input arguments ---------------------------------------------------------
clean_mode=false  # Set mode to clean up, cleans build to prepare for fresh new build
testing=false     # Whether to enable the testing (ctest) of solvers.
wheel=false       # Whether to generate python wheel.
bos_solvers_artifact=false # Fetch the bos_solvers library from artifacts (for CI/CD purposes)
iter_solvers=false # Iterative linear solvers, will be set below depending on -a and -b flags
MT=true           # Build openDARTS multi-threaded. This is for engines and bos_solvers (if defined)
GPU=false         # Build openDARTS with GPU. This applies to engines and bos_solvers.
skip_req=false    # Skip building requirements.
config="Release"  # Default configuration (install).
NT=8              # Number of threads by default 8
gpp_version=g++   # Version of g++
special_gpp=false # Whether a special compiler version (g++) is specified.
valgrind=false    # Whether support valgrind profiling or not

while getopts ":chtwmrab:d:j:g:Gpv" option; do
    case "$option" in
        h) # Display help
           Help_Info
           exit;;
        c) # Clean mode
           clean_mode=true;;
        t) # Testing
           testing=true;;
        w) # Generate wheel
           wheel=true;;
        m) # Multi-thread
           MT=true;;
        r) # skip buildrequirements
           skip_req=true;;
        a) # Fetch the bos_solvers library from artifacts
           bos_solvers_artifact=true
           iter_solvers=true;;
        b) # path to bos_solvers
           bos_solvers_dir=${OPTARG}
           iter_solvers=true;;
        d) # Select a mode
           config=${OPTARG};;
        j) # Number of threads
           NT=${OPTARG};; 
        g) # gpp version
           special_gpp=true
           gpp_version=${OPTARG};;
        G) # GPU build
           GPU=true;;
        p) # Enable IPhreeqc support
           phreeqc=true;;
        v) # Valgrind build => Debug + symbols
           valgrind=true;;
    esac
done

# Amend possible contradictory inputs
if [ "$iter_solvers" == true ] && [ "$testing" == true ]; then
    # tests are only available in open-DARTS, bos_solvers do not have testing
    testing=false
fi

if [ "$iter_solvers" == false ]; then
  if [ "$GPU" == true ]; then
    echo GPU build requires GPU bos solvers. Specify the path with -b.
    exit 1
  elif [ "$MT" == true ]; then
   echo -e '\n Warning: Open-DARTS linear solvers do not support multi-threading. Switched to the sequentional build.'
   MT=false
  fi
fi
#
# ------------------------------------------------------------------------------

# Amend the path if necessary --------------------------------------------------
# If the script is called from inside the folder helper_scripts, then place us 
# at the root directory open-darts.
if [[ "$(basename $PWD)" == "helper_scripts" ]]; then
    cd ../
fi
# ------------------------------------------------------------------------------

rm -rf dist
rm -rf darts/*.so
if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build
    echo '\n   Cleaning build folder'
    rm -rf build
fi


# Build -------------------------------------------------------------------
if [[ "$skip_req" == false ]]; then
    # update submodules
    echo -e "\n- Update submodules: START \n"
    # clean-up previous versions.
    rm -rf thirdparty/eigen \
            thirdparty/pybind11 \
            thirdparty/MshIO \
            thirdparty/hypre \
            thirdparty/iphreeqc
    # synchronize & update submodules
    git submodule sync --recursive
    git submodule update --init --recursive -- \
            thirdparty/eigen \
            thirdparty/pybind11 \
            thirdparty/MshIO \
            thirdparty/hypre
    if [[ $phreeqc == "true" ]]; then
        git submodule update --init --recursive thirdparty/iphreeqc
    fi
    # update submodules finished
    echo -e "\n- Update submodules: DONE! \n"

    # Install requirements
    echo -e "\n- Install requirements: START \n"
    cd thirdparty

    echo -e "\n-- Install EIGEN 3 \n"
    mkdir -p build/eigen
    cd build/eigen
    cmake -D CMAKE_INSTALL_PREFIX=../../install ../../eigen/  &> ../../../make_eigen.log
    make install -j $NT &>> ../../../make_eigen.log
    cd ../../

    echo -e "\n-- Install Hypre: START\n"
    cd hypre/src/cmbuild
    # Setup hypre build with no MPI support (we only use single processor)
    # Request build of tests and examples just to be sure everything is fine in the build 
    # For debugging: -DHYPRE_ENABLE_PRINT
    cmake -D HYPRE_BUILD_TESTS=ON -D HYPRE_BUILD_EXAMPLES=ON -D HYPRE_WITH_MPI=OFF -D CMAKE_INSTALL_PREFIX=../../../install .. &> ../../../../make_hypre.log
    make install -j $NT &>> ../../../../make_hypre.log
    cd ../../../
    echo -e "\n--- Building Hypre: DONE!\n"

    echo -e "\n-- Install SuperLU \n"
    cd SuperLU_5.2.1

    if [[ "$OSTYPE" == "darwin"* ]]; then
        cp conf_gcc-11_macOS_m1.mk conf.mk
        cp make_gcc-11_macOS_m1.inc make.inc
    else
        cp conf_gcc_linux.mk conf.mk
        cp make_gcc_linux.inc make.inc
    fi

    make -j $NT &> ../../make_superlu.log
    make install -j $NT &>> ../../make_superlu.log
    cd ../../

    if [[ "$bos_solvers_artifact" == true ]]; then
        cd engines
        ./update_private_artifacts.sh $SMBNAME $SMBLOGIN $SMBPASS
        cd ..
    fi

    #----------------------------------------------------------------------#
    # Build & install IPhreeqc if requested                                 #
    #----------------------------------------------------------------------#
    if [[ "$phreeqc" == true ]]; then
        echo -e "\n-- Install IPhreeqc: START\n"
        cd thirdparty
        mkdir -p build/iphreeqc && cd build/iphreeqc
        cmake \
            -D CMAKE_INSTALL_PREFIX=../../install/iphreeqc \
            -D BUILD_TESTING=OFF \
            -D BUILD_SHARED_LIBS=ON \
            ../../iphreeqc            &> ../../../make_iphreeqc.log
        make install -j $NT           &>> ../../../make_iphreeqc.log
        cd ../../..
        echo -e "\n--- Building IPhreeqc: DONE!\n"
    fi

    echo -e "\n- Install requirements: DONE! \n"
else
    echo -e "\n- Requirements already installed \n"
fi

if [[ "$bos_solvers_artifact" == true ]]; then
    bos_solvers_dir=$PWD"/engines/lib/darts_linear_solvers"
fi

echo -e "\n========================================================================"
echo "| Building openDARTS: START "
echo -e "========================================================================\n"

# Setup build folder
mkdir -p build
cd build
rm -f CMakeCache.txt  # ensures Cmake doesn't work on outdated configuration

# If valgrind requested, force Debug
if [[ "$valgrind" = true ]]; then
    config="Debug"
fi

# Setup build with cmake
cmake_options="-D CMAKE_BUILD_TYPE=${config}"

if [[ "$valgrind" = true ]]; then
    cmake_options+=" -D ENABLE_VALGRIND=ON"
fi
if [[ "$testing" == true ]]; then
    cmake_options+=" -D ENABLE_TESTING=ON"
fi
if [[ "$special_gpp" == true ]]; then
    cmake_options+=" -D CMAKE_CXX_COMPILER=${gpp_version}"
fi

build=ST    
if [[ "$GPU" == true ]]; then
  build=GPU
elif [[ "$MT" == true ]]; then
  build=MT
fi
cmake_options+=" -D OPENDARTS_CONFIG=$build"

if [[ ! -z "$bos_solvers_dir" ]]; then
    cmake_options+=" -D BOS_SOLVERS_DIR=${bos_solvers_dir}"
fi

# Pass WITH_PHREEQC to CMake to copy shared library
if [[ "$phreeqc" == true ]]; then
    cmake_options+=" -DWITH_PHREEQC=ON"
    echo "Phreeqc support: ENABLED"
else
    echo "Phreeqc support: DISABLED"
fi

echo -e "CMake options: $cmake_options\n" # Report to user the CMake options
cmake $cmake_options .. 2>&1 | tee ../make_darts.log

# Build and install openDARTS
make install -j $NT 2>> ../make_darts.log

# Test
if [[ "$testing" == true ]]; then
    ctest
fi

cd ../

echo -e "\n========================================================================"
echo "| Building openDARTS: DONE! "
echo -e "========================================================================\n"

echo "************************************************************************"
echo "| Building python package open-darts: START "
echo -e "************************************************************************\n"

# generating build info of darts-package
python3 darts/print_build_info.py

# build darts.whl
if [[ "$wheel" == true ]]; then
    cp CHANGELOG.md darts
    python3 setup.py clean
    python3 setup.py build bdist_wheel 2>&1 | tee make_wheel.log
    echo -e "-- Python wheel generated! \n"
fi

# installing python package with -e flag for interactive install (changes will be applied live)
python3 -m pip install . 2>&1 | tee -a make_wheel.log

echo -e "\n************************************************************************"
echo "| Building python package open-darts: DONE! "
echo -e "************************************************************************\n"
# ------------------------------------------------------------------------------
