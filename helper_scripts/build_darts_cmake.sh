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
  echo "$(basename "$0") [-h] [-c] [-t] [-w] [-m] [-r] [-a] [-b BOS_SOLVER_DIRECTORY] [-d INSTALL CONFIGURATION] [-j NUM THREADS] [-g g++-13]"
  echo "   Script to install opendarts on unix (linux and macOS)."
  echo "USAGE: "
  echo "   -h : displays this help menu."
  echo "   -c : cleans up build to prepare a new fresh build. Default: don't clean"
  echo "   -t : Enable testing: ctest of solvers. Default: don't test"
  echo "   -w : Enable generation of python wheel. Default: false"
  echo "   -m : Enable Multi-thread MT (with OMP) build. Warning: Solvers is not MT. Default: true"
  echo "   -r : Skip building thirdparty libraries (if you have them already compiled). Default: false"
  echo "   -a : Update private artifacts bos_solvers (instead of openDARTS solvers). This is meant to be used by CI/CD. Default: false"
  echo "   -b SPATH  : Path to bos_solvers (instead of openDARTS solvers), example: -b ./darts-linear-solvers containing lib/libdarts_linear_solvers.a (already compiled)."
  echo "   -d MODE   : Configuration for C++ code [Release, Debug]. Example: -d Debug"
  echo "   -j N      : Set number of threads (N) for compilation. Default: 8. Example: -j 4"
  echo "   -g g++VER : Specify a compiler (g++) version. Example: -g g++-13"
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
skip_req=false    # Skip building requirements.
config="Release"  # Default configuration (install).
NT=8              # Number of threads by default 8
gpp_version=g++   # Version of g++
special_gpp=false # Whether a special compiler version (g++) is specified.

while getopts ":chtwmrab:d:j:g:" option; do
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
    esac
done

# Amend possible contradictory inputs
if [ "$iter_solvers" == true ] && [ "$testing" == true ]; then
    # tests are only available in open-DARTS, bos_solvers do not have testing
    testing=false
fi
if [ "$iter_solvers" == false ] && [ "$MT" == true ]; then
   echo '\n Warning: Open-DARTS linear solvers do not support multi-threading. Switched to the sequentional build.'
   MT=false
fi
# ------------------------------------------------------------------------------

# Amend the path if necessary --------------------------------------------------
# If the script is called from inside the folder helper_scripts, then place us 
# at the root directory open-darts.
if [[ "$(basename $PWD)" == "helper_scripts" ]]; then
    cd ../
fi
# ------------------------------------------------------------------------------

rm -rf darts/*.so
rm -rf dist
	
# Build loop -------------------------------------------------------------------
if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build
    echo '\n   Cleaning build folder'
    rm -r build
else
    if [[ "$skip_req" == false ]]; then
        # update submodules
        echo -e "\n- Update submodules: START \n"
        # clean-up previous versions.
        rm -rf thirdparty/eigen thirdparty/pybind11 thirdparty/mshIO thirdparty/hypre
        git submodule sync --recursive
        git submodule update --recursive --init
        echo -e "\n- Update submodules: DONE! \n"

        # Install requirements
        echo -e "\n- Install requirements: START \n"

        echo -e "\n-- Install EIGEN 3 \n"
        cd thirdparty
        mkdir -p build/eigen
        cd build/eigen
        cmake -D CMAKE_INSTALL_PREFIX=../../install ../../eigen/  &> ../../../make_eigen.log
        make install -j $NT &>> ../../../make_eigen.log
        cd ../../

        echo -e "\n-- Install Hypre: START\n"
        cd hypre/src/cmbuild
        # Setup hypre build with no MPI support (we only use single processor)
        # Request build of tests and examples just to be sure everything is fine in the build 
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
        echo -e "\n- Install requirements: DONE! \n"
    else
        echo -e "\n- Requirements already installed \n"
    fi

    if [[ "$bos_solvers_artifact" == true ]]; then
        bos_solvers_dir=$PWD"/engines/lib/darts_linear_solvers"
    fi

    echo "\n========================================================================"
    echo "| Building openDARTS: START "
    echo "========================================================================\n"

    # Setup build folder
    rm -rf build # Just in case
    mkdir build
    cd build

    # Setup build with cmake
    cmake_options=" -D SET_CXX11_ABI_0=TRUE -D CMAKE_BUILD_TYPE=${config} -D CMAKE_INSTALL_PREFIX=../darts/"

    if [[ "$testing" == true ]]; then
        cmake_options+=" -D ENABLE_TESTING=ON"
    fi
    if [[ "$special_gpp" == true ]]; then
        cmake_options+=" -D CMAKE_CXX_COMPILER=${gpp_version}"
    fi
    if [[ "$MT" == true ]]; then
        cmake_options+=" -D OPENDARTS_CONFIG=MT"
    fi
    if [[ ! -z "$bos_solvers_dir" ]]; then
        cmake_options+=" -D BOS_SOLVERS_DIR=${bos_solvers_dir}"
    fi

    echo "CMake options: $cmake_options" # Report to user the CMake options
    cmake $cmake_options .. &> ../make_darts.log

    # Build and install openDARTS
    make install -j $NT &>> ../make_darts.log

    # Test
    if [[ "$testing" == true ]]; then
        ctest
    fi

    cd ../

    echo "\n========================================================================"
    echo "| Building openDARTS: DONE! "
    echo "========================================================================\n"

    echo "************************************************************************"
    echo "| Building python package open-darts: START "
    echo "************************************************************************\n"

    # generating build info of darts-package
    python3 darts/print_build_info.py

    # build darts.whl
    if [[ "$wheel" == true ]]; then
        python3 setup.py clean
        python3 setup.py build bdist_wheel 2>&1 | tee make_wheel.log
        echo "-- Python wheel generated! \n"
    fi

    # installing python package
    python3 -m pip install . 2>&1 | tee -a make_wheel.log

    echo "\n************************************************************************"
    echo "| Building python package open-darts: DONE! "
    echo "************************************************************************\n"
fi
# ------------------------------------------------------------------------------