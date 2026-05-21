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
  echo "   -p        : Enable building & installing IPhreeqc and Reaktoro (OFF by default, requires active Conda env)"
  echo "   -v        : Enable build with valgrind support (OFF by default)"
  echo "   CUDA_ARCH env var: Specify CUDA architecture(s), e.g. \"70\" or \"70;80\""
}

ensure_reaktoro_conda()
{
  echo -e "\n-- Install Reaktoro (conda): START\n"

  if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("reaktoro") else 1)
PY
  then
    echo "- Reaktoro already available in current Python environment"
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: 'conda' command not found. Install Conda (see https://reaktoro.org/installation/installation-using-conda.html) and activate an environment before using -p."
    exit 1
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Error: CONDA_PREFIX is empty. Activate the target conda environment (e.g., 'conda activate rkt') before running with -p."
    exit 1
  fi

  # Check Python version compatibility (Reaktoro on conda-forge requires Python >=3.10, <3.13)
  local py_minor
  py_minor=$(python3 -c "import sys; print(sys.version_info.minor)")
  if [[ "$py_minor" -lt 10 || "$py_minor" -ge 13 ]]; then
    local py_version
    py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Warning: Reaktoro on conda-forge requires Python >=3.10 and <3.13, but the current environment has Python $py_version."
    echo ""
    echo "To install Reaktoro, create a compatible conda environment (e.g., Python 3.12):"
    echo "  conda create -n darts-rkt python=3.12 -y"
    echo "  conda activate darts-rkt"
    echo ""
    echo "Then re-run this script with the -p flag."
    return
  fi

  local reaktoro_log="$PWD/make_reaktoro.log"
  echo "+ conda install -y -c conda-forge -p ${CONDA_PREFIX} reaktoro" | tee -a "$reaktoro_log"
  conda install -y -c conda-forge -p "${CONDA_PREFIX}" reaktoro 2>&1 | tee -a "$reaktoro_log"
  echo -e "\n--- Installing Reaktoro: DONE!\n"
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
CUDA_ARCH="${CUDA_ARCH:-}"

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

if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build
    echo -e '\n   Cleaning build folder'
    rm -rf build
    rm -rf dist
    rm -rf darts/*.so
else
    rm -rf dist
fi


# Build -------------------------------------------------------------------
if [[ "$skip_req" == false ]]; then
    # update submodules
    echo -e "\n- Update submodules: START \n"
    # clean-up previous versions.
    rm -rf thirdparty/pybind11 \
            thirdparty/MshIO \
            thirdparty/hypre \
            thirdparty/iphreeqc
    # synchronize & update submodules
    git submodule sync --recursive
    git submodule update --init --recursive -- \
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

    mkdir -p build
    echo -e "\n-- Install Hypre: START\n"
    cd hypre/src/cmbuild
    # Setup hypre build with no MPI support (we only use single processor)
    # Request build of tests and examples just to be sure everything is fine in the build
    # For debugging: -DHYPRE_ENABLE_PRINT
    cmake -D HYPRE_BUILD_TESTS=ON \
          -D HYPRE_BUILD_EXAMPLES=ON \
          -D HYPRE_WITH_MPI=OFF \
          -D CMAKE_INSTALL_PREFIX=../../../install \
          .. &> ../../../../make_hypre.log
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

if [[ ! -z "$CUDA_ARCH" ]]; then
    cmake_options+=" -D CUDA_ARCH=${CUDA_ARCH}"
fi

if [[ -n "${OD_CMAKE_ARGS:-}" ]]; then
    cmake_options+=" ${OD_CMAKE_ARGS}"
fi

echo -e "CMake options: $cmake_options\n" # Report to user the CMake options
cmake $cmake_options .. 2>&1 | tee ../make_darts.log

# Build and install openDARTS
# Under valgrind (-O2 -g) the auto-generated super_part*.cpp / rates_part*.cpp /
# all_part*.cpp interpolator TUs peak 2-4 GB resident per cc1plus due to massive
# template stamping (recursive_exposer over MAX_DIMS x N_OPS combinations). On a
# typical 8 GB CI runner with -j 8 the OOM killer truncates cc1plus mid-write,
# leaving the assembler choking on a partial pseudo-op (".uleb12" instead of
# ".uleb128"). The mitigation has two parts:
#   1. Pre-build interpolators with -j 2 max — each cc1plus instance gets enough
#      headroom regardless of NT or runner memory profile.
#   2. The follow-up full-build pass at -j NT then only links / copies the already
#      compiled interpolator objects; no large recompiles happen there.
# Also pass -l so make backs off if the system load average climbs (extra safety
# when the runner is shared).
if [[ "$valgrind" == true ]]; then
    HEAVY_NT=2
    if [[ "$NT" -lt "$HEAVY_NT" ]]; then HEAVY_NT="$NT"; fi
    LOAD_LIMIT=$(( NT / 2 > 0 ? NT / 2 : 1 ))
    echo "-- Pre-building interpolators target with -j $HEAVY_NT -l $LOAD_LIMIT (valgrind OOM mitigation)"
    make interpolators -j "$HEAVY_NT" -l "$LOAD_LIMIT" 2>&1 | tee -a ../make_darts.log
fi
cmake --build . --target install --parallel "$NT" 2>&1 | tee -a ../make_darts.log

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
    python3 -m pip install --upgrade build 2>&1 | tee make_wheel.log
    python3 -m build --wheel 2>&1 | tee -a make_wheel.log
    echo -e "-- Python wheel generated! \n"
fi

# installing python package with -e flag for interactive install (changes will be applied live)
python3 -m pip install . 2>&1 | tee -a make_wheel.log

if [[ "$phreeqc" == true ]]; then
    ensure_reaktoro_conda
fi

echo -e "\n************************************************************************"
echo "| Building python package open-darts: DONE! "
echo -e "************************************************************************\n"

# Build warnings/errors summary -----------------------------------------------
report_build_summary()
{
  local warn_pattern=': warning[: #]'
  local err_pattern=': error[: #]'

  # (component_name, log_file) pairs
  local components=(
    "Hypre:make_hypre.log"
    "SuperLU:make_superlu.log"
    "IPhreeqc:make_iphreeqc.log"
    "open-DARTS:make_darts.log"
  )

  # Count warnings/errors before printing (avoid reading make_darts.log while appending)
  local -A warn_counts err_counts
  for entry in "${components[@]}"; do
    local name="${entry%%:*}"
    local logfile="${entry##*:}"
    if [[ -f "$logfile" ]]; then
      warn_counts[$name]=$(grep -cE "$warn_pattern" "$logfile" 2>/dev/null || true)
      err_counts[$name]=$(grep -cE "$err_pattern" "$logfile" 2>/dev/null || true)
    fi
  done

  # Print to stdout and append to make_darts.log
  {
    echo ""
    echo "========================================="
    echo " Build warnings/errors summary"
    echo "========================================="
    printf " %-14s | %8s | %6s\n" "Component" "Warnings" "Errors"
    echo " -----------------------------------------"

    for entry in "${components[@]}"; do
      local name="${entry%%:*}"
      if [[ -n "${warn_counts[$name]+x}" ]]; then
        printf " %-14s | %8d | %6d\n" "$name" "${warn_counts[$name]}" "${err_counts[$name]}"
      fi
    done

    echo "========================================="

    local darts_warnings=${warn_counts[open-DARTS]:-0}
    if [[ $darts_warnings -gt 0 ]]; then
      echo ""
      echo " open-DARTS unique warnings:"
      grep -E "$warn_pattern" make_darts.log 2>/dev/null | sort -u | head -100
    fi

    echo ""
    echo "OPENDARTS_WARNING_COUNT=$darts_warnings"
  } | tee -a make_darts.log
}
report_build_summary
# ------------------------------------------------------------------------------
