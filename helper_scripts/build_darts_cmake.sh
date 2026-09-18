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
  echo "   -h               : displays this help menu."
  echo "   -c               : clean rebuild of everything, including thirdparty (HYPRE/SuperLU). Default: reuse existing thirdparty build if present"
  echo "   -t               : Enable testing: ctest of solvers and install open-darts[test]. Default: don't test"
  echo "   -w               : Enable generation of python wheel. Default: false"
  echo "   -m               : Enable Multi-thread MT (OpenMP) build. Engines, interpolators and the in-tree GMRES kernels run in parallel; HYPRE preconditioners (CPR/MGR) are sequential. Default: true"
  echo "   -G               : Enable GPU build. Uses the in-tree open-source solvers unless -b is given. Default: false"
  echo "   -r               : Skip building thirdparty libraries (if you have them already compiled). Default: false"
  echo "   -a               : Update private artifacts bos_solvers (instead of openDARTS solvers). This is meant to be used by CI/CD. Default: false"
  echo "   -b SPATH         : Path to bos_solvers (instead of openDARTS solvers), example: -b ./darts-linear-solvers containing lib/libdarts_linear_solvers.a (already compiled)."
  echo "   -d MODE          : Configuration for C++ code [Release, Debug, RelWithDebInfo]. RelWithDebInfo = -O2 -g (optimized + debug symbols). Example: -d RelWithDebInfo"
  echo "   -j N             : Set number of threads (N) for compilation. Default: 8. Example: -j 4"
  echo "   -g g++VER        : Specify a compiler (g++) version. Example: -g g++-13"
  echo "   -p               : Enable building & installing IPhreeqc and Reaktoro (OFF by default, requires active Conda env)"
  echo "   -v               : Enable build with valgrind support (OFF by default)"
  echo "   CUDA_ARCH env var: Specify CUDA architecture(s), e.g. \"70\" or \"70;80\""
  echo "   HYPRE_OPENMP env : Build HYPRE with OpenMP (parallel BoomerAMG/ILU in CPR/MGR). Default: true; set HYPRE_OPENMP=0 to build HYPRE sequentially. Slightly changes solver numerics. Requires -c to (re)build HYPRE."
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
    echo "To install Reaktoro, create a compatible conda environment (e.g., Python 3.11):"
    echo "  conda create -n darts-rkt python=3.11 -y"
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
install_test_extra=false # Whether to install the Python package with the test extra.
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
HYPRE_OPENMP="${HYPRE_OPENMP:-true}" # Build HYPRE with its own OpenMP threading (on by default; HYPRE_OPENMP=0 opts out)

while getopts ":chtwmrab:d:j:g:Gpv" option; do
    case "$option" in
        h) # Display help
           Help_Info
           exit;;
        c) # Clean mode
           clean_mode=true;;
        t) # Testing
           testing=true
           install_test_extra=true;;
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

# Thirdparty (HYPRE, SuperLU, IPhreeqc) builds run their own `cmake` invocation
# and don't inherit CMAKE_CXX_COMPILER from the main project's cmake_options below
# In macOS: without this they'd silently fall back to the system Apple Clang
# even when -g gcc/g++ was requested for the main build.
thirdparty_compiler_flags=""
if [[ "$special_gpp" == true ]]; then
    gcc_version="${gpp_version/g++/gcc}"
    thirdparty_compiler_flags="-D CMAKE_C_COMPILER=${gcc_version} -D CMAKE_CXX_COMPILER=${gpp_version}"
fi

if [[ "$config" != "Release" && "$config" != "Debug" && "$config" != "RelWithDebInfo" ]]; then
    echo "Error: Invalid build configuration \"$config\". Valid options: Release, Debug, RelWithDebInfo."
    exit 1
fi

# Amend possible contradictory inputs
if [ "$iter_solvers" == true ] && [ "$testing" == true ]; then
    # tests are only available in open-DARTS, bos_solvers do not have testing
    testing=false
fi

# If valgrind requested, force Debug early (affects thirdparty builds)
if [[ "$valgrind" = true ]]; then
    config="Debug"
fi

if [ "$iter_solvers" == false ]; then
  if [ "$GPU" == true ]; then
    # GPU builds default to the in-tree open-source solvers (darts.linear_solvers,
    # including the GPU solver wrappers). Pass -b <path> to build against the
    # proprietary bos_solvers instead.
    echo -e '\n openDARTS GPU build using the in-tree open-source solvers (no bos_solvers).'
  elif [ "$MT" == true ]; then
    # The in-tree open-source build now supports OpenMP: the engines assemble the
    # block_csr_matrix Jacobian in parallel over a real multi-threaded row
    # partition, the interpolators evaluate in parallel, and the in-tree GMRES
    # Krylov kernels (SpMV, dot, axpy) run in parallel. The HYPRE-based
    # preconditioner stages (CPR/MGR BoomerAMG/ILU) are threaded too, unless the
    # build opted out with HYPRE_OPENMP=0.
    echo -e '\n openDARTS multi-threaded (OpenMP) build using the in-tree open-source solvers (no bos_solvers).'
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
# Remove previously built Python extension modules and shared libraries.
# Build artifacts live both directly under darts/ (engines, discretizer, ...)
# and in subpackages such as darts/linear_solvers/ (the compiled linear_solvers module and
# libopendarts_linear_solvers). A flat darts/*.so glob misses the latter, leaving a
# stale solvers library that shadows the fresh build, so clean recursively.
# Note: the unversioned *.so glob intentionally excludes the bundled
# libstdc++.so.6 (a copied runtime dependency, re-installed by CMake).
find darts -type f \( -name '*.so' -o -name '*.pyd' -o -name '*.dylib' \) -delete 2>/dev/null || true
if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build: darts build/ plus the thirdparty
    # HYPRE build tree and install prefix, so -c forces a complete rebuild from
    # scratch (including thirdparty).
    echo -e '\n   Cleaning build folder'
    rm -rf build
    rm -rf thirdparty/hypre/src/cmbuild thirdparty/build thirdparty/install
fi

# Reuse an existing thirdparty build when one is present: if HYPRE is already
# installed and this is not a clean (-c) rebuild, skip rebuilding the
# requirements. -a (CI bos artifact) and -p (IPhreeqc) still run the full
# requirements step. Use -c to force a fresh thirdparty rebuild.
if [[ "$skip_req" == false && "$clean_mode" == false \
      && "$bos_solvers_artifact" == false && "${phreeqc:-false}" != true ]]; then
    if compgen -G "thirdparty/install/lib*/libHYPRE.*" >/dev/null 2>&1; then
        echo -e "\n- Reusing existing thirdparty build (HYPRE found); use -c for a fresh rebuild."
        skip_req=true
    fi
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
            thirdparty/hypre \
            thirdparty/superlu

    if [[ $phreeqc == "true" ]]; then
        git submodule update --init --recursive thirdparty/iphreeqc
    fi

    # AMGX backs the default GPU solver (GPU_GMRES_CPR_AMGX_ILU) and is ON by
    # default for GPU builds, so its submodule must be present. Init it here for
    # GPU builds (CI already checks out submodules recursively; this makes a
    # non-recursive local clone work too). Disable with WITH_AMGX=OFF / --no-amgx.
    if [[ "$GPU" == true && "$OD_CMAKE_ARGS" != *"WITH_AMGX=OFF"* ]]; then
        git submodule update --init --recursive thirdparty/AMGX
    fi
    # update submodules finished
    echo -e "\n- Update submodules: DONE! \n"

    # Install requirements
    echo -e "\n- Install requirements: START \n"
    cd thirdparty

    mkdir -p build
    echo -e "\n-- Install Hypre: START\n"
    mkdir -p hypre/src/cmbuild
    cd hypre/src/cmbuild
    # Setup hypre build with no MPI support (we only use single processor)
    # MGR support is enabled by default in HYPRE (no special flag needed)
    # The MGR (Multiplicative Grid Reduction) solver is always built in HYPRE
    # Tests/examples are never run, only the library is used, so don't build them
    # (on Windows they also raced on CMake's generate.stamp under parallel MSBuild)
    # For debugging: -DHYPRE_ENABLE_PRINT
    # NOTE: this branch pins a newer HYPRE (thirdparty/hypre 341f9089) whose CMake
    # option is HYPRE_ENABLE_MPI (the pre-merge development tree used the older
    # HYPRE_WITH_MPI spelling for its older pin).
    # Build HYPRE with its own OpenMP threading (parallel BoomerAMG / HYPRE_ILU
    # smoothers + SpMV). ON by default. Note it changes
    # solver numerics -- HYPRE's hybrid smoothers go processor-local, so results
    # are not identical to a sequential HYPRE and iteration counts may shift.
    # Set HYPRE_OPENMP=0 to build HYPRE sequentially.
    hypre_omp_flag="-D HYPRE_ENABLE_OPENMP=ON"
    if [[ "$HYPRE_OPENMP" == "false" || "$HYPRE_OPENMP" == "0" || "$HYPRE_OPENMP" == "OFF" ]]; then
        echo "-- HYPRE OpenMP disabled (HYPRE_OPENMP=$HYPRE_OPENMP)"
        hypre_omp_flag=""
    else
        echo "-- HYPRE OpenMP enabled (HYPRE_ENABLE_OPENMP=ON)"
    fi
    cmake -D HYPRE_BUILD_TESTS=OFF \
          -D HYPRE_BUILD_EXAMPLES=OFF \
          -D HYPRE_ENABLE_MPI=OFF \
          ${hypre_omp_flag} \
          ${thirdparty_compiler_flags} \
          -D CMAKE_BUILD_TYPE=${config} \
          -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
          -D CMAKE_INSTALL_PREFIX=../../../install \
          .. &> ../../../../make_hypre.log
    make install -j $NT >> ../../../../make_hypre.log 2>&1
    cd ../../../
    echo -e "\n--- Building Hypre: DONE!\n"

    echo -e "\n-- Install SuperLU: START\n"
    # Build upstream SuperLU (pinned git submodule thirdparty/superlu) with its
    # own CMake and install into thirdparty/install -- the same prefix and
    # pattern as HYPRE above. Notes on the options:
    #   * double precision only (enable_single/complex/complex16 OFF) -- matches
    #     the previous vendored `make double` behaviour; the wrapper only calls
    #     the d* routines.
    #   * enable_internal_blaslib=ON builds SuperLU's bundled reference CBLAS, so
    #     no system BLAS is required and the build stays self-contained/hermetic
    #     (functionally identical to the old vendored libblas.a). enable_blaslib=ON
    #     is the companion flag: SuperLU v7.0.1's superluConfig.cmake.in templates
    #     @enable_blaslib@ but the build only defines enable_internal_blaslib, so
    #     without this the installed CONFIG package wrongly takes the
    #     find_dependency(BLAS) branch and find_package(superlu) fails on the
    #     missing internal `blas` target.
    #   * enable_fortran/tests/examples OFF -- SuperLU is pure C; we need none of
    #     these (also keeps macOS/Apple Clang happy, no gfortran needed).
    #   * XSDK_INDEX_SIZE=32 keeps int_t == int. The C++ wrapper allocates int[]
    #     for perm_r/perm_c and passes opendarts::config::index_t (== int); 64-bit
    #     indexing would silently break those call sites.
    #   * PIC ON + static so the archive embeds into the shared opendarts_linear_solvers
    #     Python extension.
    rm -rf build/superlu
    mkdir -p build/superlu
    cd build/superlu
    cmake -D enable_single=OFF \
          -D enable_complex=OFF \
          -D enable_complex16=OFF \
          -D enable_double=ON \
          -D enable_internal_blaslib=ON \
          -D enable_blaslib=ON \
          -D enable_fortran=OFF \
          -D enable_tests=OFF \
          -D enable_examples=OFF \
          -D XSDK_INDEX_SIZE=32 \
          -D BUILD_SHARED_LIBS=OFF \
          ${thirdparty_compiler_flags} \
          -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
          -D CMAKE_BUILD_TYPE=${config} \
          -D CMAKE_INSTALL_PREFIX=../../install \
          ../../superlu &> ../../../make_superlu.log
    make install -j $NT >> ../../../make_superlu.log 2>&1
    cd ../../../
    echo -e "\n--- Building SuperLU: DONE!\n"

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
        if [[ "$clean_mode" == true ]]; then
            rm -rf build/iphreeqc
        fi
        mkdir -p build/iphreeqc && cd build/iphreeqc
        cmake \
            -D CMAKE_INSTALL_PREFIX=../../install/iphreeqc \
            -D BUILD_TESTING=OFF \
            -D BUILD_SHARED_LIBS=ON \
            ${thirdparty_compiler_flags} \
            ../../iphreeqc            &> ../../../make_iphreeqc.log
        make install -j $NT           >> ../../../make_iphreeqc.log 2>&1
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

# Setup build with cmake
cmake_options="-D CMAKE_BUILD_TYPE=${config}"

if [[ "$valgrind" = true ]]; then
    cmake_options+=" -D ENABLE_VALGRIND=ON"
fi
if [[ "$testing" == true ]]; then
    cmake_options+=" -D ENABLE_TESTING=ON"
fi
if [[ "$special_gpp" == true ]]; then
    # gcc_version was derived from gpp_version earlier (e.g. g++-14 -> gcc-14),
    # alongside thirdparty_compiler_flags used for the HYPRE/SuperLU/IPhreeqc builds.
    cmake_options+=" -D CMAKE_CXX_COMPILER=${gpp_version} -D CMAKE_C_COMPILER=${gcc_version}"
fi

build=ST
if [[ "$GPU" == true ]]; then
  build=GPU
elif [[ "$MT" == true ]]; then
  build=MT
fi
cmake_options+=" -D OPENDARTS_CONFIG=$build"

if [[ ! -z "$bos_solvers_dir" ]]; then
    # ENABLE_BOS_SOLVERS is the CMake switch (default OFF -> in-tree
    # open-source solvers); BOS_SOLVERS_DIR carries the library location.
    cmake_options+=" -D ENABLE_BOS_SOLVERS=ON -D BOS_SOLVERS_DIR=${bos_solvers_dir}"
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

# install the Python package; -t keeps pytest and future test dependencies in one place
install_target="."
if [[ "$install_test_extra" == true ]]; then
    install_target=".[test]"
fi
python3 -m pip install "$install_target" 2>&1 | tee -a make_wheel.log

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

  # (component_name, log_file) pairs for the thirdparty libraries this script
  # builds in their own make invocation, each with a log of its own.
  local components=(
    "Hypre:make_hypre.log"
    "SuperLU:make_superlu.log"
    "IPhreeqc:make_iphreeqc.log"
  )

  # AMGX has no log of its own: the main CMake build compiles it in-tree
  # (add_subdirectory(thirdparty/AMGX) in the top-level CMakeLists), so its
  # diagnostics land in make_darts.log next to ours and have to be split out of
  # that single log. Two reliable markers do it: a thirdparty/AMGX path (AMGX
  # sources and headers) and the amgx:: namespace (AMGX compiles thrust/cub
  # under THRUST_CUB_WRAPPED_NAMESPACE=amgx, so its template-instantiation
  # warnings -- reported against nvcc intermediate stub files outside the source
  # tree -- carry 'amgx::'; open-DARTS uses plain thrust::, never amgx::).
  local amgx_re='thirdparty/AMGX|amgx::'
  # Anything else under thirdparty/ (pybind11 / MshIO headers pulled into our
  # TUs, ...) is likewise not open-DARTS code and must not gate CI on our count.
  local nonproject_re='thirdparty/|amgx::'

  # Count first, print second. Every component is counted here, in one pass, and
  # the block below only formats what this pass produced: that block is piped
  # through `tee -a make_darts.log`, so a grep run from inside it would read the
  # very log the summary is being appended to.
  local rows=()
  local entry name logfile warn_count err_count

  for entry in "${components[@]}"; do
    name="${entry%%:*}"
    logfile="${entry##*:}"
    [[ -f "$logfile" ]] || continue
    warn_count=$(grep -cE "$warn_pattern" "$logfile" 2>/dev/null || true)
    err_count=$(grep -cE "$err_pattern" "$logfile" 2>/dev/null || true)
    rows+=("$(printf " %-14s | %8d | %6d" "$name" "$warn_count" "$err_count")")
  done

  # make_darts.log is shared by three sets of diagnostics -- AMGX, other
  # thirdparty headers compiled into our TUs, and open-DARTS itself -- so it is
  # classified by the regexes above rather than counted as a whole.
  local darts_warnings=0 darts_errors=0
  local amgx_warnings=0 amgx_errors=0 amgx_unique=0
  local other_tp_warnings=0 other_tp_errors=0
  if [[ -f make_darts.log ]]; then
    darts_warnings=$(grep -E "$warn_pattern" make_darts.log 2>/dev/null | grep -Ecv "$nonproject_re" || true)
    darts_errors=$(grep -E "$err_pattern" make_darts.log 2>/dev/null | grep -Ecv "$nonproject_re" || true)
    amgx_warnings=$(grep -E "$warn_pattern" make_darts.log 2>/dev/null | grep -Ec "$amgx_re" || true)
    amgx_errors=$(grep -E "$err_pattern" make_darts.log 2>/dev/null | grep -Ec "$amgx_re" || true)
    amgx_unique=$(grep -E "$warn_pattern" make_darts.log 2>/dev/null | grep -E "$amgx_re" | sort -u | wc -l | tr -d ' ')
    # thirdparty but not AMGX -- counted so that no diagnostic in make_darts.log
    # falls between the open-DARTS and AMGX rows and goes unnoticed.
    other_tp_warnings=$(grep -E "$warn_pattern" make_darts.log 2>/dev/null | grep -E "$nonproject_re" | grep -Ecv "$amgx_re" || true)
    other_tp_errors=$(grep -E "$err_pattern" make_darts.log 2>/dev/null | grep -E "$nonproject_re" | grep -Ecv "$amgx_re" || true)
  fi

  # AMGX is built for GPU configurations only (WITH_AMGX defaults ON there, and
  # -G initialises the submodule); report its row whenever it took part in this
  # build, including a clean 0 | 0, so the table shows it was actually checked.
  if [[ ( "$GPU" == true && "${OD_CMAKE_ARGS:-}" != *"WITH_AMGX=OFF"* ) || $amgx_warnings -gt 0 || $amgx_errors -gt 0 ]]; then
    rows+=("$(printf " %-14s | %8d | %6d" "AMGX" "$amgx_warnings" "$amgx_errors")")
  fi
  if [[ $other_tp_warnings -gt 0 || $other_tp_errors -gt 0 ]]; then
    rows+=("$(printf " %-14s | %8d | %6d" "other 3rdparty" "$other_tp_warnings" "$other_tp_errors")")
  fi
  if [[ -f make_darts.log ]]; then
    rows+=("$(printf " %-14s | %8d | %6d" "open-DARTS" "$darts_warnings" "$darts_errors")")
  fi

  {
    echo ""
    echo "========================================="
    echo " Build warnings/errors summary"
    echo "========================================="
    printf " %-14s | %8s | %6s\n" "Component" "Warnings" "Errors"
    echo " -----------------------------------------"

    local row
    for row in "${rows[@]}"; do
      echo "$row"
    done

    echo "========================================="

    if [[ $darts_warnings -gt 0 ]]; then
      echo ""
      echo " open-DARTS unique warnings:"
      # Same thirdparty exclusion as the count above (thirdparty/ paths + amgx:: stubs).
      # `|| true`: head closing the pipe early (more unique warnings than the cap)
      # makes the upstream grep fail, which under `set -e -o pipefail` would abort
      # this subshell before OPENDARTS_WARNING_COUNT below is printed.
      grep -E "$warn_pattern" make_darts.log 2>/dev/null | grep -Ev "$nonproject_re" | sort -u | head -100 || true
    fi

    if [[ $amgx_warnings -gt 0 ]]; then
      echo ""
      echo " AMGX warnings ($amgx_unique unique, thirdparty code -- not gating), first 10:"
      grep -E "$warn_pattern" make_darts.log 2>/dev/null | grep -E "$amgx_re" | sort -u | head -10 || true
    fi

    echo ""
    # Only open-DARTS warnings gate CI (.cicd/jobs/*.yml parse this one line);
    # thirdparty counts stay in the table above.
    echo "OPENDARTS_WARNING_COUNT=$darts_warnings"
  } | tee -a make_darts.log
}
report_build_summary
# ------------------------------------------------------------------------------
