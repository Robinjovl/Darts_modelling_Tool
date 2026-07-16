#!/bin/bash
set -e

# Build and install openDARTS with GPU support.
#
# By default the GPU build uses the in-tree open-source solvers (darts.linear_solvers,
# including the GPU solver wrappers). To build against the proprietary
# bos_solvers library instead, pass:
#   -b <path>   explicit path to the bos_solvers directory
#   -b          (no path) falls back to the GSELINSOLVERSPATH environment
#               variable, for backward compatibility with the old workflow
#
# The AMGX GPU solver is ALWAYS built (CMake WITH_AMGX defaults ON): it backs the
# default GPU linear solver (GPU_GMRES_CPR_AMGX_ILU), so it is not optional and
# this script exposes no flag to disable it. The thirdparty/AMGX submodule is
# initialised automatically by build_darts_cmake.sh for GPU builds.
#
# The cuDSS GPU direct solver is ON by default (CMake WITH_CUDSS). cuDSS is a
# prebuilt C++ dependency (cudss.h + libcudss.so) -- NOT a Python runtime
# dependency; the nvidia-cudss-cu13 pip wheel is merely one way to deliver that
# C++ library into the build env. thirdparty/thirdparty_cudss.cmake locates it
# from any of: CUDSS_ROOT=<extracted redistributable> (the pure-C++ route),
# a cudss-config.cmake on CMAKE_PREFIX_PATH, or the pip-wheel layout. If it is
# not found the GPU stack is built without cuDSS (warning, not error).
#   --no-cudss  disable the cuDSS solver for this build
#   --cudss     force-enable (redundant -- it is the default; kept for clarity)

CLEAN_FLAG=""
PHREEQC_FLAG=""
DEBUG_FLAG=""
REQUIREMENTS_FLAG=""
BOS_FLAG=""
TEST_FLAG=""
JOBS_ARG="-j20"

# Scan all args, including -j for parallel jobs
while (( "$#" )); do
  case "$1" in
    -c) CLEAN_FLAG="-c"; shift ;;        # trigger clean
    -p) PHREEQC_FLAG="-p"; shift ;;      # enable IPhreeqc/Reaktoro support
    -d) DEBUG_FLAG="-d Debug"; shift ;;  # enable Debug configuration
    -t) TEST_FLAG="-t"; shift ;;         # enable CTest and install test dependencies
    -r) REQUIREMENTS_FLAG="-r"; shift ;; # clean previous cmake configuration for third parties
    # NB: AMGX is always built (CMake WITH_AMGX defaults ON) -- there is
    # deliberately no --amgx/--no-amgx flag; it is not user-optional.
    --cudss)
      # cuDSS GPU direct solver -- ON by default (CMake WITH_CUDSS); this flag
      # force-enables it explicitly (redundant, kept for clarity / back-compat).
      export OD_CMAKE_ARGS="${OD_CMAKE_ARGS:-} -D WITH_CUDSS=ON"; shift ;;
    --no-cudss)
      # Disable the cuDSS GPU direct solver (prebuilt NVIDIA library not wanted /
      # not available -- e.g. CI builds that should not pull it in).
      export OD_CMAKE_ARGS="${OD_CMAKE_ARGS:-} -D WITH_CUDSS=OFF"; shift ;;
    -b)
      # Opt in to the proprietary bos_solvers. An explicit path may follow;
      # otherwise fall back to $GSELINSOLVERSPATH.
      if [[ -n "${2:-}" && "$2" != -* ]]; then
        BOS_PATH="$2"; shift 2
      else
        BOS_PATH="${GSELINSOLVERSPATH:-}"; shift
      fi
      if [[ -z "$BOS_PATH" ]]; then
        echo "Error: -b given without a path and GSELINSOLVERSPATH is not set"
        exit 1
      fi
      BOS_FLAG="-b $BOS_PATH"
      ;;
    -j)
      if [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]]; then
        JOBS_ARG="-j$2"
        shift 2
      else
        echo "Error: -j requires a numeric argument"
        exit 1
      fi
      ;;
    -j*)
      local_jobs="${1#-j}"
      if [[ "$local_jobs" =~ ^[0-9]+$ ]]; then
        JOBS_ARG="-j$local_jobs"
        shift
      else
        echo "Error: invalid -j value: $1"
        exit 1
      fi
      ;;
    *)
      echo "Warning: ignoring unknown argument: $1"
      shift
      ;;
  esac
done

if [[ -n "$BOS_FLAG" ]]; then
  echo "openDARTS GPU build: using bos_solvers at ${BOS_FLAG#-b }"
else
  echo "openDARTS GPU build: using the in-tree open-source solvers (darts.linear_solvers)"
fi

./helper_scripts/build_darts_cmake.sh \
  -G \
  $JOBS_ARG \
  $BOS_FLAG \
  -w \
  $CLEAN_FLAG \
  $PHREEQC_FLAG \
  $DEBUG_FLAG \
  $REQUIREMENTS_FLAG \
  $TEST_FLAG
