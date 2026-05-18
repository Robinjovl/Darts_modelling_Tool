#!/bin/bash
set -e

# Build and install openDARTS with GPU support.
#
# By default the GPU build uses the in-tree open-source solvers (darts.solvers,
# including the GPU solver wrappers). To build against the proprietary
# bos_solvers library instead, pass:
#   -b <path>   explicit path to the bos_solvers directory
#   -b          (no path) falls back to the GSELINSOLVERSPATH environment
#               variable, for backward compatibility with the old workflow

CLEAN_FLAG=""
PHREEQC_FLAG=""
DEBUG_FLAG=""
REQUIREMENTS_FLAG=""
BOS_FLAG=""
JOBS_ARG="-j20"

# Scan all args, including -j for parallel jobs
while (( "$#" )); do
  case "$1" in
    -c) CLEAN_FLAG="-c"; shift ;;        # trigger clean
    -p) PHREEQC_FLAG="-p"; shift ;;      # enable IPhreeqc/Reaktoro support
    -d) DEBUG_FLAG="-d Debug"; shift ;;  # enable Debug configuration
    -r) REQUIREMENTS_FLAG="-r"; shift ;; # clean previous cmake configuration for third parties
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
  echo "openDARTS GPU build: using the in-tree open-source solvers (darts.solvers)"
fi

./helper_scripts/build_darts_cmake.sh \
  -G \
  $JOBS_ARG \
  $BOS_FLAG \
  -w \
  $CLEAN_FLAG \
  $PHREEQC_FLAG \
  $DEBUG_FLAG \
  $REQUIREMENTS_FLAG
