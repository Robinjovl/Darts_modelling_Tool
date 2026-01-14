#!/bin/bash
set -e

if [[ "$GSELINSOLVERSPATH" == "" ]]; then
  echo "Error: the environment variable GSELINSOLVERSPATH is not defined!"
  exit 1
fi

CLEAN_FLAG=""
PHREEQC_FLAG=""
DEBUG_FLAG=""
JOBS_ARG="-j20"

# Scan all args, including -j for parallel jobs
while (( "$#" )); do
  case "$1" in
    -c) CLEAN_FLAG="-c"; shift ;;        # trigger clean
    -p) PHREEQC_FLAG="-p"; shift ;;      # enable IPhreeqc/Reaktoro support
    -d) DEBUG_FLAG="-d Debug"; shift ;;  # enable Debug configuration
    -r) REQUIREMENTS_FLAG="-r"; shift ;; # clean previous cmake configuration for third parties
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

./helper_scripts/build_darts_cmake.sh \
  -G \
  $JOBS_ARG \
  -b $GSELINSOLVERSPATH \
  -w \
  $CLEAN_FLAG \
  $PHREEQC_FLAG \
  $DEBUG_FLAG \
  $REQUIREMENTS_FLAG
