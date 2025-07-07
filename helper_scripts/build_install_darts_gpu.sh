#!/bin/bash
set -e

if [[ "$GSELINSOLVERSPATH" == "" ]]; then
  echo "Error: the environment variable GSELINSOLVERSPATH is not defined!"
  exit 1
fi

CLEAN_FLAG=""
PHREEQC_FLAG=""
DEBUG_FLAG=""

# Scan all args for -c and -p
for arg in "$@"; do
  case "$arg" in
    -c) CLEAN_FLAG="-c"         ;;  # trigger clean
    -p) PHREEQC_FLAG="-p"       ;;  # enable IPhreeqc support
    -d) DEBUG_FLAG="-d Debug"   ;;  # enable Debug configuration
  esac
done

./helper_scripts/build_darts_cmake.sh \
  -G \
  -j20 \
  -b $GSELINSOLVERSPATH \
  -w \
  $CLEAN_FLAG \
  $PHREEQC_FLAG \
  $DEBUG_FLAG

