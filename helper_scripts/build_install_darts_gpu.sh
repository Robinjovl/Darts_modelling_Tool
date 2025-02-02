#!/bin/bash
set -e

if [[ "$GSELINSOLVERSPATH" == "" ]]; then
  echo "Error: the environment variable GSELINSOLVERSPATH is not defined!"
  exit 1
fi

CLEAN_FLAG=""
for arg in "$@"; do
  if [[ "$arg" == "-c" ]]; then
    CLEAN_FLAG="-c"
    break
  fi
done

./helper_scripts/build_darts_cmake.sh -G -j20 -b $GSELINSOLVERSPATH -w $CLEAN_FLAG

