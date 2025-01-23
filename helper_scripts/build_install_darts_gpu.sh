#!/bin/bash
set -e

if [[ "$GSELINSOLVERSPATH" == "" ]]; then
  echo "Error: the environment variable GSELINSOLVERSPATH is not defined!"
  exit 1
fi

./helper_scripts/build_darts_cmake.sh -G -j20 -b $GSELINSOLVERSPATH -w
