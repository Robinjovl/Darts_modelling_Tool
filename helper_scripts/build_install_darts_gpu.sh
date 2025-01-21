#!/bin/bash
set -e

# get linear solvers binary compiled with GPU and include files

if [[ "$GSELINSOLVERSPATH" == "" ]]; then
  echo "Error: the environment variable GSELINSOLVERSPATH is not defined!"
  exit 1
fi

mkdir -p build
cd build
rm -f CMakeCache.txt 
cmake .. -DBOS_SOLVERS_DIR=$GSELINSOLVERSPATH -DOPENDARTS_CONFIG=GPU
make install
cd ..

# to add amgx shared library to wheels
cp -v $GSELINSOLVERSPATH/lib/libamgxsh.so ./darts

# build DARTS wheel
./helper_scripts/build_install_darts.sh

