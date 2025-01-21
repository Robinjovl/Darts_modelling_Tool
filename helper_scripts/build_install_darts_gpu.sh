#!/bin/bash
set -e

# get linear solvers binary compiled with GPU and include files
# cd engines/lib
# rm -rf darts_linear_solvers
# mkdir darts_linear_solvers && cd darts_linear_solvers && mkdir lib && mkdir include && cd ..

if [[ "$GSELINSOLVERSPATH" == "" ]]; then
  echo "Error: the environment variable GSELINSOLVERSPATH is not defined!"
  exit 1
fi
# cp -r $GSELINSOLVERSPATH/lib darts_linear_solvers
# cp -r $GSELINSOLVERSPATH/include darts_linear_solvers
# cd ../..

mkdir -p build
cd build
cmake .. -DBOS_SOLVERS_DIR=$GSELINSOLVERSPATH -DOPENDARTS_CONFIG=GPU
make install
cd ..

# to add amgx shared library to wheels
cp -v ./engines/lib/darts_linear_solvers/lib/libamgxsh.so ./darts

# build DARTS wheel
./helper_scripts/build_install_darts.sh

