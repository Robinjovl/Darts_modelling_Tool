#!/bin/bash
set -e

# get linear solvers binary compiled with GPU and include files
cd engines/lib
mkdir darts_linear_solvers && cd darts_linear_solvers && mkdir lib && mkdir include && cd ..
cp -r /oahu/data/open-darts-gitlab-runner-data/darts-linear-solvers-gpu/lib darts_linear_solvers
cp -r /oahu/data/open-darts-gitlab-runner-data/darts-linear-solvers-gpu/include darts_linear_solvers
cd ../..

# compile discretizer using the Makefile (no GPU)
cd discretizer
set +e # temporarily turn off set -e
make release -j 20 USE_OPENDARTS_LINEAR_SOLVERS=false 1>../make_discretizer_out.log 2>../make_discretizer_err.log
# sometimes the command above fails for file discretizer_build_info.cpp.in, so run it twice
make release USE_OPENDARTS_LINEAR_SOLVERS=false 1>>../make_discretizer_out.log 2>>../make_discretizer_err.log
cd ..

# need to link engines
cd engines
cp ../darts/discretizer.so .

# compile engines using the Makefile
make clean
make gpu -j 20 USE_OPENDARTS_LINEAR_SOLVERS=false 1>../make_engines_out.log 2>../make_engines_err.log
cd ..

# to add amgx shared library to wheels
cp -v ./engines/lib/darts_linear_solvers/lib/libamgxsh.so ./darts

# build DARTS wheel
./helper_scripts/build_install_darts.sh

