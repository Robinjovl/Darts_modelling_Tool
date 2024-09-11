#!/bin/bash

# get linear solvers binary compiled with GPU and include files
cd engines/lib
mkdir darts_linear_solvers && cd darts_linear_solvers && mkdir lib && mkdir include && cd ..
cp -r /oahu/data/open-darts-gitlab-runner-data/darts-linear-solvers-gpu/lib darts_linear_solvers/lib
cp -r /oahu/data/open-darts-gitlab-runner-data/darts-linear-solvers-gpu/include darts_linear_solvers/include
cd ../..

# compile discretizer using the Makefile (no GPU)
cd discretizer
make release -j 20 USE_OPENDARTS_LINEAR_SOLVERS=false 2>&1 | tee ../make_discretizer.log
# sometimes the command above fails for file discretizer_build_info.cpp.in, so run it twice
make release USE_OPENDARTS_LINEAR_SOLVERS=false | tee -a ../make_discretizer.log
cd ..

# need to link engines
cd engines
cp ../darts/discretizer.so .

# compile engines using the Makefile
make clean
make gpu -j 20 USE_OPENDARTS_LINEAR_SOLVERS=false 2>&1 | tee ../make_engines.log
cd ..

# to add amgx shared library to wheels
cp ../darts-linear-solvers/lib/AMGX/build/libamgxsh.so ./darts

# build DARTS wheel
./helper_scripts/build_install_darts.sh

