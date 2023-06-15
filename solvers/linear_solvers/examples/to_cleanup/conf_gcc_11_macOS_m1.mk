CXX         = g++-11
CXXFLAGS    = -O3 -march=native -fPIC -std=c++11 -mmacosx-version-min=12.2
CXXFLAGS_DEBUG = -O0 -g -rdynamic -fPIC -std=c++11 -mmacosx-version-min=12.2

CC          = gcc-11
CFLAGS      = -O3 -march=native -fPIC

ARCH        = /usr/bin/ar
ARCHFLAGS   = cr
RANLIB      = /usr/bin/ranlib

OMP_FLAG 	= -fopenmp

DARTS_LINEAR_SOLVERS = /Users/apalha/work/dev/onset/darts/darts-engines/lib/darts_linear_solvers
