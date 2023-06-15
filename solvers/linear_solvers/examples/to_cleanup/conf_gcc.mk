CXX         = g++
CXXFLAGS    = -O3 -march=native -fPIC -std=c++11
CXXFLAGS_DEBUG = -O0 -g -rdynamic -fPIC -std=c++11

CC          = gcc
CFLAGS      = -O3 -march=native -fPIC

ARCH        = /usr/bin/ar
ARCHFLAGS   = cr
RANLIB      = /usr/bin/ranlib

OMP_FLAG 	= -fopenmp

DARTS_LINEAR_SOLVERS = /Users/apalha/work/dev/onset/darts/darts-engines/lib/darts_linear_solvers
