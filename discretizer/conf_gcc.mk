CXX         	= g++
CXXFLAGS    	= -O3 -fPIC -march=native -std=c++20 -Wno-deprecated-declarations -Wno-attributes
CXXFLAGS_DEBUG	=  -O0 -g -rdynamic -fPIC -march=native -std=c++20 -Wno-deprecated-declarations -Wno-attributes

CC          	= gcc
CFLAGS      	= -O3 -march=native -fPIC -std=c11

ARCH        	= /usr/bin/ar
ARCHFLAGS   	= cr
RANLIB      	= /usr/bin/ranlib

OMP_FLAG    	= -fopenmp

# Setup build for using opendarts-linear-solvers or deprecated version of linear-solvers
ifndef $(USE_OPENDARTS_LINEAR_SOLVERS)
USE_OPENDARTS_LINEAR_SOLVERS = true
endif

ifeq ($(USE_OPENDARTS_LINEAR_SOLVERS), true)
$(info Building darts-engines with solvers)
CXXFLAGS += -D OPENDARTS_LINEAR_SOLVERS
CXXFLAGS_DEBUG += -D OPENDARTS_LINEAR_SOLVERS
LINEAR_SOLVERS_DIR = ../engines/lib/solvers
else
$(info Building darts-engines with bos solvers)
LINEAR_SOLVERS_DIR = ../engines/lib/darts_linear_solvers
endif